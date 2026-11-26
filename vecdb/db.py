"""VectorDB: the database itself.

Two segments hold the data:
  * sealed segment: immutable, indexed by IVF (built by build_index / compact);
  * delta segment:  vectors inserted since the last build, append-only, searched by flat scan.
Deletes only clear a bit in an `alive` bitmap (tombstones). compact() folds the delta into
the sealed segment and physically drops dead rows.

Durability: every insert/delete is appended to the write-ahead log before it is applied.
checkpoint() writes a snapshot and starts a new log. open() loads the snapshot named by the
CURRENT file and replays the log on top of it.
"""
import json
import os
import shutil
import time

import numpy as np

from .flat import flat_scan, squared_norms, topk
from .ivf import IVFIndex
from .kmeans import assign
from .wal import OP_DELETE, OP_INSERT, WAL, replay, sync_fd

CURRENT = "CURRENT"


def _fsync_dir(path):
    """Make a rename/create inside `path` durable (directory entries are data too)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _save(path, arr, full):
    with open(path, "wb") as f:
        np.save(f, arr)
        f.flush()
        sync_fd(f.fileno(), full)


class VectorDB:
    def __init__(self, path, dim, wal_opts):
        self.path = path
        self.dim = dim
        self._wal_opts = wal_opts
        self.seq = 0                 # sequence number of the last applied operation
        self.gen = 0                 # checkpoint generation: names the snapshot and the log
        self.id2loc = {}             # id -> row (>= 0: sealed row r; < 0: delta row -loc-1)
        self.ivf = None
        self.S_ids = np.empty(0, dtype=np.int64)
        self.S_alive = np.empty(0, dtype=bool)
        self.S_dead = 0
        self.default_nprobe = 1
        self.last_scanned = 0        # rows scanned by the last search (for the benchmarks)
        self.recovery = {}
        self.wal = None
        self._reset_delta()

    # ------------------------------------------------------------------ open / recovery
    @classmethod
    def open(cls, path, dim, sync="batch", full_sync=False, wal_group=64, wal_group_ms=50.0):
        """Open or create a database directory and recover its state."""
        db = cls(path, dim, dict(sync=sync, full_sync=full_sync, group=wal_group, group_ms=wal_group_ms))
        os.makedirs(path, exist_ok=True)
        t0 = time.perf_counter()
        cur = db._read_current()
        if cur:                                   # 1) load the last checkpoint
            db.gen = cur["gen"]
            db._load_snapshot(os.path.join(path, cur["snapshot"]))
        t1 = time.perf_counter()
        wal_path = db._wal_path(db.gen)
        replayed = torn = 0
        if os.path.exists(wal_path):              # 2) replay what happened after it

            def apply(seq, op, ids, vecs):
                nonlocal replayed
                if seq <= db.seq:                 # already inside the snapshot: skip (idempotent)
                    return
                if op == OP_INSERT:
                    db._apply_insert(ids, vecs)
                else:
                    db._apply_delete(ids)
                db.seq = seq
                replayed += 1

            good = replay(wal_path, dim, apply)
            size = os.path.getsize(wal_path)
            if good < size:                       # 3) cut off a torn tail
                torn = size - good
                os.truncate(wal_path, good)
        db.recovery = dict(snapshot_s=t1 - t0, replay_s=time.perf_counter() - t1,
                           replayed_records=replayed, torn_bytes=torn)
        db._cleanup()
        db.wal = WAL(wal_path, **db._wal_opts)
        return db

    def close(self):
        if self.wal is not None:
            self.wal.close()
            self.wal = None

    # ------------------------------------------------------------------ writes
    def insert(self, id, vec):
        """Insert one vector. An existing id is replaced."""
        self.insert_many([id], np.asarray(vec, dtype=np.float32).reshape(1, -1))

    def insert_many(self, ids, vecs):
        """Insert a batch as ONE log record (one write, at most one sync)."""
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if vecs.shape != (len(ids), self.dim):
            raise ValueError(f"expected vectors of shape ({len(ids)}, {self.dim}), got {vecs.shape}")
        seq = self.seq + 1
        self.wal.append(seq, OP_INSERT, ids, vecs)   # log first ...
        self._apply_insert(ids, vecs)                # ... then change memory
        self.seq = seq

    def delete(self, id):
        """Delete by id. Returns False if the id is not present (nothing is logged)."""
        if id not in self.id2loc:
            return False
        ids = np.array([id], dtype=np.int64)
        seq = self.seq + 1
        self.wal.append(seq, OP_DELETE, ids)
        self._apply_delete(ids)
        self.seq = seq
        return True

    def _reset_delta(self, capacity=1024):
        self.D_vecs = np.empty((capacity, self.dim), dtype=np.float32)
        self.D_norms = np.empty(capacity, dtype=np.float32)
        self.D_ids = np.empty(capacity, dtype=np.int64)
        self.D_alive = np.zeros(capacity, dtype=bool)
        self.D_n = 0
        self.D_dead = 0

    def _reserve_delta(self, extra):
        need = self.D_n + extra
        cap = len(self.D_ids)
        if need <= cap:
            return
        cap = max(need, 2 * cap)                     # doubling: amortised O(1) per insert
        n = self.D_n
        for name in ("D_vecs", "D_norms", "D_ids", "D_alive"):
            old = getattr(self, name)
            new = np.zeros((cap,) + old.shape[1:], dtype=old.dtype)
            new[:n] = old[:n]
            setattr(self, name, new)

    def _kill(self, loc):
        """Tombstone: clear the alive bit, leave the row where it is."""
        if loc >= 0:
            self.S_alive[loc] = False
            self.S_dead += 1
        else:
            self.D_alive[-loc - 1] = False
            self.D_dead += 1

    def _apply_insert(self, ids, vecs):
        n = len(ids)
        self._reserve_delta(n)
        start = self.D_n
        self.D_vecs[start:start + n] = vecs
        self.D_norms[start:start + n] = squared_norms(vecs)
        self.D_ids[start:start + n] = ids
        self.D_alive[start:start + n] = True
        id2loc = self.id2loc
        for j, i in enumerate(ids.tolist()):
            old = id2loc.get(i)
            if old is not None:                      # replace: the old version becomes a tombstone
                self._kill(old)
            id2loc[i] = -(start + j) - 1
        self.D_n += n

    def _apply_delete(self, ids):
        for i in ids.tolist():
            loc = self.id2loc.pop(i, None)
            if loc is not None:
                self._kill(loc)

    # ------------------------------------------------------------------ reads
    def __len__(self):
        return len(self.id2loc)

    def __contains__(self, id):
        return id in self.id2loc

    def get(self, id):
        loc = self.id2loc[id]
        return self.ivf.vecs[loc].copy() if loc >= 0 else self.D_vecs[-loc - 1].copy()

    def search(self, q, k=10, nprobe=None, exact=False):
        """k nearest neighbours of q by L2 distance. Returns (ids, distances), nearest first.

        The sealed segment is searched through the IVF index (nprobe lists) unless
        exact=True or nprobe covers every list, in which case a flat scan is cheaper and
        exact. The delta segment is always scanned exactly. Results are merged.
        """
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query has dimension {q.shape[0]}, expected {self.dim}")
        cand_v, cand_id = [], []
        scanned = 0
        if self.ivf is not None and len(self.S_ids):
            alive = self.S_alive if self.S_dead else None
            nprobe = nprobe or self.default_nprobe
            if exact or nprobe >= self.ivf.nlist:       # tiny planner: full scan beats probing all lists
                dist = flat_scan(self.ivf.vecs, self.ivf.norms, q, alive)
                rows = topk(dist, k)
                scanned += len(dist)
            else:
                rows, _, n_scanned = self.ivf.search(q, k, nprobe, alive)
                scanned += n_scanned
            cand_v.append(self.ivf.vecs[rows])
            cand_id.append(self.S_ids[rows])
        n = self.D_n
        if n:
            alive = self.D_alive[:n] if self.D_dead else None
            rows = topk(flat_scan(self.D_vecs[:n], self.D_norms[:n], q, alive), k)
            cand_v.append(self.D_vecs[rows])
            cand_id.append(self.D_ids[rows])
            scanned += n
        self.last_scanned = scanned
        if not cand_v:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        ids = np.concatenate(cand_id)
        # Merge step. The scan ranks with ||x||^2 - 2 x.q, which is fast but loses a few
        # float32 digits to cancellation. For the at most 2k candidates we recompute the
        # true distance ||x - q|| directly, then keep the best k.
        diff = np.concatenate(cand_v) - q
        dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        order = np.argsort(dists, kind="stable")[:k]
        return ids[order], dists[order]

    # ------------------------------------------------------------------ index maintenance
    def _gather_alive(self):
        parts_v, parts_i = [], []
        if self.ivf is not None and len(self.S_ids):
            if self.S_dead:
                parts_v.append(self.ivf.vecs[self.S_alive])
                parts_i.append(self.S_ids[self.S_alive])
            else:
                parts_v.append(self.ivf.vecs)
                parts_i.append(self.S_ids)
        n = self.D_n
        if n:
            if self.D_dead:
                keep = self.D_alive[:n]
                parts_v.append(self.D_vecs[:n][keep])
                parts_i.append(self.D_ids[:n][keep])
            else:
                parts_v.append(self.D_vecs[:n])
                parts_i.append(self.D_ids[:n])
        if not parts_v:
            return np.empty((0, self.dim), dtype=np.float32), np.empty(0, dtype=np.int64)
        if len(parts_v) == 1:
            return parts_v[0], parts_i[0]
        return np.concatenate(parts_v), np.concatenate(parts_i)

    def _install(self, index, ids):
        """Swap in a new sealed segment; the delta and all tombstones are gone."""
        self.ivf = index
        self.S_ids = np.ascontiguousarray(ids)
        self.S_alive = np.ones(len(ids), dtype=bool)
        self.S_dead = 0
        self._reset_delta()
        self.id2loc = dict(zip(self.S_ids.tolist(), range(len(ids))))

    def build_index(self, nlist, nprobe=None, sample_per_list=64, iters=10, seed=0, checkpoint=True):
        """Train k-means on all alive vectors and build a fresh sealed IVF segment."""
        vecs, ids = self._gather_alive()
        if len(ids) < nlist:
            raise ValueError(f"need at least nlist={nlist} vectors, have {len(ids)}")
        index, order = IVFIndex.build(vecs, nlist, sample_per_list=sample_per_list, iters=iters, seed=seed)
        self._install(index, ids[order])
        self.default_nprobe = nprobe or max(1, round(nlist ** 0.5))
        if checkpoint:
            self.checkpoint()
        return index.stats

    def compact(self, retrain=False, checkpoint=True):
        """Fold the delta into the sealed segment and drop dead rows.

        retrain=False keeps the existing centroids (fast; new vectors go to their nearest
        old centroid). retrain=True runs k-means again (slow; centroids follow the data).
        """
        if self.ivf is None:
            raise RuntimeError("no index yet: call build_index first")
        if retrain:
            return self.build_index(self.ivf.nlist, nprobe=self.default_nprobe, checkpoint=checkpoint)
        t0 = time.perf_counter()
        keep = self.S_alive
        n = self.D_n
        keep_d = self.D_alive[:n]
        v_new = self.D_vecs[:n][keep_d]
        a_new = assign(v_new, self.ivf.centroids, self.ivf.cnorms) if len(v_new) else np.empty(0, dtype=np.int32)
        vecs = np.concatenate((self.ivf.vecs[keep], v_new))
        a = np.concatenate((self.ivf.list_ids()[keep], a_new))
        ids = np.concatenate((self.S_ids[keep], self.D_ids[:n][keep_d]))
        index, order = IVFIndex.from_assignment(vecs, a, self.ivf.centroids)
        index.stats["compact_s"] = time.perf_counter() - t0
        self._install(index, ids[order])
        if checkpoint:
            self.checkpoint()
        return index.stats

    # ------------------------------------------------------------------ checkpoint
    def _wal_path(self, gen):
        return os.path.join(self.path, f"wal-{gen:06d}.log")

    def _read_current(self):
        p = os.path.join(self.path, CURRENT)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def checkpoint(self):
        """Write a snapshot of the whole state, then switch to a new empty log.

        Order matters for crash safety:
          1. write the snapshot into a temp directory, sync, rename;
          2. create the new log file;
          3. atomically replace CURRENT so it names the new snapshot (os.replace);
          4. only then delete the old snapshot and the old log.
        A crash before step 3 leaves CURRENT pointing at the old, still complete pair.
        """
        full = self._wal_opts["full_sync"]
        self.wal.sync()
        new_gen = self.gen + 1
        name = f"snap-{new_gen:06d}"
        tmp = os.path.join(self.path, name + ".tmp")
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        n = self.D_n
        arrays = {"D_vecs": self.D_vecs[:n], "D_ids": self.D_ids[:n], "D_alive": self.D_alive[:n]}
        if self.ivf is not None:
            arrays.update(S_vecs=self.ivf.vecs, S_ids=self.S_ids, S_alive=self.S_alive,
                          centroids=self.ivf.centroids, offsets=self.ivf.offsets)
        for key, arr in arrays.items():
            _save(os.path.join(tmp, key + ".npy"), arr, full)
        meta = dict(seq=self.seq, dim=self.dim, has_index=self.ivf is not None,
                    default_nprobe=self.default_nprobe)
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f)
            f.flush()
            sync_fd(f.fileno(), full)
        os.rename(tmp, os.path.join(self.path, name))
        new_wal = self._wal_path(new_gen)
        open(new_wal, "ab").close()
        cur_tmp = os.path.join(self.path, CURRENT + ".tmp")
        with open(cur_tmp, "w") as f:
            json.dump({"gen": new_gen, "snapshot": name}, f)
            f.flush()
            sync_fd(f.fileno(), full)
        os.replace(cur_tmp, os.path.join(self.path, CURRENT))     # the atomic switch
        _fsync_dir(self.path)
        self.wal.close()
        self.gen = new_gen
        self.wal = WAL(new_wal, **self._wal_opts)
        self._cleanup()

    def _load_snapshot(self, d):
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        if meta["dim"] != self.dim:
            raise ValueError(f"database has dim={meta['dim']}, opened with dim={self.dim}")
        self.seq = meta["seq"]
        self.default_nprobe = meta["default_nprobe"]
        load = lambda name: np.load(os.path.join(d, name + ".npy"))
        d_ids = load("D_ids")
        n = len(d_ids)
        self._reset_delta(capacity=max(1024, n))
        if n:
            self.D_vecs[:n] = load("D_vecs")
            self.D_norms[:n] = squared_norms(self.D_vecs[:n])
            self.D_ids[:n] = d_ids
            self.D_alive[:n] = load("D_alive")
        self.D_n = n
        self.D_dead = int(n - self.D_alive[:n].sum())
        self.id2loc = {}
        if meta["has_index"]:
            # the index comes back from disk as it was: no k-means on startup
            self.ivf = IVFIndex(load("centroids"), load("offsets"), load("S_vecs"))
            self.S_ids = load("S_ids")
            self.S_alive = load("S_alive")
            self.S_dead = int(len(self.S_ids) - self.S_alive.sum())
            rows = np.flatnonzero(self.S_alive)
            self.id2loc = dict(zip(self.S_ids[rows].tolist(), rows.tolist()))
        rows = np.flatnonzero(self.D_alive[:n])
        self.id2loc.update(zip(self.D_ids[rows].tolist(), (-rows - 1).tolist()))

    def _cleanup(self):
        """Remove files that CURRENT no longer references (old generations, temp dirs)."""
        keep = {CURRENT, f"snap-{self.gen:06d}", f"wal-{self.gen:06d}.log"}
        for name in os.listdir(self.path):
            if name in keep:
                continue
            p = os.path.join(self.path, name)
            if name.endswith(".tmp") or name.startswith("snap-"):
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
            elif name.startswith("wal-") and name.endswith(".log"):
                os.remove(p)

    # ------------------------------------------------------------------ introspection
    def stats(self):
        s = dict(alive=len(self.id2loc), sealed_rows=len(self.S_ids), sealed_dead=self.S_dead,
                 delta_rows=self.D_n, delta_dead=self.D_dead, seq=self.seq, gen=self.gen,
                 nlist=self.ivf.nlist if self.ivf is not None else 0,
                 default_nprobe=self.default_nprobe)
        if self.ivf is not None:
            s.update(self.ivf.stats)
        return s

    def disk_bytes(self):
        total = 0
        for root, _, files in os.walk(self.path):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return total
