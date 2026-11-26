"""IVF-Flat index: k-means partitions, each partition stored as one contiguous slice."""
import time

import numpy as np

from .flat import squared_norms, topk
from .kmeans import assign, kmeans


class IVFIndex:
    """Immutable inverted-file index.

    Layout: `vecs` holds all vectors sorted by list id, so list c is the slice
    vecs[offsets[c]:offsets[c+1]]. Scanning a list is one BLAS call on a view,
    with no copying and no Python loop over vectors.
    """

    def __init__(self, centroids, offsets, vecs):
        self.centroids = centroids
        self.cnorms = squared_norms(centroids)
        self.offsets = offsets
        self.vecs = vecs
        self.norms = squared_norms(vecs)
        self.nlist = len(centroids)
        self._buf = np.empty(len(vecs), dtype=np.float32)   # reused by every query
        self.stats = {}

    # ---------- build ----------
    @classmethod
    def build(cls, vecs, nlist, sample_per_list=64, iters=10, seed=0):
        """Train centroids on a sample, then assign every vector to its nearest centroid.

        Returns (index, order) where order[i] is the input row stored at sorted row i.
        """
        rng = np.random.default_rng(seed)
        n = len(vecs)
        t0 = time.perf_counter()
        n_train = min(n, sample_per_list * nlist)
        sample = vecs if n_train == n else vecs[np.sort(rng.choice(n, n_train, replace=False))]
        centroids = kmeans(sample, nlist, iters=iters, seed=seed)
        t1 = time.perf_counter()
        a = assign(vecs, centroids, squared_norms(centroids))
        index, order = cls.from_assignment(vecs, a, centroids)
        index.stats.update(train_s=t1 - t0, add_s=time.perf_counter() - t1, n_train=n_train)
        return index, order

    @classmethod
    def from_assignment(cls, vecs, a, centroids):
        """Group rows by list id (stable sort) and record where each list starts."""
        nlist = len(centroids)
        order = np.argsort(a, kind="stable")
        counts = np.bincount(a, minlength=nlist)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        index = cls(centroids, offsets, np.ascontiguousarray(vecs[order]))
        n = max(1, len(vecs))
        index.stats.update(
            list_min=int(counts.min()), list_mean=float(counts.mean()), list_max=int(counts.max()),
            # 1.0 = all lists equal; it multiplies the expected number of rows scanned
            imbalance=float((counts.astype(np.float64) ** 2).sum() * nlist / n ** 2),
        )
        return index, order

    def list_ids(self):
        """List id of every sorted row (used by compaction to avoid re-assigning)."""
        return np.repeat(np.arange(self.nlist, dtype=np.int32), np.diff(self.offsets))

    # ---------- search ----------
    def search(self, q, k, nprobe, alive=None):
        """Scan the nprobe lists whose centroids are closest to q.

        Returns (rows, ranking_distances, rows_scanned). rows index the sorted layout.
        """
        cd = self.centroids @ q
        cd *= -2.0
        cd += self.cnorms
        if nprobe < self.nlist:
            probe = np.argpartition(cd, nprobe - 1)[:nprobe]
        else:
            probe = np.arange(self.nlist)
        off, vecs, norms, buf = self.offsets, self.vecs, self.norms, self._buf
        pos = 0
        seg_pos, seg_row = [], []           # where each probed list landed in buf
        for c in probe.tolist():
            lo, hi = off[c], off[c + 1]
            n = hi - lo
            if n == 0:
                continue
            out = buf[pos:pos + n]
            np.dot(vecs[lo:hi], q, out=out)  # one BLAS call per list, written in place
            out *= -2.0
            out += norms[lo:hi]
            if alive is not None:
                out[~alive[lo:hi]] = np.inf
            seg_pos.append(pos)
            seg_row.append(lo)
            pos += n
        if pos == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0
        dist = buf[:pos]
        idx = topk(dist, k)
        seg_pos = np.asarray(seg_pos)
        seg_row = np.asarray(seg_row)
        s = np.searchsorted(seg_pos, idx, side="right") - 1   # which probed list each winner is in
        rows = seg_row[s] + (idx - seg_pos[s])
        return rows, dist[idx].copy(), pos
