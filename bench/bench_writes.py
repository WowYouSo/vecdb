"""Cost of durability: single-vector inserts per second for every sync mode, and recovery time."""
import common  # noqa: F401  (must be first)

import argparse
import shutil
import tempfile
import time

import numpy as np

from common import SIFT, load, write_csv
from vecdb import VectorDB
from vecdb import wal as walmod

HAS_FULL = walmod.fcntl is not None and hasattr(walmod.fcntl, "F_FULLFSYNC")


def bench_mode(mode, full, n, V):
    path = tempfile.mkdtemp(prefix="vecdb-w-")
    db = VectorDB.open(path, V.shape[1], sync=mode, full_sync=full)
    t0 = time.perf_counter()
    for i in range(n):
        db.insert(i, V[i % len(V)])
    dt = time.perf_counter() - t0
    db.close()
    shutil.rmtree(path, ignore_errors=True)
    call = "F_FULLFSYNC" if full else ("fsync" if mode in ("batch", "always") else "none")
    return dict(mode=mode, sync_call=call, inserts=n, seconds=dt, inserts_per_s=n / dt, us_per_insert=1e6 * dt / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=SIFT)
    ap.add_argument("--n", type=int, default=200_000, help="vectors in the recovery experiment")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--log-records", type=int, default=100_000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    V = rng.standard_normal((4096, 128), dtype=np.float32)
    plan = [("buffered", False, 20_000), ("os", False, 20_000), ("batch", False, 20_000), ("always", False, 1_000)]
    if HAS_FULL:
        plan += [("batch", True, 20_000), ("always", True, 300)]
    rows = []
    print("mode      sync call     inserts/s   us/insert")
    for mode, full, n in plan:
        r = bench_mode(mode, full, n, V)
        rows.append(r)
        print(f"{mode:9s} {r['sync_call']:12s} {r['inserts_per_s']:10.0f}  {r['us_per_insert']:10.1f}")
    if not HAS_FULL:
        print("(F_FULLFSYNC is macOS-only: not available on this machine)")
    write_csv("writes.csv", rows)

    # ---------------- recovery time ----------------
    ds = load(args.data, args.n, 10, 10, need_gt=False)
    base = ds["base"]
    n, dim = len(base), base.shape[1]
    nlist = min(args.nlist, max(16, n // 64))
    rec = []

    path = tempfile.mkdtemp(prefix="vecdb-r-")
    db = VectorDB.open(path, dim, sync="os")
    for s in range(0, n, 50_000):
        db.insert_many(np.arange(s, min(s + 50_000, n)), base[s:s + 50_000])
    db.close()
    t0 = time.perf_counter()
    db = VectorDB.open(path, dim, sync="os")
    rec.append(dict(scenario="log only, no snapshot, no index", open_s=time.perf_counter() - t0, **db.recovery, vectors=n))
    st = db.build_index(nlist)                       # ends with a checkpoint
    rec.append(dict(scenario="rebuild the index from scratch (k-means + assign)", open_s=st["train_s"] + st["add_s"], vectors=n))
    db.close()
    t0 = time.perf_counter()
    db = VectorDB.open(path, dim, sync="os")
    rec.append(dict(scenario="snapshot with index, empty log", open_s=time.perf_counter() - t0, **db.recovery, vectors=n))
    m = args.log_records
    for i in range(m):
        db.insert(n + i, V[i % len(V)])
    db.close()
    t0 = time.perf_counter()
    db = VectorDB.open(path, dim, sync="os")
    rec.append(dict(scenario=f"snapshot with index + {m:,} log records", open_s=time.perf_counter() - t0, **db.recovery, vectors=n + m))
    db.close()
    shutil.rmtree(path, ignore_errors=True)
    print()
    for r in rec:
        print(f"{r['scenario']:52s} {r['open_s']:7.2f}s")
    write_csv("recovery.csv", rec)


if __name__ == "__main__":
    main()
