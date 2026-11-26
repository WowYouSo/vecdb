"""What inserts and deletes after the index build cost, and what compaction buys back.

The index is built on 90% of the data. The rest is inserted afterwards (it lands in the
delta segment), then 10% of the indexed rows are deleted, then we compact twice: reusing
the old centroids, and retraining them. After every stage we measure query time and recall.
The ground truth at every stage is the exact search over the same database state.
"""
import common  # noqa: F401  (must be first)

import argparse
import shutil
import tempfile
import time

import numpy as np

from common import SIFT, TOL, load, write_csv
from vecdb import VectorDB


def measure(db, Q, k):
    for q in Q[:10]:
        db.search(q, k)
    rec = 0.0
    for q in Q:
        _, d = db.search(q, k)                         # default nprobe
        _, d_true = db.search(q, k, exact=True)
        rec += float(np.count_nonzero(d <= d_true[-1] * (1 + TOL))) / k
    passes = []
    for _ in range(3):                                 # best of 3 passes: removes scheduler noise
        t0 = time.perf_counter()
        for q in Q:
            db.search(q, k)
        passes.append(time.perf_counter() - t0)
    s = db.stats()
    return dict(mean_ms=1e3 * min(passes) / len(Q), recall=rec / len(Q), sealed_rows=s["sealed_rows"],
                sealed_dead=s["sealed_dead"], delta_rows=s["delta_rows"], alive=s["alive"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=SIFT)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=1024)
    args = ap.parse_args()

    ds = load(args.data, args.n, args.queries, args.k, need_gt=False)
    base, Q = ds["base"], ds["queries"]
    n, dim = len(base), base.shape[1]
    n_idx = int(0.9 * n)
    nlist = min(args.nlist, max(16, n_idx // 64))

    path = tempfile.mkdtemp(prefix="vecdb-delta-")
    db = VectorDB.open(path, dim, sync="os")
    for s in range(0, n_idx, 50_000):
        e = min(s + 50_000, n_idx)
        db.insert_many(np.arange(s, e), base[s:e])
    db.build_index(nlist, checkpoint=False)
    rows = []

    def stage(label, seconds=0.0):
        r = measure(db, Q, args.k)
        r.update(stage=label, stage_seconds=seconds, nlist=nlist, nprobe=db.default_nprobe, dataset=ds["name"])
        rows.append(r)
        print(f"{label:34s} {r['mean_ms']:8.3f} ms  recall {r['recall']:.4f}  delta {r['delta_rows']:7d}  dead {r['sealed_dead']:7d}")

    stage("index only")
    done = n_idx
    for share in (0.5, 1, 2, 5, 10):
        upto = n_idx + int(n_idx * share / 100)
        if upto > n:
            break
        db.insert_many(np.arange(done, upto), base[done:upto])
        done = upto
        stage(f"delta = {share}% of index")
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for i in rng.choice(n_idx, n_idx // 10, replace=False).tolist():
        db.delete(i)
    stage("+ 10% of indexed rows deleted", time.perf_counter() - t0)
    t0 = time.perf_counter()
    db.compact(retrain=False, checkpoint=False)
    stage("after compact (same centroids)", time.perf_counter() - t0)
    t0 = time.perf_counter()
    db.compact(retrain=True, checkpoint=False)
    stage("after compact (retrained)", time.perf_counter() - t0)
    db.close()
    shutil.rmtree(path, ignore_errors=True)
    write_csv("delta.csv", rows)


if __name__ == "__main__":
    main()
