"""Recall vs speed of our database: exact scan and IVF-Flat for several nlist / nprobe."""
import common  # noqa: F401  (must be first: pins BLAS threads)

import argparse
import shutil
import tempfile
import time

import numpy as np

from common import SIFT, exact_dists, load, nprobe_sweep, recall, timing_row, write_csv
from vecdb import VectorDB


def measure(db, base, Q, gt, k, nprobe, exact, nq):
    for q in Q[:20]:                                   # warm-up, not timed
        db.search(q, k, nprobe=nprobe, exact=exact)
    times, rec, scanned = [], 0.0, 0
    for j in range(nq):
        t0 = time.perf_counter()
        ids, _ = db.search(Q[j], k, nprobe=nprobe, exact=exact)
        times.append(time.perf_counter() - t0)         # only the search is timed
        scanned += db.last_scanned
        rec += recall(exact_dists(base, Q[j], ids), gt[j], k)   # distances recomputed outside the timer
    row = timing_row(times)
    row.update(recall=rec / nq, scanned_share=scanned / (nq * len(base)), queries=nq)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=SIFT)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, nargs="+", default=[1024, 4096])
    args = ap.parse_args()

    ds = load(args.data, args.n, args.queries, args.k)
    base, Q, gt = ds["base"], ds["queries"], ds["gt_kth"]
    n, dim, nq = len(base), base.shape[1], len(Q)
    print(f"dataset: {ds['name']}, ground truth: {ds['gt_source']}")

    path = tempfile.mkdtemp(prefix="vecdb-bench-")
    db = VectorDB.open(path, dim, sync="os")
    t0 = time.perf_counter()
    for s in range(0, n, 50_000):
        db.insert_many(np.arange(s, min(s + 50_000, n)), base[s:s + 50_000])
    print(f"loaded {n:,} vectors in {time.perf_counter() - t0:.1f}s")

    rows = []
    row = measure(db, base, Q, gt, args.k, None, True, min(nq, 300))
    row.update(system="vecdb flat", nlist=0, param=0, dataset=ds["name"], n=n, k=args.k)
    rows.append(row)
    flat_ms = row["mean_ms"]
    print(f"flat: recall {row['recall']:.4f}, {row['mean_ms']:.2f} ms/query, {row['qps']:.0f} QPS")

    for nlist in args.nlist:
        if nlist * 8 > n:
            print(f"skip nlist={nlist}: too few vectors")
            continue
        st = db.build_index(nlist, checkpoint=False)
        extra = (nlist * dim * 4 + (nlist + 1) * 8 + n * 4) / n     # centroids + offsets + norms
        print(f"\nnlist={nlist}: train {st['train_s']:.1f}s on {st['n_train']:,} vectors, add {st['add_s']:.1f}s, "
              f"lists min/mean/max {st['list_min']}/{st['list_mean']:.0f}/{st['list_max']}, imbalance {st['imbalance']:.2f}")
        print(" nprobe  recall@%d   ms/query      QPS   vs flat  scanned" % args.k)
        for nprobe in nprobe_sweep(nlist):
            row = measure(db, base, Q, gt, args.k, nprobe, False, nq if nprobe <= 128 else min(nq, 200))
            row.update(system=f"vecdb ivf{nlist}", nlist=nlist, param=nprobe, dataset=ds["name"], n=n, k=args.k,
                       train_s=st["train_s"], add_s=st["add_s"], imbalance=st["imbalance"],
                       index_bytes_per_vec=extra)
            rows.append(row)
            print(f"{nprobe:7d}  {row['recall']:9.4f}  {row['mean_ms']:9.3f}  {row['qps']:7.0f}  "
                  f"{flat_ms / row['mean_ms']:6.1f}x  {100 * row['scanned_share']:6.2f}%")
    db.close()
    shutil.rmtree(path, ignore_errors=True)
    write_csv("search.csv", rows)


if __name__ == "__main__":
    main()
