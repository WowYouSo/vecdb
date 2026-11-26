"""External baselines on the same data and machine: FAISS Flat, IVFFlat and HNSW.

FAISS is an optimised C++ library and is NOT our code. It is here to show where a
production-grade implementation sits. Queries run one at a time on one thread.
"""
import common  # noqa: F401  (must be first)

import argparse
import os
import time

import faiss
import numpy as np

from common import SIFT, exact_dists, load, nprobe_sweep, recall, timing_row, write_csv


def measure(index, base, Q, gt, k, nq):
    for j in range(20):
        index.search(Q[j:j + 1], k)
    times, rec = [], 0.0
    for j in range(nq):
        t0 = time.perf_counter()
        _, ids = index.search(Q[j:j + 1], k)
        times.append(time.perf_counter() - t0)
        ids = ids[0][ids[0] >= 0]
        rec += recall(exact_dists(base, Q[j], ids), gt[j], k)
    row = timing_row(times)
    row.update(recall=rec / nq, queries=nq)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=SIFT)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--hnsw-m", type=int, default=32)
    args = ap.parse_args()

    ds = load(args.data, args.n, args.queries, args.k)
    base, Q, gt = ds["base"], ds["queries"], ds["gt_kth"]
    n, dim, nq = len(base), base.shape[1], len(Q)
    common_cols = dict(dataset=ds["name"], n=n, k=args.k)
    rows = []
    faiss.omp_set_num_threads(1)

    index = faiss.IndexFlatL2(dim)
    index.add(base)
    row = measure(index, base, Q, gt, args.k, min(nq, 300))
    row.update(system="faiss flat", nlist=0, param=0, **common_cols)
    rows.append(row)
    print(f"faiss flat: recall {row['recall']:.4f}, {row['mean_ms']:.2f} ms/query, {row['qps']:.0f} QPS")

    for nlist in args.nlist:
        if nlist * 8 > n:
            continue
        t0 = time.perf_counter()
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
        index.train(base)
        index.add(base)
        build = time.perf_counter() - t0
        print(f"\nfaiss ivf{nlist}: build {build:.1f}s")
        for nprobe in nprobe_sweep(nlist):
            index.nprobe = nprobe
            row = measure(index, base, Q, gt, args.k, nq if nprobe <= 128 else min(nq, 200))
            row.update(system=f"faiss ivf{nlist}", nlist=nlist, param=nprobe, build_s=build, **common_cols)
            rows.append(row)
            print(f"  nprobe {nprobe:5d}  recall {row['recall']:.4f}  {row['mean_ms']:.3f} ms  {row['qps']:.0f} QPS")

    # HNSW: built with all cores (build time is not what we compare), queried on one thread
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, args.hnsw_m)
    index.hnsw.efConstruction = 100
    index.add(base)
    build = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)
    print(f"\nfaiss hnsw (M={args.hnsw_m}, efConstruction=100): build {build:.1f}s on {os.cpu_count()} threads")
    for ef in (16, 32, 64, 128, 256, 512):
        index.hnsw.efSearch = ef
        row = measure(index, base, Q, gt, args.k, nq)
        row.update(system=f"faiss hnsw{args.hnsw_m}", nlist=0, param=ef, build_s=build, **common_cols)
        rows.append(row)
        print(f"  efSearch {ef:4d}  recall {row['recall']:.4f}  {row['mean_ms']:.3f} ms  {row['qps']:.0f} QPS")
    write_csv("faiss.csv", rows)


if __name__ == "__main__":
    main()
