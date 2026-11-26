"""A scripted walk through the whole database, meant to be recorded.

    python demo.py                  # 200k vectors from SIFT1M (or synthetic if the file is missing)
    python demo.py --n 1000000      # full dataset
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench"))
import common  # noqa: E402  (pins BLAS threads; must come before numpy)

import argparse  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import crash_test  # noqa: E402
from vecdb import VectorDB  # noqa: E402

K = 10


def title(n, text):
    print(f"\n{'=' * 78}\n[{n}] {text}\n{'=' * 78}")


def files(path):
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else os.path.getsize(p)
        print(f"    {name:22s} {size / 1e6:9.1f} MB")


def timed_search(db, Q, **kw):
    t0 = time.perf_counter()
    out = [db.search(q, K, **kw) for q in Q]
    return out, 1e3 * (time.perf_counter() - t0) / len(Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=common.SIFT)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--nlist", type=int, default=0, help="0 = n/1000 (the pgvector rule)")
    ap.add_argument("--path", default="demo_db")
    args = ap.parse_args()

    title(1, "Data")
    ds = common.load(args.data, args.n + 5000, 200, K, need_gt=False)
    base, extra, Q = ds["base"][:-5000], ds["base"][-5000:], ds["queries"]
    n, dim = base.shape
    nlist = args.nlist or max(16, n // 1000)
    print(f"  {ds['name']}: {n:,} vectors to load, {len(extra):,} kept aside for later inserts, {len(Q)} queries")

    title(2, "Create the database and load the vectors (every batch goes to the log first)")
    shutil.rmtree(args.path, ignore_errors=True)
    db = VectorDB.open(args.path, dim, sync="batch")
    t0 = time.perf_counter()
    for s in range(0, n, 50_000):
        e = min(s + 50_000, n)
        db.insert_many(np.arange(s, e), base[s:e])
    print(f"  inserted {len(db):,} vectors in {time.perf_counter() - t0:.2f}s, last sequence number {db.seq}")
    files(args.path)

    title(3, "Exact search: compare the query with every vector")
    exact, exact_ms = timed_search(db, Q, exact=True)
    ids, dists = exact[0]
    print(f"  query 0 -> ids   {ids.tolist()}")
    print(f"             dists {[round(float(x), 1) for x in dists]}")
    print(f"  {exact_ms:.2f} ms per query, {n:,} vectors scanned each time, recall 1.0 by definition")

    title(4, f"Build the IVF index: k-means into {nlist} lists")
    st = db.build_index(nlist)
    print(f"  train {st['train_s']:.2f}s on {st['n_train']:,} sampled vectors, assign+sort {st['add_s']:.2f}s")
    print(f"  list sizes min/mean/max = {st['list_min']}/{st['list_mean']:.0f}/{st['list_max']}, imbalance {st['imbalance']:.2f} (1.00 = perfectly even)")
    print(f"  default nprobe = sqrt(nlist) = {db.default_nprobe}; build ended with a checkpoint:")
    files(args.path)

    title(5, "Approximate search: scan only the nprobe closest lists")
    kth = [d[-1] for _, d in exact]
    print("   nprobe   recall@10   ms/query   speed-up   rows scanned")
    for nprobe in [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p < nlist]:
        res, ms = timed_search(db, Q, nprobe=nprobe)
        scanned = 0
        for q in Q[:50]:
            db.search(q, K, nprobe=nprobe)
            scanned += db.last_scanned / 50
        rec = np.mean([common.recall(d, kth[j], K) for j, (_, d) in enumerate(res)])
        print(f"   {nprobe:6d}   {rec:9.3f}   {ms:8.3f}   {exact_ms / ms:7.1f}x   {100 * scanned / n:10.2f}%")
    ids1, _ = db.search(Q[0], K, nprobe=1)
    missed = sorted(set(ids.tolist()) - set(ids1.tolist()))
    print(f"  query 0 with nprobe=1 misses {len(missed)} of the 10 true neighbours: {missed}")
    print("  they live in lists that were not scanned. That is the only kind of error IVF-Flat makes.")

    title(6, "Insert after the index is built: the vector goes to the delta segment")
    new_id = 10_000_000
    db.insert(new_id, Q[0])
    ids2, d2 = db.search(Q[0], K)
    s = db.stats()
    print(f"  inserted id {new_id} (a copy of query 0); search returns it first: id {ids2[0]}, distance {d2[0]:.3f}")
    print(f"  sealed rows {s['sealed_rows']:,}, delta rows {s['delta_rows']}: no re-clustering happened")

    title(7, "Delete: a tombstone, the row stays on disk until compaction")
    db.delete(new_id)
    ids3, _ = db.search(Q[0], K)
    print(f"  deleted id {new_id}; still returned? {new_id in ids3.tolist()}; delta rows {db.stats()['delta_rows']}, dead in delta {db.stats()['delta_dead']}")

    title(8, "Crash: kill -9 a writer process, reopen, count acknowledged inserts that survived")
    print("   mode       acknowledged   recovered   lost acknowledged")
    for mode in ("buffered", "os", "always"):
        r = crash_test.run_trial(mode, 0.8)
        print(f"   {mode:9s}  {r['acknowledged']:12,d}   {r['recovered']:9,d}   {r['lost_acknowledged']:10d}")
    print("  buffered loses the records still sitting in process memory; the others were already handed to the OS.")
    print("  (a process kill cannot tell 'os' from 'always': only a power loss can. See bench_writes for the price.)")

    title(9, "Restart: load the snapshot, replay the log")
    db.insert_many(np.arange(n, n + 2000), extra[:2000])
    before = db.search(Q[1], K)[0].tolist()
    db.close()
    t0 = time.perf_counter()
    db = VectorDB.open(args.path, dim, sync="batch")
    r = db.recovery
    print(f"  open took {time.perf_counter() - t0:.2f}s: snapshot {r['snapshot_s']:.2f}s, log replay {r['replay_s']:.3f}s "
          f"({r['replayed_records']} records), torn bytes {r['torn_bytes']}")
    print(f"  {len(db):,} vectors, index present: {db.ivf is not None} (not rebuilt), same answer as before: {db.search(Q[1], K)[0].tolist() == before}")

    title(10, "Compaction: fold the delta into the index, drop dead rows")
    db.insert_many(np.arange(n + 2000, n + 5000), extra[2000:])
    rng = np.random.default_rng(0)
    for i in rng.choice(n, n // 50, replace=False).tolist():
        db.delete(i)
    s = db.stats()
    _, ms_before = timed_search(db, Q)
    print(f"  before: sealed {s['sealed_rows']:,} rows ({s['sealed_dead']:,} dead), delta {s['delta_rows']:,} rows, {ms_before:.3f} ms/query")
    t0 = time.perf_counter()
    db.compact()
    s = db.stats()
    _, ms_after = timed_search(db, Q)
    print(f"  after:  sealed {s['sealed_rows']:,} rows ({s['sealed_dead']:,} dead), delta {s['delta_rows']:,} rows, {ms_after:.3f} ms/query "
          f"(compaction took {time.perf_counter() - t0:.2f}s)")
    files(args.path)
    db.close()
    print("\nDone. Full measurements: ./run_all.sh, then open results/summary.md")


if __name__ == "__main__":
    main()
