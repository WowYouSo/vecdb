# Design decisions

For each decision: the options, what we chose, why, what it costs, and where the evidence is.
"Measured" lines are filled from `results/summary.md` after `./run_all.sh` on the target machine.

## D0. Data model: float32, L2, vectors stored uncompressed ("Flat")
- **Options:** float32 / float16 / int8 / product quantization; L2 / cosine / inner product.
- **Choice:** float32, L2, no compression.
- **Why:** distances stay exact, so the index can only make one kind of mistake (a neighbour in a list
  we did not scan). That keeps the recall story simple and the code small.
- **Cost:** 512 bytes per 128-dim vector. 1M vectors = 512 MB in RAM and on disk.

## D1. How we measure accuracy: recall@10 by distance threshold
- **Options:** match by id; match by distance (ANN-Benchmarks definition).
- **Choice:** a returned point counts if its exact distance is at most the true 10th distance (0.1% tolerance).
- **Why:** SIFT coordinates are small integers, so ties at the 10th distance happen. Id matching would
  count a tied point as an error.
- **Note:** a top-k answer always has exactly k items, so every wrong item displaces one true neighbour.
  False positives equal false negatives and precision@k equals recall@k: one number is enough.

## D2. Number of lists (nlist)
- **Options:** 1024 (pgvector rule: rows/1000) or 4096 (FAISS rule: 4*sqrt(N)); cost model
  `distances = nlist + nprobe * N / nlist`.
- **Choice:** nlist = 4096 at 1M vectors. The best nlist grows with N: at 200k, 1024 wins; at 1M, 4096 wins.
- **Cost:** more lists = fewer rows per probe but more probes needed for the same recall, longer k-means.
- **Measured:** at recall >= 0.95, ivf1024 gives 1,746 QPS (0.57 ms, nprobe 32) and ivf4096 gives 2,226 QPS
  (0.45 ms, nprobe 64). k-means took 0.8 s for nlist = 1024 and 10.0 s for nlist = 4096.

## D3. k-means settings
- **Options:** train on everything or on a sample; random or k-means++ init; converge or fixed iterations.
- **Choice:** sample of 64 vectors per list, random init, 10 iterations, empty clusters re-seeded.
- **Why:** centroids only need to be roughly right; assignment of all vectors happens afterwards anyway.
- **Cost:** lists are uneven. We print the imbalance factor (1.0 = even); it multiplies the rows scanned.
- **Measured:** imbalance 1.20 for nlist = 1024 and 1.23 for nlist = 4096.

## D4. Default nprobe
- **Options:** 1 (pgvector default), sqrt(nlist) (pgvector's suggested start), or a recall target.
- **Choice:** sqrt(nlist); the caller can override per query.
- **Measured:** nlist = 1024 (nprobe 32): recall 0.9796 at 0.573 ms/query. nlist = 4096 (nprobe 64):
  recall 0.9720 at 0.449 ms/query.

## D5. Durability mode and sync call
- **Options:** `buffered` (in process memory), `os` (written, never synced), `batch` (group commit:
  sync every 64 records or 50 ms), `always` (sync before returning). Sync call: `fsync` or macOS `F_FULLFSYNC`.
- **Choice:** default `batch`. It survives a process crash completely and bounds what a power loss can take.
- **Why not always:** one sync per insert caps throughput at the sync rate of the disk.
- **Why F_FULLFSYNC matters:** on macOS `fsync` hands data to the drive but does not force the drive to
  flush its own cache. `F_FULLFSYNC` does. Without it "synced" data can still be lost on power failure.
- **Cost:** The default `batch` uses plain fsync. On macOS that does not force the drive to flush its
  cache, so a power loss can still lose synced records. `full_sync=True` closes the gap at the
  measured F_FULLFSYNC rate.
- **Measured:** inserts per second: `buffered` 321,374; `os` 175,307; `batch` 123,503 with `fsync` and
  11,604 with `F_FULLFSYNC`; `always` 36,091 with `fsync` and 255 with `F_FULLFSYNC`.

## D6. What a checkpoint contains
- **Options:** data only (rebuild the index on every start) or data + index.
- **Choice:** data + index (centroids, list offsets, vectors already in list order).
- **Why:** the log never contains index structures, so the snapshot is the only place the index can come from
  without running k-means again.
- **Cost:** nothing extra for IVF-Flat, because the sorted vectors are the data.
- **Measured:** on 1,000,000 vectors: open with snapshot (index included, empty log) 0.10 s; rebuilding the
  index from scratch (k-means + assign) 13.79 s; open with snapshot + 100,000 log records to replay 0.44 s.

## D7. Inserts after the index is built
- **Options:** (a) append to the nearest centroid's list in place (pgvector); (b) a separate delta segment
  scanned exactly (Milvus "growing segment", LSM-style).
- **Choice:** (b).
- **Why:** our lists are one contiguous array; inserting into the middle would mean moving data or giving up
  contiguity, which is what makes scans fast. A delta keeps the sealed segment immutable.
- **Cost:** every query scans the whole delta, so query time grows linearly with it until compaction.
- **Measured:** 0.430 ms/query at delta = 0; 0.535 ms at 1%; 1.192 ms at 5%; 1.781 ms at 10% of the index.

## D8. Deletes
- **Options:** remove the row physically; tombstone + mask during the scan; tombstone + over-fetch and filter.
- **Choice:** tombstone bit, masked during the scan (dead rows get distance +inf).
- **Why:** masking can never return fewer than k results while k alive rows are in the scanned lists.
  Over-fetching is cheaper but can come back short. Masking is skipped entirely when nothing is dead.
- **Cost:** dead rows are still scanned and still occupy disk until compaction.

## D9. Compaction
- **Options:** reuse centroids (assign new rows to old centroids) or retrain k-means.
- **Choice:** reuse by default, `retrain=True` available. Triggered manually.
- **Why:** reuse is one assignment pass; retraining costs a full k-means. Reuse is right while the data
  distribution has not moved.
- **Measured:** before compaction (delta = 10%, 10% of indexed rows deleted): 1.749 ms/query, recall 0.9810.
  After reuse: 0.418 ms/query, recall 0.9800, took 0.64 s. After retrain: 0.421 ms/query, recall 0.9840,
  took 13.46 s.
- **Note:** The inserted vectors come from the same distribution as the indexed ones, so retraining
  shows no measurable gain here. A shifted distribution was not tested.

## D10. Concurrency
- **Choice:** none. One process, one thread, each operation finishes before the next starts.
- **Consequence:** execution is serial, so there are no isolation anomalies to discuss, and no parallelism
  either. Opening the same directory from two processes is not protected.

## D11. Ranking formula and final re-check
- **Choice:** rank with `||x||^2 - 2 x.q` (one BLAS call per list), then recompute the true distance for the
  final candidates (at most 2k rows).
- **Why:** the fast formula loses float32 digits to cancellation (an exact copy can come out at distance
  0.004 instead of 0). Recomputing 2k distances costs microseconds and returns exact values.

## D12. A one-rule planner
- **Choice:** if `nprobe >= nlist` (or `exact=True`) do a flat scan instead of probing every list.
- **Why:** probing all lists gives the same answer with per-list overhead on top. pgvector's planner makes
  the same switch.

## D13. Log record and recovery
- **Choice:** `length | crc32 | payload`, sequence number in every record; recovery stops at the first
  short or CRC-failing record and truncates there; records already covered by the snapshot are skipped.
- **Why:** a crash can leave half a record at the tail. Length + CRC detects it; sequence numbers make
  replay idempotent.

## D14. Checkpoint switch
- **Choice:** write the snapshot into a temp directory, sync, rename; then atomically replace a small
  `CURRENT` file that names the live snapshot; only then delete the old snapshot and log.
- **Why:** at every instant `CURRENT` names a complete snapshot + log pair. This is the LevelDB manifest idea.
