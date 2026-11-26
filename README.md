# vecdb

A small single-node vector database written for a Databases course. Python + NumPy, about 750 lines of engine code.

It stores `(id, float32 vector)` pairs and answers "give me the k nearest vectors by L2 distance":

- **exact search** (flat scan), the ground truth and the fallback;
- **approximate search** with our own **IVF-Flat** index (k-means lists, `nprobe` knob);
- **durability**: write-ahead log with four sync modes, checkpoints, crash recovery;
- **changes after the index is built**: delta segment, tombstones, compaction.

Not implemented, on purpose: HNSW, vector compression (PQ), SQL, a network API, concurrent
transactions. One process, one thread, operations run one after another.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && curl -L -o data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5

python -m pytest -q            # 12 correctness tests
python demo.py                 # scripted walk-through on 200k vectors (about 5 s on an Apple M4)
python demo.py --n 1000000     # the same walk-through on the full 1M (about 11 s)
./run_all.sh                   # every benchmark on the full 1M, then results/summary.md
N=200000 ./run_all.sh          # the same, quicker
```

Without the SIFT file everything still runs on synthetic data and says so loudly.

## Layout

| Path | What it is |
|---|---|
| `vecdb/flat.py` | distance by one BLAS call, top-k selection |
| `vecdb/kmeans.py` | k-means for the IVF lists |
| `vecdb/ivf.py` | IVF-Flat index: build and search |
| `vecdb/wal.py` | write-ahead log: record format, sync modes, replay |
| `vecdb/db.py` | `VectorDB`: segments, tombstones, checkpoint, recovery, compaction |
| `bench/` | benchmarks, crash test, plots (`report.py`) |
| `demo.py` | the walk-through used in the video |
| `DECISIONS.md` | every design decision: options, choice, cost, evidence |
| `WALKTHROUGH.md` | block-by-block explanation of the code (Russian) |

## API

```python
from vecdb import VectorDB
db = VectorDB.open("mydb", dim=128, sync="batch")   # buffered | os | batch | always
db.insert_many(ids, vectors)        # one log record
db.insert(42, vector)               # replaces an existing id
db.delete(42)
db.build_index(nlist=1024)          # k-means + IVF lists, ends with a checkpoint
ids, dists = db.search(q, k=10, nprobe=32)      # approximate
ids, dists = db.search(q, k=10, exact=True)     # exact
db.compact()                        # fold the delta in, drop deleted rows
db.checkpoint(); db.close()
```

## Honest limits

- FAISS is faster at the same recall: it is C++ with SIMD, ours is NumPy. The baseline is there to show the gap.
- The crash test uses SIGKILL. That cannot show the difference between "written to the OS" and "fsynced";
  only a power loss can. We measure what each mode costs and state what it guarantees.
- SIFT1M is a 2010-era dataset of hand-made image descriptors, not modern neural embeddings.
- Deletes and replaces cost a Python dict operation each; bulk deletes are slow.
