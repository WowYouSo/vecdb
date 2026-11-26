"""Correctness tests. Run: pytest -q"""
import os

import numpy as np
import pytest

from vecdb import VectorDB
from vecdb.flat import flat_scan, squared_norms, topk
from vecdb.ivf import IVFIndex
from vecdb.kmeans import assign

DIM = 32


def make(n, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((50, DIM)).astype(np.float32)
    return (centers[rng.integers(0, 50, n)] + 0.5 * rng.standard_normal((n, DIM))).astype(np.float32)


def brute(X, q, k):
    d = np.linalg.norm(X - q, axis=1)
    return np.argsort(d, kind="stable")[:k]


@pytest.fixture
def data():
    return make(5000), make(40, seed=1)


def test_flat_matches_bruteforce(data):
    X, Q = data
    norms = squared_norms(X)
    for q in Q:
        got = topk(flat_scan(X, norms, q), 10)
        assert set(got.tolist()) == set(brute(X, q, 10).tolist())


def test_ivf_with_all_lists_is_exact(data):
    X, Q = data
    index, order = IVFIndex.build(X, nlist=64)
    assert index.offsets[-1] == len(X) and sorted(order.tolist()) == list(range(len(X)))
    for q in Q:
        rows, _, scanned = index.search(q, 10, nprobe=64)
        assert scanned == len(X)
        assert set(order[rows].tolist()) == set(brute(X, q, 10).tolist())


def test_ivf_recall_grows_with_nprobe(data):
    X, Q = data
    index, order = IVFIndex.build(X, nlist=64)
    recalls = []
    for nprobe in (1, 4, 16, 64):
        hit = sum(len(set(order[index.search(q, 10, nprobe)[0]].tolist()) & set(brute(X, q, 10).tolist())) for q in Q)
        recalls.append(hit / (10 * len(Q)))
    assert recalls == sorted(recalls) and recalls[-1] == 1.0 and recalls[0] < 1.0


def test_db_insert_search_delete_replace(tmp_path, data):
    X, Q = data
    db = VectorDB.open(str(tmp_path / "db"), DIM, sync="os")
    db.insert_many(np.arange(len(X)), X)
    ids, dists = db.search(Q[0], k=10)                       # no index yet: exact
    assert ids.tolist() == brute(X, Q[0], 10).tolist()
    assert np.allclose(dists, np.linalg.norm(X[ids] - Q[0], axis=1), rtol=1e-3, atol=1e-3)
    db.build_index(nlist=64)
    assert db.search(Q[0], k=10, exact=True)[0].tolist() == brute(X, Q[0], 10).tolist()
    # insert after build: found at once, at distance 0, coming from the delta segment
    db.insert(999_999, Q[0])
    ids, dists = db.search(Q[0], k=10, nprobe=4)
    assert ids[0] == 999_999 and dists[0] < 1e-2 and db.stats()["delta_rows"] == 1
    # delete: never returned again
    assert db.delete(999_999) and not db.delete(999_999)
    assert 999_999 not in db.search(Q[0], k=10, nprobe=4)[0]
    # delete a sealed row, replace another one
    victim = int(db.search(Q[1], k=1, exact=True)[0][0])
    db.delete(victim)
    assert victim not in db.search(Q[1], k=10, exact=True)[0]
    db.insert(7, Q[2])
    assert db.search(Q[2], k=1, nprobe=4)[0][0] == 7 and len(db) == len(X) - 1
    db.close()


def test_recovery_from_log_and_snapshot(tmp_path, data):
    X, Q = data
    path = str(tmp_path / "db")
    db = VectorDB.open(path, DIM, sync="os")
    db.insert_many(np.arange(3000), X[:3000])
    db.build_index(nlist=32)                                  # checkpoints
    db.insert_many(np.arange(3000, 5000), X[3000:])           # only in the log
    db.delete(10)
    before = [db.search(q, 10, exact=True)[0].tolist() for q in Q]
    db.close()
    db = VectorDB.open(path, DIM)
    assert db.recovery["replayed_records"] == 2 and len(db) == 4999 and 10 not in db
    assert db.ivf is not None and db.stats()["delta_rows"] == 2000
    assert [db.search(q, 10, exact=True)[0].tolist() for q in Q] == before
    db.close()


def test_torn_tail_is_dropped(tmp_path, data):
    X, _ = data
    path = str(tmp_path / "db")
    db = VectorDB.open(path, DIM, sync="os")
    for i in range(10):
        db.insert(i, X[i])
    wal = db.wal.path
    db.close()
    size = os.path.getsize(wal)
    os.truncate(wal, size - 7)                                # crash in the middle of the last record
    db = VectorDB.open(path, DIM, sync="os")
    assert len(db) == 9 and 9 not in db and db.recovery["torn_bytes"] > 0
    assert os.path.getsize(wal) == size // 10 * 9
    db.insert(9, X[9])                                        # the log is usable again
    db.close()
    assert len(VectorDB.open(path, DIM)) == 10


def test_corrupted_record_is_dropped(tmp_path, data):
    X, _ = data
    path = str(tmp_path / "db")
    db = VectorDB.open(path, DIM, sync="os")
    for i in range(5):
        db.insert(i, X[i])
    wal = db.wal.path
    db.close()
    with open(wal, "r+b") as f:                               # flip one byte in the last record
        f.seek(-3, os.SEEK_END)
        b = f.read(1)
        f.seek(-3, os.SEEK_END)
        f.write(bytes([b[0] ^ 0xFF]))
    assert len(VectorDB.open(path, DIM)) == 4


def test_compaction_keeps_results(tmp_path, data):
    X, Q = data
    path = str(tmp_path / "db")
    db = VectorDB.open(path, DIM, sync="os")
    db.insert_many(np.arange(4000), X[:4000])
    db.build_index(nlist=32)
    db.insert_many(np.arange(4000, 5000), X[4000:])
    for i in range(0, 5000, 7):
        db.delete(i)
    before = [db.search(q, 10, exact=True) for q in Q]
    for retrain in (False, True):
        db.compact(retrain=retrain)
        s = db.stats()
        assert s["delta_rows"] == 0 and s["sealed_dead"] == 0 and s["sealed_rows"] == len(db)
        for q, (ids, dists) in zip(Q, before):
            got_ids, got_d = db.search(q, 10, exact=True)
            assert np.allclose(got_d, dists, rtol=1e-3, atol=1e-3) and set(got_ids.tolist()) == set(ids.tolist())
            rows = db.ivf.search(q, 10, nprobe=32)[0]              # the IVF path itself, all lists
            assert set(db.S_ids[rows].tolist()) == set(ids.tolist())
        if not retrain:                                           # every row still sits in its nearest list
            a = assign(db.ivf.vecs, db.ivf.centroids, db.ivf.cnorms)
            assert (a == db.ivf.list_ids()).mean() > 0.999
    db.close()
    db = VectorDB.open(path, DIM)                             # compaction checkpointed
    assert db.recovery["replayed_records"] == 0 and len(db) == s["sealed_rows"]
    assert sorted(os.listdir(path)) == ["CURRENT", f"snap-{db.gen:06d}", f"wal-{db.gen:06d}.log"]
    db.close()


@pytest.mark.parametrize("mode", ["buffered", "os", "batch", "always"])
def test_all_sync_modes_roundtrip(tmp_path, data, mode):
    X, _ = data
    path = str(tmp_path / mode)
    db = VectorDB.open(path, DIM, sync=mode)
    for i in range(200):
        db.insert(i, X[i])
    db.close()
    assert len(VectorDB.open(path, DIM)) == 200
