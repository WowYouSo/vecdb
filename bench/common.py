"""Shared helpers for the benchmark scripts.

Import this module BEFORE numpy anywhere else: it pins BLAS to one thread, so query
times are single-thread numbers (the ANN-Benchmarks convention).
"""
import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import json
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SIFT = os.path.join(ROOT, "data", "sift-128-euclidean.hdf5")
RESULTS = os.path.join(ROOT, "results")
TOL = 1e-3          # relative tolerance when comparing float32 distances


def synthetic(n, nq, dim=128, seed=0):
    """Gaussian mixture stand-in used when the SIFT file is missing."""
    rng = np.random.default_rng(seed)
    comp = rng.standard_normal((2000, dim), dtype=np.float32)
    base = comp[rng.integers(0, 2000, n)]
    base += 0.9 * rng.standard_normal((n, dim), dtype=np.float32)
    queries = comp[rng.integers(0, 2000, nq)] + 0.9 * rng.standard_normal((nq, dim), dtype=np.float32)
    return base, queries.astype(np.float32)


def exact_kth(base, queries, k):
    """True k-th nearest distance (Euclidean) for every query, by brute force in chunks."""
    norms = np.einsum("ij,ij->i", base, base)
    chunk = max(1, 64_000_000 // len(base))
    out = np.empty(len(queries), dtype=np.float64)
    for s in range(0, len(queries), chunk):
        qs = queries[s:s + chunk]
        d = qs @ base.T
        d *= -2.0
        d += norms[None, :]
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        for j in range(len(qs)):
            out[s + j] = np.linalg.norm(base[idx[j]] - qs[j], axis=1).max()
    return out


def load(path, n, nq, k, need_gt=True):
    """Return dict(base, queries, gt_kth, name, gt_source). Falls back to synthetic data."""
    if path and os.path.exists(path):
        import h5py
        with h5py.File(path, "r") as f:
            print("HDF5 keys:", {key: f[key].shape for key in f.keys()})
            total = f["train"].shape[0]
            n = min(n, total)
            base = np.ascontiguousarray(f["train"][:n], dtype=np.float32)
            queries = np.ascontiguousarray(f["test"][:nq], dtype=np.float32)
            gt, source = None, "none"
            if need_gt and n == total and "distances" in f and f["distances"].shape[1] >= k:
                gt = np.asarray(f["distances"][:nq, k - 1], dtype=np.float64)
                true0 = float(np.linalg.norm(base[int(f["neighbors"][0, k - 1])] - queries[0]))
                if abs(gt[0] - true0) <= 1e-2 * max(1.0, true0):
                    source = "file (Euclidean)"
                elif abs(np.sqrt(gt[0]) - true0) <= 1e-2 * max(1.0, true0):
                    gt, source = np.sqrt(gt), "file (was squared)"
                else:
                    gt = None
        name = f"{os.path.basename(path)} [{n:,} x {base.shape[1]}]"
    else:
        print(f"!! {path} not found: using SYNTHETIC data. Numbers are not comparable to SIFT.")
        base, queries = synthetic(n, nq)
        gt, source = None, "none"
        name = f"synthetic mixture [{n:,} x {base.shape[1]}]"
    if need_gt and gt is None:
        t0 = time.perf_counter()
        gt = exact_kth(base, queries, k)
        source = f"computed by brute force in {time.perf_counter() - t0:.1f}s"
    return dict(base=base, queries=queries, gt_kth=gt, name=name, gt_source=source)


def recall(dists, gt_kth, k):
    """ANN-Benchmarks recall: share of returned points no farther than the true k-th neighbour.

    `dists` must be exact distances of the returned points. Distance-based (not id-based),
    so ties at the k-th distance are not counted as errors.
    """
    return float(np.count_nonzero(dists <= gt_kth * (1.0 + TOL))) / k


def exact_dists(base, q, ids):
    return np.linalg.norm(base[ids] - q, axis=1)


def timing_row(times):
    t = np.asarray(times) * 1e3
    return dict(qps=1e3 / t.mean(), mean_ms=t.mean(), p50_ms=np.percentile(t, 50), p99_ms=np.percentile(t, 99))


def machine():
    info = dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                numpy=np.__version__, cpus=os.cpu_count(),
                threads={v: os.environ.get(v) for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")})
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        info["blas"] = f"{blas.get('name')} {blas.get('version')}"
    except Exception:
        info["blas"] = "unknown"
    return info


def write_csv(name, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    keys = list(dict.fromkeys(key for r in rows for key in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, "machine.json"), "w") as f:
        json.dump(machine(), f, indent=2)
    print("wrote", os.path.relpath(path, ROOT))
    return path


def nprobe_sweep(nlist):
    return [p for p in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if p < nlist]
