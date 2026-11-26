"""k-means used to partition the vectors into IVF lists."""
import numpy as np

from .flat import squared_norms


def assign(V: np.ndarray, C: np.ndarray, cnorms: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid for every row of V.

    Works in chunks so the (chunk x nlist) distance matrix stays around 256 MB:
    a full N x nlist matrix for 1M x 4096 would need 16 GB.
    """
    k = C.shape[0]
    chunk = max(1024, 64_000_000 // k)
    out = np.empty(len(V), dtype=np.int32)
    for s in range(0, len(V), chunk):
        d = V[s:s + chunk] @ C.T          # (chunk, k) dot products, one BLAS sgemm
        d *= -2.0
        d += cnorms[None, :]
        out[s:s + chunk] = d.argmin(axis=1)
    return out


def kmeans(V: np.ndarray, k: int, iters: int = 10, seed: int = 0) -> np.ndarray:
    """Lloyd's algorithm with a fixed number of iterations.

    Init: k distinct random points. Update: rows are sorted by cluster once, then
    np.add.reduceat sums each contiguous group (no Python loop over clusters, no
    N x k x d array). A cluster that ends up empty is re-seeded with a random point.
    """
    rng = np.random.default_rng(seed)
    n = len(V)
    if k > n:
        raise ValueError(f"need at least k={k} training vectors, got {n}")
    C = V[rng.choice(n, k, replace=False)].astype(np.float32, copy=True)
    for _ in range(iters):
        a = assign(V, C, squared_norms(C))
        order = np.argsort(a, kind="stable")
        counts = np.bincount(a, minlength=k)
        nonempty = counts > 0
        starts = (np.cumsum(counts) - counts)[nonempty]
        sums = np.add.reduceat(V[order], starts, axis=0)
        C[nonempty] = sums / counts[nonempty, None]
        n_empty = int(k - nonempty.sum())
        if n_empty:
            C[~nonempty] = V[rng.choice(n, n_empty, replace=False)]
    return C
