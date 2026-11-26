"""Exact search building blocks: distances by one BLAS call and top-k selection."""
import numpy as np


def squared_norms(X: np.ndarray) -> np.ndarray:
    """||x||^2 for every row. Computed once per vector, reused by every query."""
    return np.einsum("ij,ij->i", X, X)


def flat_scan(X: np.ndarray, norms: np.ndarray, q: np.ndarray, alive=None) -> np.ndarray:
    """Ranking distance from q to every row of X.

    ||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2. The last term is the same for all rows,
    so ranking only needs ||x||^2 - 2 x.q: one matrix-vector product (BLAS sgemv).
    Dead rows (alive == False) get +inf so they can never be selected.
    """
    dist = X @ q            # one BLAS call, float32
    dist *= -2.0
    dist += norms
    if alive is not None:
        dist[~alive] = np.inf
    return dist


def topk(dist: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k smallest finite values, ascending.

    argpartition is O(n) and only the k winners are sorted, instead of sorting all n.
    """
    n = dist.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if k < n:
        idx = np.argpartition(dist, k - 1)[:k]
    else:
        idx = np.arange(n)
    idx = idx[np.argsort(dist[idx], kind="stable")]
    return idx[np.isfinite(dist[idx])]
