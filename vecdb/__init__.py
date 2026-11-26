"""vecdb: a small single-node vector database (exact + IVF-Flat search, WAL, snapshots)."""
from .db import VectorDB

__all__ = ["VectorDB"]
