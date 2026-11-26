"""Write-ahead log: every change is appended here before it is applied in memory.

Record layout (little endian):
    u32 payload_len | u32 crc32(payload) | payload
    payload = u64 seq | u8 op | u32 count | count x i64 ids | count x dim x f32 vectors (INSERT only)

The log stores operations on data. Index structures are never logged: they are derived.
"""
import os
import struct
import time
import zlib

import numpy as np

try:
    import fcntl
except ImportError:          # Windows
    fcntl = None

HEADER = struct.Struct("<II")
BODY = struct.Struct("<QBI")
OP_INSERT, OP_DELETE = 1, 2
SYNC_MODES = ("buffered", "os", "batch", "always")


def sync_fd(fd: int, full: bool = False) -> None:
    """Ask the OS to make the file durable.

    os.fsync flushes the OS page cache to the drive. On macOS the drive may still hold
    the data in its own volatile cache; F_FULLFSYNC asks the drive to flush that too.
    """
    if full and fcntl is not None and hasattr(fcntl, "F_FULLFSYNC"):
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    else:
        os.fsync(fd)


class WAL:
    """Append-only log with four durability modes.

    buffered: records wait in a Python list, written every `group` records. A process
              crash loses the buffer.
    os:       written to the OS immediately, never synced. Survives a process crash,
              not a power loss.
    batch:    written immediately, synced every `group` records or `group_ms` ms
              (group commit). Power loss can lose the last window.
    always:   written and synced before append() returns.
    """

    def __init__(self, path, sync="batch", full_sync=False, group=64, group_ms=50.0):
        if sync not in SYNC_MODES:
            raise ValueError(f"sync must be one of {SYNC_MODES}")
        self.path = path
        self.mode = sync
        self.full_sync = full_sync
        self.group = group
        self.group_s = group_ms / 1000.0
        self.f = open(path, "ab", buffering=0)     # unbuffered: write() goes straight to the OS
        self._buf = []
        self._pending = 0
        self._last_sync = time.perf_counter()

    def _write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[self.f.write(view):]

    def append(self, seq: int, op: int, ids: np.ndarray, vecs=None) -> None:
        payload = BODY.pack(seq, op, len(ids)) + ids.tobytes()
        if vecs is not None:
            payload += vecs.tobytes()
        rec = HEADER.pack(len(payload), zlib.crc32(payload)) + payload
        if self.mode == "buffered":
            self._buf.append(rec)
            self._pending += 1
            if self._pending >= self.group:
                self.flush()
            return
        self._write(rec)
        if self.mode == "always":
            sync_fd(self.f.fileno(), self.full_sync)
        elif self.mode == "batch":
            self._pending += 1
            now = time.perf_counter()
            if self._pending >= self.group or now - self._last_sync >= self.group_s:
                sync_fd(self.f.fileno(), self.full_sync)
                self._pending = 0
                self._last_sync = now

    def flush(self) -> None:
        """Hand buffered records to the OS (no durability guarantee yet)."""
        if self._buf:
            self._write(b"".join(self._buf))
            self._buf.clear()
        if self.mode == "buffered":
            self._pending = 0

    def sync(self) -> None:
        """Flush and force to disk. Called by checkpoint and close."""
        self.flush()
        sync_fd(self.f.fileno(), self.full_sync)
        self._pending = 0
        self._last_sync = time.perf_counter()

    def close(self) -> None:
        if not self.f.closed:
            self.sync()
            self.f.close()


def replay(path: str, dim: int, apply) -> int:
    """Call apply(seq, op, ids, vecs) for every intact record, in order.

    Stops at the first record that is cut short or fails its CRC: that is the torn tail
    left by a crash in the middle of a write. Returns the offset just past the last
    intact record so the caller can truncate the garbage.
    """
    good = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(HEADER.size)
            if len(head) < HEADER.size:
                break
            n, crc = HEADER.unpack(head)
            payload = f.read(n)
            if len(payload) < n or zlib.crc32(payload) != crc or n < BODY.size:
                break
            seq, op, count = BODY.unpack_from(payload, 0)
            p = BODY.size
            ids = np.frombuffer(payload, dtype=np.int64, count=count, offset=p)
            vecs = None
            if op == OP_INSERT:
                p += 8 * count
                vecs = np.frombuffer(payload, dtype=np.float32, count=count * dim, offset=p).reshape(count, dim)
            apply(seq, op, ids, vecs)
            good = f.tell()
    return good
