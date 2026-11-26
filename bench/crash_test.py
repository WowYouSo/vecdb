"""Kill the database process with SIGKILL in the middle of inserts and see what survives.

A child process inserts one vector at a time and prints the id only AFTER insert()
returns: a printed id is an acknowledged write. The parent kills the child at a random
moment, reopens the database and checks which acknowledged ids are still there.

What this can and cannot show: SIGKILL destroys the process memory but not the OS page
cache. So it separates `buffered` (acknowledged writes can be lost) from the other
modes. Each record is written with a single write() call, so SIGKILL never leaves half
a record; torn and corrupted tails are covered by tests/test_vecdb.py. It cannot
separate `os` from `always`: only a power loss or kernel crash would, and we do not
simulate that.
"""
import common  # noqa: F401  (must be first)

import argparse
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from common import write_csv
from vecdb import VectorDB

DIM = 128


def child(path, mode):
    db = VectorDB.open(path, DIM, sync=mode)
    rng = np.random.default_rng(1)
    V = rng.standard_normal((1024, DIM), dtype=np.float32)
    out = sys.stdout
    i = 0
    while True:
        db.insert(i, V[i % 1024])
        out.write(f"{i}\n")          # acknowledged
        out.flush()
        i += 1


def run_trial(mode, seconds):
    path = tempfile.mkdtemp(prefix="vecdb-crash-")
    ack_path = os.path.join(path, "acks.txt")
    db_path = os.path.join(path, "db")
    with open(ack_path, "wb") as ack:
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", db_path, mode], stdout=ack)
        time.sleep(seconds)
        p.send_signal(signal.SIGKILL)
        p.wait()
    with open(ack_path, "rb") as f:
        data = f.read()
    acked = [int(x) for x in data[:data.rfind(b"\n") + 1].split()]
    db = VectorDB.open(db_path, DIM, sync="os")
    lost = sum(1 for i in acked if i not in db)
    row = dict(mode=mode, acknowledged=len(acked), recovered=len(db), lost_acknowledged=lost,
               recovered_unacknowledged=len(db) - (len(acked) - lost), torn_bytes=db.recovery["torn_bytes"])
    db.close()
    shutil.rmtree(path, ignore_errors=True)
    return row


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["buffered", "os", "batch", "always"])
    args = ap.parse_args()
    random.seed(0)
    rows = []
    print("mode      acknowledged  recovered  LOST acknowledged  torn bytes")
    for mode in args.modes:
        for _ in range(args.trials):
            r = run_trial(mode, random.uniform(0.6, 1.4))
            rows.append(r)
            print(f"{mode:9s} {r['acknowledged']:12d} {r['recovered']:10d} {r['lost_acknowledged']:18d} {r['torn_bytes']:11d}")
    write_csv("crash.csv", rows)


if __name__ == "__main__":
    main()
