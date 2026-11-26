"""Turn results/*.csv into plots and results/summary.md."""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"      # fixed categorical order: ours, FAISS IVF, FAISS HNSW

plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160, "legend.frameon": False,
})


def read(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key, v in r.items():
            try:
                r[key] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def titled(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=22)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)


def plot_recall_qps(rows):
    systems = list(dict.fromkeys(r["system"] for r in rows))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    dash_of = {}
    for s in systems:
        pts = sorted((r for r in rows if r["system"] == s), key=lambda r: r["recall"])
        fam = s.split()[0]
        if "flat" in s:
            color = BLUE if fam == "vecdb" else ORANGE
            ax.plot([pts[0]["recall"]], [pts[0]["qps"]], "s", ms=8, color=color, mec="white", mew=1.2, label=f"{s} (exact)", zorder=4)
            ax.annotate(f"{s}: {pts[0]['qps']:.0f} QPS", (pts[0]["recall"], pts[0]["qps"]), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK)
            continue
        color, marker = (AQUA, "D") if "hnsw" in s else ((BLUE, "o") if fam == "vecdb" else (ORANGE, "^"))
        n_before = dash_of.setdefault(fam + ("hnsw" if "hnsw" in s else ""), [])
        style = "-" if not n_before else "--"          # second nlist of the same system is dashed
        n_before.append(s)
        ax.plot([p["recall"] for p in pts], [p["qps"] for p in pts], style, color=color, lw=2, marker=marker,
                ms=6, mec="white", mew=1, label=s)
    ax.set_yscale("log")
    lo = min(r["recall"] for r in rows)
    ax.set_xlim(max(0.0, lo - 0.03), 1.012)
    ax.set_xlabel("recall@10 (share of the true 10 nearest neighbours found)")
    ax.set_ylabel("queries per second, one thread (log scale)")
    r0 = rows[0]
    titled(ax, "Recall vs throughput", f"{r0.get('dataset', '')}, k = {int(r0.get('k', 10))}, one query at a time. Up and to the right is better.")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "recall_qps.png"))
    plt.close(fig)


def plot_writes(rows):
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(rows) + 1.6))
    labels = [f"{r['mode']}" + ("" if r["sync_call"] == "none" else f" ({r['sync_call']})") for r in rows]
    y = list(range(len(rows)))[::-1]
    x = [r["inserts_per_s"] for r in rows]
    ax.hlines(y, min(x) / 3, x, color=GRID, lw=1.2)
    ax.plot(x, y, "o", ms=9, color=BLUE, mec="white", mew=1.2)
    for xi, yi in zip(x, y):
        ax.annotate(f"{xi:,.0f}/s", (xi, yi), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(min(x) / 3, max(x) * 6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("single-vector inserts per second (log scale)")
    titled(ax, "What durability costs", "128-dim vectors, one insert = one log record. More durable modes are lower.")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "writes.png"))
    plt.close(fig)


def wrap_stage(s):
    if s.startswith("+ 10%"):
        return "+ 10% of rows\ndeleted"
    return s.replace(" (", "\n(").replace(" of ", "\nof ")


def plot_delta(rows):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = range(len(rows))
    y = [r["mean_ms"] for r in rows]
    ax.bar(x, y, width=0.55, color=BLUE)
    for xi, r in zip(x, rows):
        ax.annotate(f"{r['mean_ms']:.2f} ms\nrecall {r['recall']:.3f}", (xi, r["mean_ms"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([wrap_stage(r["stage"]) for r in rows], fontsize=8)
    ax.set_ylim(0, max(y) * 1.22)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("ms per query (default nprobe)")
    r0 = rows[0]
    titled(ax, "Inserts and deletes after the build, then compaction",
           f"{r0.get('dataset', '')}, nlist = {int(r0['nlist'])}, nprobe = {int(r0['nprobe'])}. The delta is scanned in full on every query.")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "delta.png"))
    plt.close(fig)


def best_at(rows, system, target):
    ok = [r for r in rows if r["system"] == system and r["recall"] >= target]
    if not ok:
        return "not reached"
    b = max(ok, key=lambda r: r["qps"])
    return f"{b['qps']:,.0f} QPS ({b['mean_ms']:.2f} ms, param {int(b['param'])})"


def table(header, lines):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, l)) + " |" for l in lines])


def main():
    search, faiss, writes, rec, crash, delta = (read(n) for n in ("search.csv", "faiss.csv", "writes.csv", "recovery.csv", "crash.csv", "delta.csv"))
    out = ["# Results\n"]
    mpath = os.path.join(RESULTS, "machine.json")
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        out.append(f"Machine: {m['platform']} ({m['machine']}), {m['cpus']} CPUs, Python {m['python']}, NumPy {m['numpy']}, BLAS {m['blas']}. Query timings use one thread.\n")
    both = search + faiss
    if both:
        plot_recall_qps(both)
        out.append(f"## Search ({both[0].get('dataset', '')})\n\n![recall vs QPS](recall_qps.png)\n")
        systems = list(dict.fromkeys(r["system"] for r in both))
        lines = []
        for s in systems:
            if "flat" in s:
                r = next(r for r in both if r["system"] == s)
                lines.append([s, f"{r['qps']:,.0f} QPS ({r['mean_ms']:.2f} ms), exact", "same", "same"])
            else:
                lines.append([s, best_at(both, s, 0.90), best_at(both, s, 0.95), best_at(both, s, 0.99)])
        out.append(table(["system", "recall >= 0.90", "recall >= 0.95", "recall >= 0.99"], lines) + "\n")
        ours = [r for r in search if "ivf" in r["system"]]
        if ours:
            seen, lines = set(), []
            for r in ours:
                if r["system"] not in seen:
                    seen.add(r["system"])
                    lines.append([r["system"], f"{r['train_s']:.1f}s", f"{r['add_s']:.1f}s", f"{r['imbalance']:.2f}", f"{r['index_bytes_per_vec']:.1f}"])
            out.append("Our index build:\n\n" + table(["index", "k-means", "assign + sort", "list imbalance", "index bytes per vector"], lines) + "\n")
            lines = [[r["system"], int(r["param"]), f"{r['recall']:.4f}", f"{r['mean_ms']:.3f}", f"{r['p99_ms']:.3f}", f"{r['qps']:,.0f}", f"{100 * r['scanned_share']:.2f}%"] for r in ours]
            out.append("Full sweep of our index:\n\n" + table(["index", "nprobe", "recall@10", "mean ms", "p99 ms", "QPS", "rows scanned"], lines) + "\n")
    if writes:
        plot_writes(writes)
        out.append("## Write throughput by sync mode\n\n![writes](writes.png)\n")
        out.append(table(["mode", "sync call", "inserts/s", "microseconds per insert"],
                         [[r["mode"], r["sync_call"], f"{r['inserts_per_s']:,.0f}", f"{r['us_per_insert']:.1f}"] for r in writes]) + "\n")
    if rec:
        out.append("## Recovery time\n\n" + table(["scenario", "vectors", "seconds", "log records replayed"],
                   [[r["scenario"], int(r["vectors"]), f"{r['open_s']:.2f}", "" if r.get("replayed_records", "") == "" else int(r["replayed_records"])] for r in rec]) + "\n")
    if crash:
        out.append("## Crash test (SIGKILL during inserts)\n\n" + table(["mode", "acknowledged", "recovered", "lost acknowledged", "torn bytes"],
                   [[r["mode"], int(r["acknowledged"]), int(r["recovered"]), int(r["lost_acknowledged"]), int(r["torn_bytes"])] for r in crash]) + "\n")
    if delta:
        plot_delta(delta)
        out.append("## Changes after the build\n\n![delta](delta.png)\n")
        out.append(table(["stage", "ms/query", "recall@10", "delta rows", "dead rows", "stage took"],
                         [[r["stage"], f"{r['mean_ms']:.3f}", f"{r['recall']:.4f}", int(r["delta_rows"]), int(r["sealed_dead"]),
                           f"{r['stage_seconds']:.2f}s" if r["stage_seconds"] else ""] for r in delta]) + "\n")
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(out))
    print("wrote results/summary.md and plots")


if __name__ == "__main__":
    main()
