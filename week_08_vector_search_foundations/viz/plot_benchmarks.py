"""Visualization B — benchmark curves.

B1: index recall@10 vs median query latency as nprobe varies (1 and 8 threads); brute force = reference
B2: query latency vs corpus size, log-log, median line with p50–p95 band, brute force vs IVF (nprobe=8)
B3: index recall@10 vs % of corpus scanned — real Quora embeddings vs random-vector control

Run from week_08_vector_search_foundations/:
    python -m viz.plot_benchmarks
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

THREADS = ["1", "8"]
K = 10                                   # index recall@K plotted everywhere
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#555555"
LINE = {"1": "-", "8": "--"}
MARK = {"1": "o", "8": "s"}


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def operating_points(rows, bf_p50, targets=(0.90, 0.95, 0.99)):
    """Smallest nprobe reaching each index-recall target, with its latency and speedup."""
    out = {}
    for t in targets:
        hits = [r for r in rows if r[f"index_recall@{K}"] >= t]
        if not hits:
            out[f"index_recall@{K}>={t}"] = None
            continue
        r = min(hits, key=lambda r: r["nprobe"])
        out[f"index_recall@{K}>={t}"] = {
            "nprobe": r["nprobe"],
            "p50_ms": float(r["p50_latency"]),
            "p95_ms": float(r["p95_latency"]),
            "pct_corpus_scanned": float(r["perc_corpus_scanned"]),
            "speedup_vs_brute_force_p50": float(bf_p50 / r["p50_latency"]),
        }
    return out


def loglog_slope(x, y):
    """Slope of log10(y) vs log10(x): the growth exponent (1.0 = linear in N)."""
    return float(np.polyfit(np.log10(x), np.log10(y), 1)[0])


def main():
    summary = {"k": K, "B1": {}, "B2": {}, "B3": {}}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.2))

    # ───────────────────────── B1: recall vs latency ─────────────────────────
    for t in THREADS:
        d = load(f"bench_nprobe_t{t}.json")
        rows = sorted(d["experiment"], key=lambda r: r["nprobe"])
        x = [r["p50_latency"] for r in rows]
        y = [r[f"index_recall@{K}"] for r in rows]
        ax1.plot(x, y, LINE[t], marker=MARK[t], color=ORANGE, lw=1.6, ms=5,
                 label=f"IVF, {t} thread{'s' if t != '1' else ''}")
        if t == "1":                                         # label points once to avoid clutter
            for r, xi, yi in zip(rows, x, y):
                ax1.annotate(str(r["nprobe"]), (xi, yi), textcoords="offset points",
                             xytext=(4, -11), fontsize=7.5, color=GREY)
        bf = d["brute_force"]
        ax1.scatter([bf["p50_latency"]], [1.0], marker="*", s=180, color=BLUE,
                    edgecolor="black", zorder=5, label=f"brute force (exact), {t} thr")
        summary["B1"][f"threads_{t}"] = {
            "brute_force_p50_ms": float(bf["p50_latency"]),
            "brute_force_p95_ms": float(bf["p95_latency"]),
            "operating_points": operating_points(rows, bf["p50_latency"]),
        }
    cfg = load("bench_nprobe_t1.json")["config"]
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.03)
    ax1.set_xlabel("median query latency (ms, log scale)")
    ax1.set_ylabel(f"index recall@{K}  (overlap with exact brute-force top-{K})")
    ax1.set_title(f"B1 — what approximate search costs\n"
                  f"Quora N={cfg['N']:,}, nlist={cfg['nlist']}, labels = nprobe", fontsize=10)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="lower right")

    # ───────────────────────── B2: latency vs N ─────────────────────────
    for t in THREADS:
        rows = sorted(load(f"bench_scaling_t{t}.json")["experiment"], key=lambda r: r["N"])
        N = np.array([r["N"] for r in rows], dtype=float)
        series = {
            "brute force": (BLUE, [r["brute_latency_p50"] for r in rows],
                                  [r["brute_latency_p95"] for r in rows]),
            "IVF nprobe=8": (ORANGE, [r["ivf_latency_p50"] for r in rows],
                                     [r["ivf_latency_p95"] for r in rows]),
        }
        res = {}
        for name, (color, p50, p95) in series.items():
            p50, p95 = np.array(p50), np.array(p95)
            ax2.plot(N, p50, LINE[t], marker=MARK[t], color=color, lw=1.6, ms=5,
                     label=f"{name}, {t} thr (p50)")
            ax2.fill_between(N, p50, p95, color=color, alpha=0.12)
            big = N >= 50_000
            res[name] = {
                "loglog_slope_all_N": loglog_slope(N, p50),
                "loglog_slope_N>=50k": loglog_slope(N[big], p50[big]) if big.sum() >= 2 else None,
                "p50_ms_by_N": {int(n): float(v) for n, v in zip(N, p50)},
                "p95_over_p50_by_N": {int(n): float(b / a) for n, a, b in zip(N, p50, p95)},
            }
        bf50 = np.array(series["brute force"][1])
        iv50 = np.array(series["IVF nprobe=8"][1])
        faster = np.flatnonzero(iv50 < bf50)
        res["crossover_first_N_where_ivf_p50_faster"] = int(N[faster[0]]) if faster.size else None
        res["ivf_index_recall@10_by_N"] = {int(r["N"]): float(r["index_recall@10"]) for r in rows}
        res["ivf_build_s_by_N"] = {int(r["N"]): float(r["ivf_build_time"]) for r in rows}
        summary["B2"][f"threads_{t}"] = res
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("indexed vectors N (log scale)")
    ax2.set_ylabel("query latency (ms, log scale) — line p50, band p50→p95")
    ax2.set_title("B2 — how latency grows with corpus size\n"
                  "nlist = √N, nprobe = 8, 500 queries per point", fontsize=10)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7.5, loc="upper left")

    # ───────────────────────── B3: the control ─────────────────────────
    real = sorted(load("bench_nprobe_t1.json")["experiment"], key=lambda r: r["nprobe"])
    rand = sorted(load("rand_nprobe_t1.json")["experiments"], key=lambda r: r["nprobe"])
    for rows, color, name in [(real, ORANGE, "real Quora embeddings"),
                              (rand, BLUE, "random Gaussian vectors (control)")]:
        x = [r["perc_corpus_scanned"] for r in rows]
        y = [r[f"index_recall@{K}"] for r in rows]
        ax3.plot(x, y, "-", marker="o", color=color, lw=1.6, ms=5, label=name)
        summary["B3"][name] = [{"nprobe": r["nprobe"], "pct_scanned": float(r["perc_corpus_scanned"]),
                                f"index_recall@{K}": float(r[f"index_recall@{K}"])} for r in rows]
    diag = np.logspace(np.log10(min(r["perc_corpus_scanned"] for r in real + rand)), 2, 50)
    ax3.plot(diag, diag / 100, ":", color=GREY, label="recall = fraction scanned (no structure)")
    ax3.set_xscale("log")
    ax3.set_ylim(0, 1.03)
    ax3.set_xlabel("% of corpus scanned per query (log scale)")
    ax3.set_ylabel(f"index recall@{K}")
    ax3.set_title("B3 — does the data's structure do the work?\n"
                  "same N, nlist, train_size, nprobe sweep; 1 thread", fontsize=10)
    ax3.grid(alpha=0.3, which="both")
    ax3.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"IVF vs exact brute force — all-MiniLM-L6-v2 (d={cfg['d']}), "
                 f"index recall measured against brute force, NOT against relevance labels",
                 fontsize=11)
    fig.tight_layout()
    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / "benchmark_curves.png"
    fig.savefig(out, dpi=150)
    (RESULTS / "viz_b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved {out}\n")
    for t in THREADS:
        b1 = summary["B1"][f"threads_{t}"]
        print(f"[{t} thread] brute force p50 {b1['brute_force_p50_ms']:.2f} ms")
        for target, op in b1["operating_points"].items():
            print(f"   {target}: " + ("not reached" if op is None else
                  f"nprobe={op['nprobe']}, p50 {op['p50_ms']:.2f} ms, "
                  f"{op['pct_corpus_scanned']:.1f}% scanned, {op['speedup_vs_brute_force_p50']:.1f}x"))
        b2 = summary["B2"][f"threads_{t}"]
        print(f"   slopes  brute {b2['brute force']['loglog_slope_all_N']:.2f} "
              f"(N>=50k: {b2['brute force']['loglog_slope_N>=50k']:.2f})   "
              f"IVF {b2['IVF nprobe=8']['loglog_slope_all_N']:.2f}   "
              f"crossover N = {b2['crossover_first_N_where_ivf_p50_faster']}")


if __name__ == "__main__":
    main()