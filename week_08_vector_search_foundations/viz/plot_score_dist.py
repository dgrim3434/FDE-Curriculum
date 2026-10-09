"""Visualization A — retrieval score distributions (relevant vs irrelevant), small multiples.

Run from week_08_vector_search_foundations/:
    python -m viz.plot_score_dist
Reads:  results/score_dist_{tag}.json
Writes: artifacts/score_distribution.png, results/viz_a_summary.json
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
TAGS = ["minilm", "mpnet", "bge", "bge_noprefix", "e5", "e5_noprefix"]

IRR_COLOR = "#0072B2"   # blue   (Okabe-Ito: colorblind- and greyscale-safe)
REL_COLOR = "#E69F00"   # orange
N_BINS = 60


def load_scores(tag):
    data = json.loads((RESULTS / f"score_dist_{tag}.json").read_text(encoding="utf-8"))
    rel = np.array([s for q in data.values() for s in q["relevant"]], dtype=np.float64)
    irr = np.array([s for q in data.values() for s in q["irrelevant"]], dtype=np.float64)
    # float32 rounding can put a cosine a hair above 1.0, outside the last bin edge
    return np.clip(rel, -1.0, 1.0), np.clip(irr, -1.0, 1.0), len(data)


def auc(rel, irr):
    """P(random relevant score > random irrelevant score); ties count half."""
    gt = (rel[:, None] > irr[None, :]).mean()
    eq = (rel[:, None] == irr[None, :]).mean()
    return float(gt + 0.5 * eq)


def mode_of(x, bins):
    counts, edges = np.histogram(x, bins=bins)
    i = counts.argmax()
    return float((edges[i] + edges[i + 1]) / 2)


def overlap(rel, irr, bins):
    """Shared area under the two density curves: 0 = fully separated, 1 = identical."""
    d_rel, edges = np.histogram(rel, bins=bins, density=True)
    d_irr, _ = np.histogram(irr, bins=bins, density=True)
    return float(np.sum(np.minimum(d_rel, d_irr) * np.diff(edges)))


def summarize(rel, irr, n_queries, bins):
    thr = float(np.percentile(irr, 95))              # candidate threshold
    return {
        "n_queries": n_queries,
        "n_relevant": int(rel.size),
        "n_irrelevant": int(irr.size),
        "bin_width": float(bins[1] - bins[0]),
        "mean_relevant": float(rel.mean()),
        "mean_irrelevant": float(irr.mean()),
        "mode_relevant": mode_of(rel, bins),
        "mode_irrelevant": mode_of(irr, bins),
        "std_relevant": float(rel.std()),
        "std_irrelevant": float(irr.std()),
        "gap": float(rel.mean() - irr.mean()),
        "gap_in_irr_std": float((rel.mean() - irr.mean()) / irr.std()),
        "auc": auc(rel, irr),
        "overlap": overlap(rel, irr, bins),
        "threshold_irrelevant_p95": thr,
        "frac_relevant_kept_at_threshold": float((rel > thr).mean()),
        "frac_relevant_lost_at_threshold": float((rel <= thr).mean()),
    }


def main():
    loaded = {tag: load_scores(tag) for tag in TAGS}
    all_scores = np.concatenate([np.concatenate([r, i]) for r, i, _ in loaded.values()])
    lo = np.floor(all_scores.min() * 20) / 20          # round down to the nearest 0.05
    bins = np.linspace(lo, 1.0, N_BINS + 1)             # ONE bin array for every series and panel

    summary = {tag: summarize(*loaded[tag], bins) for tag in TAGS}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    for ax, tag in zip(axes.ravel(), TAGS):
        rel, irr, _ = loaded[tag]
        s = summary[tag]
        for data, color, name in [(irr, IRR_COLOR, "irrelevant"), (rel, REL_COLOR, "relevant")]:
            ax.hist(data, bins=bins, density=True, color=color, alpha=0.30)
            ax.hist(data, bins=bins, density=True, color=color, histtype="step", lw=1.6,
                    label=f"{name} (n={data.size:,})")
            ax.axvline(data.mean(), color=color, ls=":", lw=1.2)
        ax.axvline(s["threshold_irrelevant_p95"], color="black", ls="--", lw=1.2,
                   label="candidate threshold (irrelevant p95)")
        ax.set_title(tag, fontsize=12)
        ax.text(0.02, 0.97,
                f"AUC {s['auc']:.3f}\n"
                f"gap {s['gap']:.3f} ({s['gap_in_irr_std']:.1f}σ)\n"
                f"overlap {s['overlap']:.2f}\n"
                f"kept at threshold {s['frac_relevant_kept_at_threshold']:.0%}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.legend(fontsize=7.5, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("cosine similarity (query, document)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density (each series normalized to area 1)")
    n_q = summary[TAGS[0]]["n_queries"]
    fig.suptitle(f"SciFact: score distributions, labeled-relevant vs irrelevant documents — "
                 f"{n_q} test queries, exact brute-force search, bin width {bins[1]-bins[0]:.3f}",
                 fontsize=12)
    fig.tight_layout()

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / "score_distribution.png"
    fig.savefig(out, dpi=150)
    (RESULTS / "viz_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved {out}\n")
    print(f"{'tag':14s} {'AUC':>6s} {'gap σ':>6s} {'overlap':>8s} {'mode rel':>9s} "
          f"{'mode irr':>9s} {'thr':>6s} {'kept':>6s}")
    for tag in TAGS:
        s = summary[tag]
        print(f"{tag:14s} {s['auc']:6.3f} {s['gap_in_irr_std']:6.1f} {s['overlap']:8.2f} "
              f"{s['mode_relevant']:9.3f} {s['mode_irrelevant']:9.3f} "
              f"{s['threshold_irrelevant_p95']:6.3f} {s['frac_relevant_kept_at_threshold']:6.0%}")


if __name__ == "__main__":
    main()