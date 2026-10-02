
import json
from pathlib import Path

import matplotlib.pyplot as plt

from demos.constants import RESULTS_DIR

LOG_PATH = Path(RESULTS_DIR) / "optimizer.jsonl"
OUT_PATH = Path("artifacts") / "optimizer_versions.png"

# version_id -> accuracy on the 200 test questions (only versions you scored on test)
TEST_SCORES = {0: 0.00, 1: 0.53}

# small sideways nudge so the three markers on one version never sit on top of each other
DEV_DX, FMT_DX, TEST_DX = -0.15, 0.15, 0.0

# colors: one per measure, plus neutral inks (validated: colorblind-safe pair)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def load_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    records = load_log(LOG_PATH)

    # x order: baseline first, then by round, then by version id inside a round
    records.sort(key=lambda r: (r["round"], r["version_id"]))
    xs = list(range(len(records)))

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # round bands: light alternating background so each round reads as one group
    rounds = sorted({r["round"] for r in records})
    for rd in rounds:
        idx = [i for i, r in enumerate(records) if r["round"] == rd]
        if rd % 2 == 1:
            ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, color="#f1f0eb", zorder=0)
        label = "baseline" if rd == 0 else f"round {rd}"
        ax.text((min(idx) + max(idx)) / 2, 1.04, label, ha="center", va="bottom",
                fontsize=9, color=MUTED, transform=ax.get_xaxis_transform())

    # best-so-far line: the dev accuracy of the current best prompt after each candidate
    best_curve, best = [], None
    for r in records:
        if r["status"] in ("baseline", "accepted"):
            best = r["dev_acc"]
        best_curve.append(best)
    ax.step(xs, best_curve, where="post", color=BLUE, linewidth=2, alpha=0.35,
            label="dev accuracy of current best", zorder=1)

    # one point per scored candidate: dev accuracy (filled) and format pass rate (hollow)
    for x, r in zip(xs, records):
        if r["dev_acc"] is None:          # rejected_check: never scored
            ax.scatter(x, 0.02, marker="x", s=60, color=MUTED, linewidths=2, zorder=3)
            ax.annotate(r["reason"].split()[0].lower(), (x, 0.02), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8, color=MUTED)
            continue
        ax.scatter(x + DEV_DX, r["dev_acc"], s=70, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=4)
        ax.scatter(x + FMT_DX, r["format_pass_rate"], s=70, marker="s", facecolors=SURFACE,
                   edgecolors=ORANGE, linewidths=2, zorder=4)
        if r["status"] == "accepted":
            ax.annotate(f"accepted v{r['version_id']}\nb={r['b']}, c={r['c']}, p={r['p']:.1g}",
                        (x + DEV_DX, r["dev_acc"]), xytext=(30, 60), textcoords="offset points",
                        fontsize=9, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))

    # test scores as a second marker on the same version
    for x, r in zip(xs, records):
        if r["version_id"] in TEST_SCORES:
            t = TEST_SCORES[r["version_id"]]
            ax.scatter(x + TEST_DX, t, s=90, marker="D", color=INK, edgecolors=SURFACE, linewidths=2, zorder=5)
            ax.annotate(f"test {t:.2f}", (x + TEST_DX, t), xytext=(-8, 10), ha="right", textcoords="offset points",
                        fontsize=9, color=INK)

    # legend proxies (so marker shapes are explained even without color)
    ax.scatter([], [], s=70, color=BLUE, label="dev accuracy")
    ax.scatter([], [], s=70, marker="s", facecolors=SURFACE, edgecolors=ORANGE, linewidths=2,
               label="format pass rate (dev)")
    ax.scatter([], [], s=90, marker="D", color=INK, label="test accuracy")
    ax.scatter([], [], s=60, marker="x", color=MUTED, linewidths=2, label="rejected by check (not scored)")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"v{r['version_id']}" for r in records], fontsize=9, color=MUTED)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of questions", color=MUTED)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.set_title("Prompt optimizer: every version's dev score, in the order it was tried",
                 loc="left", fontsize=12, color=INK, pad=24)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9,
              labelcolor=INK)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=SURFACE)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()