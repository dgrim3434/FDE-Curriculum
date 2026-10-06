"""Week 7 required chart: validation outcome by schema, local vs API.

Reads  results/extraction_summary.json   (written by your evaluation script)
Writes artifacts/validation_outcomes.png  (the chart for the submission)
Prints the same numbers as a table        (paste it under the chart in the write-up)

Run from week07/:   python -m experiments.plot_outcomes
Re-run it any time the summary file changes (e.g. after a Graph rerun).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

WEEK = Path(__file__).resolve().parent.parent
SUMMARY = WEEK / "results" / "extraction_summary.json"
OUT = WEEK / "artifacts" / "validation_outcomes.png"

# x-axis order = complexity ladder, so the chart shows WHERE reliability drops
LADDER = ["EntityTypes", "Extraction", "Event", "Graph"]
LADDER_LABELS = ["EntityTypes\n(flat)", "Extraction\n(nested)",
                 "Event\n(+ optionals)", "Graph\n(+ relations)"]

# Stack order bottom -> top: good at the base, failure on top where the eye lands
BANDS = ["first_pass", "repaired", "retry", "failed"]
BAND_LABELS = {
    "first_pass": "Valid on first pass",
    "repaired": "Repaired in code (no extra call)",
    "retry": "Valid after retry (extra calls)",
    "failed": "Failed after all retries",
}
BAND_COLORS = {                            # colorblind-checked as a set
    "first_pass": "#0ca30c",               # green  - nothing needed
    "repaired": "#2a78d6",                 # blue   - fixed for free
    "retry": "#eda100",                    # yellow - fixed, but cost model calls
    "failed": "#d03b3b",                   # red    - lost
}

# Panel titles. Keys must match the model names in the summary file.
TIER_TITLES = {
    "qwen2.5:3b": "Local · qwen2.5:3b (Ollama)",
    "claude-haiku-4-5-20251001": "API · claude-haiku-4-5",
}

SURFACE = "#fcfcfb"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def band_shares(entry):
    """The four outcome shares for one schema. Missing bands (never happened) are 0."""
    shares = [entry.get(b, 0.0) for b in BANDS]
    total = sum(shares)
    if abs(total - 1.0) > 0.01:
        print(f"  WARNING: bands add up to {total:.3f}, not 1.0 - check the summary file")
    return shares


def draw_panel(ax, tier_data, title):
    xs = range(len(LADDER))
    bottoms = [0.0] * len(LADDER)
    shares_by_schema = [band_shares(tier_data[s]) for s in LADDER]

    for b_idx, band in enumerate(BANDS):
        heights = [shares[b_idx] for shares in shares_by_schema]
        ax.bar(xs, heights, bottom=bottoms, width=0.5,
               color=BAND_COLORS[band],
               edgecolor=SURFACE, linewidth=2)          # 2px surface gap between segments

        # Label a segment only when it is tall enough to hold the text
        for x, h, b0 in zip(xs, heights, bottoms):
            if h >= 0.08:
                text_color = INK if band == "retry" else "white"   # dark text on yellow
                ax.text(x, b0 + h / 2, f"{h:.0%}", ha="center", va="center",
                        fontsize=9, color=text_color)
        bottoms = [b0 + h for b0, h in zip(bottoms, heights)]

    # n above each bar (the write-up must state n per schema)
    for x, schema in zip(xs, LADDER):
        ax.text(x, 1.02, f"n={tier_data[schema].get('n', '?')}", ha="center", va="bottom",
                fontsize=8.5, color=MUTED)

    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=18)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(LADDER_LABELS, fontsize=9, color=INK_2)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_facecolor(SURFACE)


def print_table(summary):
    """The same numbers as the chart, as a markdown table (the chart's table view)."""
    print("| Tier | Schema | n | First pass | Repaired | Retry | Failed |")
    print("|---|---|---|---|---|---|---|")
    for tier, tier_data in summary.items():
        for schema in LADDER:
            e = tier_data[schema]
            cells = " | ".join(f"{e.get(b, 0.0):.0%}" for b in BANDS)
            print(f"| {tier} | {schema} | {e.get('n', '?')} | {cells} |")


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    tiers = list(summary)                                  # panel order = file order

    fig, axes = plt.subplots(1, len(tiers), figsize=(11, 5.2), sharey=True)
    if len(tiers) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for ax, tier in zip(axes, tiers):
        draw_panel(ax, summary[tier], TIER_TITLES.get(tier, tier))

    handles = [Patch(color=BAND_COLORS[b], label=BAND_LABELS[b]) for b in BANDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("What happened to each model output, by schema complexity",
                 x=0.06, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}\n")
    print_table(summary)


if __name__ == "__main__":
    main()