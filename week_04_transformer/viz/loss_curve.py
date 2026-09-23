import argparse
import collections
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from viz.plot_style import apply_style, smooth, SERIES, INK, INK_MUTED

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"


# --------------------------------------------------------------------------- #
def ngram_floors(path=None, split=0.9):
    """Cross-entropy in nats of add-one smoothed n-gram models, fit on train,
    evaluated on val. Same units as F.cross_entropy, so directly comparable."""
    path = path or REPO_ROOT / "data" / "input.txt"
    text = Path(path).read_text(encoding="utf-8")
    n = int(split * len(text))
    train, val = text[:n], text[n:]
    V = len(set(text))

    c1 = collections.Counter(train); n1 = sum(c1.values())
    unigram = -sum(math.log((c1.get(ch, 0) + 1) / (n1 + V)) for ch in val) / len(val)

    c2 = collections.Counter(zip(train, train[1:])); ctx1 = collections.Counter(train[:-1])
    bigram = -sum(math.log((c2.get((a, b), 0) + 1) / (ctx1.get(a, 0) + V))
                  for a, b in zip(val, val[1:])) / (len(val) - 1)

    c3 = collections.Counter(zip(train, train[1:], train[2:]))
    ctx2 = collections.Counter(zip(train[:-1], train[1:-1]))
    trigram = -sum(math.log((c3.get((a, b, c), 0) + 1) / (ctx2.get((a, b), 0) + V))
                   for a, b, c in zip(val, val[1:], val[2:])) / (len(val) - 2)

    return {"uniform": math.log(V), "unigram": unigram, "bigram": bigram, "trigram": trigram}


# --------------------------------------------------------------------------- #
def classify(hist, evals, floors):
    """Apply the pattern table to the numbers. Returns (row, evidence dict)."""
    losses = [h["loss"] for h in hist]
    deltas = [abs(b - a) for a, b in zip(losses, losses[1:])]
    tail = losses[-max(50, len(losses) // 10):]
    tail_mean = sum(tail) / len(tail)
    tail_range = max(tail) - min(tail)
    first_mean = sum(losses[:10]) / 10
    mean_step = sum(deltas) / len(deltas)
    max_jump = max(deltas)
    val_series = [e["val"] for e in evals]
    val_min = min(val_series)
    val_min_step = evals[val_series.index(val_min)]["step"]
    final = evals[-1]
    gap = final["val"] - final["train"]

    ev = {
        "first10_mean": first_mean, "tail_mean": tail_mean, "tail_range": tail_range,
        "mean_abs_step_delta": mean_step, "max_single_jump": max_jump,
        "max_loss": max(losses), "final_train": final["train"], "final_val": final["val"],
        "gap": gap, "val_min": val_min, "val_min_step": val_min_step,
        "val_rise_from_min": final["val"] - val_min,
    }

    if any(not math.isfinite(v) for v in losses):
        return "diverging (NaN)", ev
    if ev["max_loss"] > 3 * first_mean:
        return "diverging", ev
    if max_jump > 1.0:
        return "spiking", ev
    if ev["val_rise_from_min"] > 0.15:
        return "down-then-up (overfitting)", ev
    if abs(tail_mean - floors["unigram"]) < 0.12:
        return "flat at the unigram floor", ev
    if first_mean - tail_mean < 0.3:
        return "flat", ev
    if mean_step > 0.07:
        return "jagged but trending down", ev
    return "smooth decrease then plateau", ev


# --------------------------------------------------------------------------- #
def plot(tag="run", show_floors=True, window=25, out=None):
    apply_style()
    data = json.loads((ARTIFACTS / f"{tag}_history.json").read_text())
    hist, evals = data["history"], data["evals"]
    cfg = data.get("config", {})

    steps = [h["step"] for h in hist]
    losses = [h["loss"] for h in hist]
    gnorms = [h["gnorm"] for h in hist]
    lrs = [h["lr"] for h in hist]
    ev_steps = [e["step"] for e in evals]

    floors = ngram_floors() if show_floors else None
    row, evidence = classify(hist, evals, floors or {"unigram": -1})

    fig, (ax, ax_g, ax_lr) = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1, 0.8], "hspace": 0.12})

    # ---- panel 1: loss -------------------------------------------------- #
    ax.plot(steps, losses, color=SERIES["train"], alpha=0.22, linewidth=0.9, zorder=1)
    ax.plot(steps, smooth(losses, window), color=SERIES["train"], linewidth=2.0,
            label=f"train (batch loss, {window}-step mean)", zorder=3)
    ax.plot(ev_steps, [e["val"] for e in evals], color=SERIES["val"], linewidth=2.0,
            marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.2,
            label="validation (held out, eval mode)", zorder=4)

    if show_floors:
        for name, value in floors.items():
            ax.axhline(value, color=INK_MUTED, linestyle=":", linewidth=1.0, alpha=0.75, zorder=2)
            ax.annotate(f"{name} {value:.2f}", xy=(steps[-1], value),
                        xytext=(4, 2), textcoords="offset points",
                        fontsize=8, color=INK_MUTED, va="bottom", ha="left")

    best = min(evals, key=lambda e: e["val"])
    ax.plot([best["step"]], [best["val"]], marker="o", markersize=9,
            markerfacecolor="none", markeredgecolor=SERIES["val"], markeredgewidth=1.8, zorder=5)
    ax.annotate(f"best val {best['val']:.3f} @ {best['step']}",
                xy=(best["step"], best["val"]), xytext=(6, 10), textcoords="offset points",
                fontsize=9, color=INK_MUTED)

    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title(f"Training loss — {cfg.get('n_layer','?')} layers, d_model "
                 f"{cfg.get('d_model','?')}, {cfg.get('norm','?')}-norm")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.02)

    # ---- panel 2: gradient norm ----------------------------------------- #
    ax_g.plot(steps, gnorms, color=SERIES["third"], alpha=0.25, linewidth=0.9)
    ax_g.plot(steps, smooth(gnorms, window), color=SERIES["third"], linewidth=1.8)
    ax_g.set_ylabel("grad norm\n(pre-clip)")
    ax_g.margins(x=0.02)

    # ---- panel 3: learning rate ----------------------------------------- #
    ax_lr.plot(steps, lrs, color=INK_MUTED, linewidth=1.8)
    ax_lr.set_ylabel("lr")
    ax_lr.set_xlabel("step")
    ax_lr.margins(x=0.02)

    out = Path(out) if out else ARTIFACTS / "04_loss_curve.png"
    fig.savefig(out)
    plt.close(fig)

    # ---- the numbers the write-up needs --------------------------------- #
    print(f"\nwritten -> {out}\n")
    print("PATTERN TABLE VERDICT")
    print("=" * 62)
    print(f"  your curve matches : {row}")
    print("  evidence:")
    for k, v in evidence.items():
        print(f"    {k:<24} {v:.4f}" if isinstance(v, float) else f"    {k:<24} {v}")
    if show_floors:
        print("\n  floors:")
        for k, v in floors.items():
            print(f"    {k:<24} {v:.4f}")
        print(f"\n  final val is {floors['trigram'] - evidence['final_val']:+.3f} vs the trigram floor "
              f"(positive = the model beat it)")

    print("\nRED FLAG CHECKLIST")
    print("=" * 62)
    checks = [
        ("starts near ln(V)",
         abs(evidence["first10_mean"] - floors["uniform"]) < 0.25 if show_floors else None),
        ("ends clearly below the unigram floor",
         evidence["final_val"] < floors["unigram"] - 0.3 if show_floors else None),
        ("beats the trigram floor",
         evidence["final_val"] < floors["trigram"] if show_floors else None),
        ("no single-step jump > 1.0", evidence["max_single_jump"] <= 1.0),
        ("no divergence (max loss < 3x the start)",
         evidence["max_loss"] < 3 * evidence["first10_mean"]),
        ("val is not rising from its minimum by > 0.15",
         evidence["val_rise_from_min"] <= 0.15),
        ("train/val gap under 0.5", abs(evidence["gap"]) < 0.5),
        ("raw per-step series is plotted, not only the smoothed line", True),
        ("both train and val are shown", True),
        ("floors are drawn as reference lines", show_floors),
    ]
    for label, ok in checks:
        mark = "ok  " if ok else "FLAG"
        print(f"  [{mark}] {label}")
    return out, row, evidence


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--no-floors", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    plot(tag=a.tag, show_floors=not a.no_floors, window=a.window, out=a.out)