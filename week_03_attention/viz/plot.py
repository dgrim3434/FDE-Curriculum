"""Plot primitives. Every function takes a tensor, returns an Axes/Figure, saves nothing."""
import math
import numpy as np
import matplotlib.pyplot as plt
from viz.style import SEQ, DIV, SERIES, INK_MUTED
from viz import metrics

def attention_heatmap(w, tokens, ax, title="", vmax=1.0, cmap=SEQ, annotate=False):
    """w: (T_q, T_k). Rows = queries (attending FROM), cols = keys (attended TO)."""
    a = w.detach().cpu().numpy()
    im = ax.imshow(a, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(a.shape[1])); ax.set_xticklabels(tokens[:a.shape[1]], rotation=90)
    ax.set_yticks(range(a.shape[0])); ax.set_yticklabels(tokens[:a.shape[0]])
    ax.set_xlabel("key — attended TO"); ax.set_ylabel("query — attending FROM")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-.5, a.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, a.shape[0], 1), minor=True)
    ax.grid(which="minor", color=ax.get_facecolor(), linewidth=1.5)  # cell separators
    ax.tick_params(which="minor", length=0)
    if annotate:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] >= 0.005:
                    ax.text(j, i, f"{a[i,j]:.2f}", ha="center", va="center", fontsize=6,
                            color="#ffffff" if a[i, j] > 0.55 else INK_MUTED)
    return im

def head_grid(w_heads, tokens, suptitle, cols=4, annotate=False):
    """w_heads: (h, T_q, T_k) - one panel per head, shared 0..1 scale."""
    h = w_heads.shape[0]
    cols = min(h, cols); rows = math.ceil(h / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows), squeeze=False)
    im = None
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < h:
            im = attention_heatmap(w_heads[i], tokens, ax, f"head {i}", annotate=annotate)
            if i % cols: ax.set_ylabel("")
            if i // cols != rows - 1: ax.set_xlabel("")
        else:
            ax.axis("off")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cb.set_label("attention weight")
    fig.suptitle(suptitle, fontsize=12)
    return fig

def diff_heatmap(w_a, w_b, tokens, ax, title=""):
    """Signed difference: diverging ramp, symmetric limits, neutral at exactly 0."""
    d = (w_a - w_b).detach().cpu().numpy()
    lim = max(float(np.abs(d).max()), 1e-12)
    im = ax.imshow(d, cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens)
    ax.set_xlabel("key"); ax.set_ylabel("query")
    ax.set_title(f"{title}\nmax |Δ| = {lim:.2e}", fontsize=10)
    return im

def row_sums(w, ax):
    s = w.sum(-1).detach().cpu().numpy().reshape(-1)
    ax.axhline(1.0, color=INK_MUTED, lw=1, ls="--", zorder=1)
    ax.plot(s, marker="o", ms=4, lw=2, color=SERIES[0], zorder=2)
    ax.set_ylim(min(0.0, s.min() - 0.05), max(1.1, s.max() + 0.05))
    ax.set_xlabel("row (head × query)"); ax.set_ylabel("Σ weights")
    ax.set_title(f"row sums — must all be 1.0 (max deviation {abs(s-1).max():.2e})", fontsize=10)
    ax.grid(axis="y", lw=0.6)

def entropy_profile(w_heads, ax, causal=True):
    """Normalized entropy per query position, one thin line per head + mean."""
    e = metrics.normalized_row_entropy(w_heads, causal).detach().cpu().numpy()
    pos = np.arange(1, e.shape[1])          # skip query 0: it has ONE key, 0/log(1) is undefined
    for i in range(e.shape[0]):
        ax.plot(pos, e[i, 1:], lw=1, color=SERIES[0], alpha=0.35)
    ax.plot(pos, e[:, 1:].mean(0), lw=2, color=SERIES[0], label="mean over heads")
    ax.axhline(1.0, color=INK_MUTED, lw=1, ls="--")
    ax.text(0.02, 1.02, "uniform over available keys", transform=ax.get_yaxis_transform(),
            fontsize=8, color=INK_MUTED)
    ax.set_ylim(0, 1.15); ax.set_xlabel("query position"); ax.set_ylabel("normalized entropy")
    ax.set_title("focus per position (1.0 = uniform, 0 = one key)", fontsize=10)
    ax.grid(axis="y", lw=0.6)

def position_profile(w_heads, ax, causal=True):
    """Attention RECEIVED per key position, exposure-corrected, vs uniform baseline."""
    r = metrics.attention_received(w_heads, causal).mean(0).detach().cpu().numpy()
    base = metrics.uniform_baseline_received(w_heads.shape[-1]).detach().cpu().numpy()
    ax.plot(base, lw=2, ls="--", color=INK_MUTED, label="uniform baseline")
    ax.plot(r, lw=2, marker="o", ms=4, color=SERIES[0], label="measured")
    ax.set_xlabel("key position"); ax.set_ylabel("weight received / #queries able to see it")
    ax.set_title("who gets looked at (exposure-corrected)", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", lw=0.6)

def head_metric_bars(w_heads, ax, causal=True):
    s = metrics.summary(w_heads, causal)
    names = ["norm_entropy", "prev_token", "self", "sink"]
    vals = np.stack([s[n].detach().cpu().numpy() for n in names])        # (4, h)
    h = vals.shape[1]; y = np.arange(h); height = 0.2
    for k, n in enumerate(names):
        ax.barh(y + (k - 1.5) * height, vals[k], height=height * 0.9,
                color=SERIES[k], label=n)
        for i in range(h):                                               # direct labels
            ax.text(vals[k][i] + 0.01, y[i] + (k - 1.5) * height, f"{vals[k][i]:.2f}",
                    va="center", fontsize=6, color=INK_MUTED)
    ax.set_yticks(y); ax.set_yticklabels([f"head {i}" for i in range(h)])
    ax.set_xlim(0, 1.2); ax.set_xlabel("score")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))            # legend above, never over the data
    ax.grid(axis="x", lw=0.6)