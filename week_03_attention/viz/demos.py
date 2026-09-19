"""Produces every figure in artifacts/. Run: python -m viz.demos"""
import math, numpy as np, torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from attention.multi_head import Multi_Head_Attention
from attention.masks import causal_mask
from attention.functional import scaled_dot_product_attention

from viz import plot, metrics
from viz.style import save, SEQ, SERIES, INK_MUTED

TOKENS = "the cat sat on the mat because it was tired".split()
T, D, H = len(TOKENS), 64, 4

def sinusoidal_pe(L, d):
    pe = np.zeros((L, d), np.float32); pos = np.arange(L)[:, None]
    div = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(pos * div); pe[:, 1::2] = np.cos(pos * div)
    return torch.tensor(pe)

def shift_matrix(d, k):
    R = torch.zeros(d, d)
    om = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    for i, w in enumerate(om):
        c, s = math.cos(w * k), math.sin(w * k)
        R[2*i, 2*i], R[2*i, 2*i+1] = c, s
        R[2*i+1, 2*i], R[2*i+1, 2*i+1] = -s, c
    return R

def build():
    torch.manual_seed(0)
    mha = Multi_Head_Attention(D, H).eval()
    pe = sinusoidal_pe(T, D)
    x = (torch.randn(1, T, D) * 0.5 + pe).float()
    cm = causal_mask(T)
    _, w = mha(x, mask=cm)
    return mha, x, cm, w[0], pe

# 1 --------------------------------------------------------------------------
def fig_head_grid(w):
    fig = plot.head_grid(w, TOKENS,
        "Untrained multi-head attention, causal mask — random weights, no linguistic meaning",
        annotate=False)
    save(fig, "01_heads_causal.png")

# 2 --------------------------------------------------------------------------
def fig_prev_token_head(w, pe):
    beta = 16.0
    Q = (pe @ shift_matrix(D, -1).T) * beta
    _, w_prev = scaled_dot_product_attention(Q[None], pe[None], pe[None], causal_mask(T), single_head=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plot.attention_heatmap(w_prev[0], TOKENS, axes[0],
                           f"hand-built previous-token head (beta={beta:g})", annotate=True)
    im = plot.attention_heatmap(w[0], TOKENS, axes[1], "random head 0 — same mask")
    axes[1].set_ylabel("")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8).set_label("attention weight")
    fig.suptitle("A pattern you designed vs a pattern that means nothing", fontsize=12)
    save(fig, "02_prev_token_head.png")

# 3 --------------------------------------------------------------------------
def fig_diagnostics(mha, x, cm, w):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot.row_sums(w, axes[0][0])
    Q, K, V = (mha._split(p(x)) for p in (mha.W_q, mha.W_k, mha.W_v))
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    mine, _ = mha(x, mask=cm)
    im = plot.diff_heatmap(w[0], torch.softmax(
        (Q[0,0] @ K[0,0].T / math.sqrt(mha.d_k)).masked_fill(~cm, float("-inf")), -1),
        TOKENS, axes[0][1], "head 0: mine − torch.softmax reference")
    fig.colorbar(im, ax=axes[0][1], shrink=0.8)
    plot.entropy_profile(w, axes[1][0])
    plot.position_profile(w, axes[1][1])
    fig.suptitle("Diagnostics — read these before you interpret any heatmap", fontsize=12)
    fig.tight_layout()
    save(fig, "03_diagnostics.png")
    print("  max |mine-ref| output:", (mha.W_o(mha._merge(ref)) - mine).abs().max().item())

# 4 --------------------------------------------------------------------------
def fig_head_metrics(w):
    fig, ax = plt.subplots(figsize=(8, 4.0))
    plot.head_metric_bars(w, ax)
    ax.set_title("Per-head signature — scan this before opening a heatmap",
                 fontsize=11, pad=26)
    save(fig, "04_head_metrics.png")

if __name__ == "__main__":
    mha, x, cm, w, pe = build()
    fig_head_grid(w); fig_prev_token_head(w, pe); fig_diagnostics(mha, x, cm, w); fig_head_metrics(w)
    s = metrics.summary(w)
    for k, v in s.items():
        print(f"{k:>13}: " + "  ".join(f"h{i}={x:.3f}" for i, x in enumerate(v.tolist())))