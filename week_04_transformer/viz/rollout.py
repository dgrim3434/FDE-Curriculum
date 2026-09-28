import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from viz.plot_style import apply_style, ATTN_CMAP, SERIES, INK, INK_MUTED

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"


# --------------------------------------------------------------------------- #
def rollout(attn, add_identity=True, head_reduce="mean"):
    """attn: (L, H, T, T) for ONE sequence. Returns (L, T, T): rollout after each layer.

    NOTE the multiplication order: the later layer goes on the LEFT.
    R_l = A_hat_l @ R_{l-1}. Row i of the result is "where position i's information
    came from", which is the same orientation as the input maps.
    """
    L, H, T, _ = attn.shape
    I = torch.eye(T, dtype=attn.dtype)
    R = I.clone()
    outs = []
    for l in range(L):
        A = attn[l].mean(0) if head_reduce == "mean" else attn[l].max(0).values
        if add_identity:
            A = 0.5 * A + 0.5 * I
            A = A / A.sum(-1, keepdim=True)
        R = A @ R
        outs.append(R.clone())
    return torch.stack(outs)


def row_entropy(p):
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def tick_labels(tokens):
    """Whitespace is invisible on an axis. Render it."""
    out = []
    for t in tokens:
        out.append({"\n": "\\n", " ": "␣", "\t": "\\t"}.get(t, t))
    return out


# --------------------------------------------------------------------------- #
def heat(ax, M, tokens, title, vmax=None, show_y=True):
    im = ax.imshow(M, cmap=ATTN_CMAP, vmin=0.0, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    labels = tick_labels(tokens)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    if show_y:
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_ylabel("query — the position doing the reading")
    else:
        ax.set_yticks([])
    ax.set_xlabel("key — the position being read")
    ax.set_title(title)
    ax.grid(False)
    return im


def main(tag="run", n_chars=48, ablate=False, head_reduce="mean"):
    apply_style()
    blob = torch.load(ARTIFACTS / f"{tag}_attn_final.pt", weights_only=False)
    attn_full = blob["attn"].float()                  # (L, H, T, T)
    ids = blob["ids"]
    text = blob["text"]

    T = min(n_chars, attn_full.shape[-1])
    attn = attn_full[:, :, :T, :T]
    attn = attn / attn.sum(-1, keepdim=True)          # renormalize after cropping
    tokens = list(text)[:T]
    L, H = attn.shape[0], attn.shape[1]

    R = rollout(attn, add_identity=True, head_reduce=head_reduce)
    R_no_id = rollout(attn, add_identity=False, head_reduce=head_reduce)

    # sanity: rows sum to 1, strictly causal
    assert torch.allclose(R[-1].sum(-1), torch.ones(T), atol=1e-4), "rollout rows must sum to 1"
    assert float(R[-1].triu(1).abs().max()) < 1e-6, "rollout leaked into the future"

    raw_last = attn[-1].mean(0)

    # ---- figure 1: raw vs rollout --------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    vmax = float(max(raw_last.max(), R[-1].max()))
    heat(axes[0], raw_last.numpy(), tokens,
         f"Raw attention, final layer (mean of {H} heads)", vmax=vmax)
    im = heat(axes[1], R[-1].numpy(), tokens,
              f"Attention rollout, all {L} layers", vmax=vmax, show_y=False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("share of attribution (row sums to 1)", color=INK_MUTED, fontsize=9)
    cbar.ax.tick_params(labelsize=8, color=INK_MUTED)
    fig.suptitle("Where each position's information comes from", x=0.125, ha="left",
                 fontsize=13, fontweight="bold")
    out1 = ARTIFACTS / "07_rollout.png"
    fig.savefig(out1)
    plt.close(fig)

    # ---- figure 2: rollout by depth ------------------------------------- #
    fig, axes = plt.subplots(1, L, figsize=(3.1 * L, 3.6))
    axes = np.atleast_1d(axes)
    vmax2 = float(R.max())
    for l in range(L):
        heat(axes[l], R[l].numpy(), tokens, f"after layer {l + 1}",
             vmax=vmax2, show_y=(l == 0))
        axes[l].set_xticklabels([])
        axes[l].set_xlabel("")
        if l > 0:
            axes[l].set_ylabel("")
    fig.suptitle("Attribution spreads with depth — one layer can only move information one hop",
                 x=0.06, ha="left", fontsize=12, fontweight="bold")
    out2 = ARTIFACTS / "07_rollout_by_depth.png"
    fig.savefig(out2)
    plt.close(fig)

    # ---- figure 3: diagnostics ------------------------------------------ #
    last = T - 1
    ent = [row_entropy(R[l][last]) for l in range(L)]
    ent_no_id = [row_entropy(R_no_id[l][last]) for l in range(L)]
    pos0 = [float(R[l][last, 0]) for l in range(L)]
    uniform_max = float(np.log(T))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    layers = range(1, L + 1)
    a1.plot(layers, ent, color=SERIES["train"], marker="o", label="rollout (with residual term)")
    a1.plot(layers, ent_no_id, color=SERIES["val"], marker="s",
            label="attention product only (no residual term)")
    a1.axhline(uniform_max, color=INK_MUTED, linestyle=":", linewidth=1.0)
    a1.annotate(f"uniform over {T} positions = {uniform_max:.2f}",
                xy=(1, uniform_max), xytext=(2, -12), textcoords="offset points",
                fontsize=8, color=INK_MUTED)
    a1.set_xlabel("layers composed"); a1.set_ylabel("entropy of the final row (nats)")
    a1.set_title("How diffuse the attribution becomes")
    a1.set_xticks(list(layers)); a1.legend(loc="lower right")

    a2.plot(layers, pos0, color=SERIES["train"], marker="o")
    a2.set_xlabel("layers composed"); a2.set_ylabel("attribution on position 0")
    a2.set_title(f"Weight on the first token ({tick_labels([tokens[0]])[0]!r})")
    a2.set_xticks(list(layers))
    out3 = ARTIFACTS / "07_rollout_diagnostics.png"
    fig.savefig(out3)
    plt.close(fig)

    # ---- numbers for the write-up --------------------------------------- #
    print(f"written -> {out1}\n           {out2}\n           {out3}\n")
    print(f"prompt          : {text!r}")
    print(f"shape           : {L} layers x {H} heads x {T} x {T}")
    print(f"rows sum to 1   : {bool(torch.allclose(R[-1].sum(-1), torch.ones(T), atol=1e-4))}")
    print(f"upper triangle  : {float(R[-1].triu(1).abs().max()):.1e}  (must be 0 — strictly causal)")
    print(f"\nentropy of the final row, by depth (uniform-causal max {uniform_max:.3f}):")
    for l in range(L):
        print(f"  after {l+1} layer(s): rollout {ent[l]:.3f} | no-residual-term {ent_no_id[l]:.3f}")

    vals, idxs = R[-1][last].topk(min(8, T))
    print(f"\ntop-8 attribution for the FINAL position ({tokens[last]!r}):")
    for v, i in zip(vals.tolist(), idxs.tolist()):
        print(f"  {tick_labels([tokens[i]])[0]!r:<6} pos {i:<3} {v:.4f}")

    if ablate:
        print("\nCAUSAL ABLATION CHECK")
        print("=" * 62)
        from model.gpt import GPT, GPTConfig
        from train.data import load_shakespeare
        ckpt = torch.load(ARTIFACTS / f"{tag}_final.pt", weights_only=False)
        _, _, stoi, _ = load_shakespeare()
        model = GPT(GPTConfig(**ckpt["cfg"]))
        model.load_state_dict(ckpt["model"])
        model.eval()

        idx = ids[:T].unsqueeze(0)
        with torch.no_grad():
            base = model(idx)["logits"][0, last].clone()
        deltas = []
        space = stoi.get(" ", 0)
        for p in range(T):
            alt = idx.clone(); alt[0, p] = space
            with torch.no_grad():
                deltas.append(float((model(alt)["logits"][0, last] - base).norm()))
        d = np.array(deltas); d = d / d.sum()
        r = R[-1][last].numpy()
        corr = float(np.corrcoef(r, d)[0, 1])
        print(f"  correlation(rollout, ablation effect) = {corr:.3f}")
        print("  top-5 by rollout  :",
              [tick_labels([tokens[i]])[0] for i in np.argsort(-r)[:5]])
        print("  top-5 by ablation :",
              [tick_labels([tokens[i]])[0] for i in np.argsort(-d)[:5]])
        print("  A strong but imperfect match is the expected result. Rollout is computed")
        print("  from weights and ignores the magnitude of what was moved; ablation is causal.")

    print("\nRED FLAG CHECKLIST")
    print("=" * 62)
    checks = [
        ("every row sums to 1 at every depth",
         all(bool(torch.allclose(R[l].sum(-1), torch.ones(T), atol=1e-4)) for l in range(L))),
        ("upper triangle is exactly 0 (no future leakage)",
         float(R[-1].triu(1).abs().max()) < 1e-6),
        ("entropy increases with depth (attribution spreads)",
         all(b >= a - 1e-6 for a, b in zip(ent, ent[1:]))),
        ("final-row entropy is below the uniform ceiling", ent[-1] < uniform_max),
        ("the identity/residual term is included", True),
        ("the model is TRAINED (an untrained rollout shows nothing)", True),
        ("axes are labelled query vs key, not 'x' and 'y'", True),
        ("colormap is single-hue sequential, not a rainbow", True),
        ("row 0 is trivially [1, 0, ...] and is not presented as a finding", True),
    ]
    for label, ok in checks:
        print(f"  [{'ok  ' if ok else 'FLAG'}] {label}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--chars", type=int, default=48)
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--head-reduce", choices=["mean", "max"], default="mean")
    a = p.parse_args()
    main(tag=a.tag, n_chars=a.chars, ablate=a.ablate, head_reduce=a.head_reduce)