
import argparse
import json
import math
from pathlib import Path

import torch

from model.gpt import GPT, GPTConfig
from train.data import load_shakespeare, get_batch
from train.loop import configure_optimizer, lr_at

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def strip_residuals(model):
    """Replace each block's forward with the non-residual version: x = ffn(ln2(attn(ln1(x))))."""
    for blk in model.blocks:
        def fwd(x, mask=None, return_attn=False, blk=blk, **kw):
            a, _ = blk.attn(blk.ln1(x), mask=mask)
            return blk.ffn(blk.ln2(a)), None
        blk.forward = fwd


def run(tag, n_layer=12, norm="pre", warmup=0, residual=True, tie_weights=True,
        steps=400, lr=1e-3, block_size=64, batch_size=32, d_model=128, n_head=4,
        seed=0):
    torch.manual_seed(seed)
    train_data, val_data, stoi, _ = load_shakespeare()
    cfg = GPTConfig(vocab_size=len(stoi), block_size=block_size, n_layer=n_layer,
                    n_head=n_head, d_model=d_model, dropout=0.0, norm=norm,
                    tie_weights=tie_weights)
    model = GPT(cfg)
    if not residual:
        strip_residuals(model)

    opt = configure_optimizer(model, lr=lr)
    g = torch.Generator().manual_seed(1234)
    hist, gnorms = [], []

    for step in range(steps):
        cur = lr * (step + 1) / warmup if (warmup and step < warmup) else lr
        for pg in opt.param_groups:
            pg["lr"] = cur
        x, y = get_batch(train_data, block_size, batch_size, generator=g)
        loss = model(x, targets=y)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
        opt.step()
        hist.append(loss.item())

    def at(i):
        return hist[i] if i < len(hist) else float("nan")

    print(f"{tag:<34} init {hist[0]:.3f} | s50 {at(49):.3f} | s100 {at(99):.3f} | "
          f"s200 {at(199):.3f} | s{steps} {hist[-1]:.3f} | min {min(hist):.3f} | "
          f"gnorm s0 {gnorms[0]:.1f}")
    return {"tag": tag, "hist": hist, "gnorms": gnorms, "params": model.n_params()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--all", action="store_true")
    a = p.parse_args()

    print("ablations: d=128, h=4, T=64, B=32, AdamW lr=1e-3")
    print("reference floors: uniform 4.1744 | unigram 3.3474 | bigram 2.4819 | trigram 2.0684\n")

    results = [
        run("pre-norm,  no warmup", n_layer=a.depth, norm="pre", warmup=0, steps=a.steps),
        run("post-norm, no warmup", n_layer=a.depth, norm="post", warmup=0, steps=a.steps),
        run("post-norm, warmup=200", n_layer=a.depth, norm="post", warmup=200, steps=a.steps),
        run("NO residual connections", n_layer=a.depth, residual=False, steps=a.steps),
    ]

    if a.all:
        print()
        for depth in (1, 2, 4):
            results.append(run(f"pre-norm, depth {depth}", n_layer=depth, steps=a.steps))
            results.append(run(f"no residual, depth {depth}", n_layer=depth,
                               residual=False, steps=a.steps))
        print()
        results.append(run("tied embeddings", n_layer=a.depth, tie_weights=True, steps=a.steps))
        results.append(run("untied embeddings", n_layer=a.depth, tie_weights=False, steps=a.steps))

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "ablations.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten -> {ARTIFACTS / 'ablations.json'}")
    print("\nRead the plateaus against the floors above. A run pinned at ~3.35 learned")
    print("the character frequencies and nothing context-dependent.")