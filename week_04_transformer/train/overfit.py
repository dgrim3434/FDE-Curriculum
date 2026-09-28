import argparse
import collections
import math

import torch

from model.gpt import GPT, GPTConfig
from train.data import load_shakespeare


def overfit(n_tokens=50, steps=400, lr=1e-3, n_layer=4, n_head=4, d_model=128,
            norm="pre", tie_weights=True, seed=0, use_corpus=True, verbose=True):
    torch.manual_seed(seed)

    if use_corpus:
        train_data, _, stoi, _ = load_shakespeare()
        vocab_size = len(stoi)
        ids = train_data[: n_tokens + 1].unsqueeze(0)
    else:
        vocab_size = 65
        ids = torch.randint(0, vocab_size, (1, n_tokens + 1))

    x, y = ids[:, :-1], ids[:, 1:]

    cfg = GPTConfig(vocab_size=vocab_size, block_size=max(64, n_tokens + 1),
                    n_layer=n_layer, n_head=n_head, d_model=d_model,
                    dropout=0.0, norm=norm, tie_weights=tie_weights)   # dropout OFF
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    counts = collections.Counter(y.flatten().tolist())
    n = y.numel()
    unigram = -sum(c / n * math.log(c / n) for c in counts.values())

    history = []
    for step in range(steps):
        loss = model(x, targets=y)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(loss.item())
        if verbose and (step % 50 == 0 or step == steps - 1):
            print(f"  step {step:4d}  loss {loss.item():.6f}")

    final = history[-1]
    if verbose:
        print(f"\n  init loss      {history[0]:.4f}   (ln({vocab_size}) = {math.log(vocab_size):.4f})")
        print(f"  final loss     {final:.6f}   (gate: < 0.05)")
        print(f"  unigram floor  {unigram:.4f}   <- a plateau HERE means a wiring bug")
        if final < 0.05:
            print("\n  PASS - the model can learn. Proceed to the leak test and the data tests.")
        elif abs(final - unigram) < 0.1:
            print("\n  FAIL - pinned at the unigram floor. No information is reaching the head.")
            print("         Check: residual adds (x = x + f(x)), ln_f, the output head.")
            print("         Run the zero-branch identity test on one block.")
        elif abs(final - math.log(vocab_size)) < 0.1:
            print("\n  FAIL - stuck at ln(V). Parameters are not being updated.")
            print("         Check: opt.zero_grad/step, params in the optimizer, lr > 0.")
        else:
            print(f"\n  FAIL - plateaued at {final:.4f}, which matches no known floor.")
    return final, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_tokens", type=int, default=50)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--norm", choices=["pre", "post"], default="pre")
    p.add_argument("--no_tie", action="store_true")
    p.add_argument("--random", action="store_true", help="random tokens instead of the corpus")
    a = p.parse_args()

    final, _ = overfit(n_tokens=a.n_tokens, steps=a.steps, lr=a.lr, n_layer=a.n_layer,
                       n_head=a.n_head, d_model=a.d_model, norm=a.norm,
                       tie_weights=not a.no_tie, use_corpus=not a.random)
    raise SystemExit(0 if final < 0.05 else 1)