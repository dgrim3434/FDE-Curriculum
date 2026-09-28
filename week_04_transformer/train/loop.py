import json
import math
import time
from pathlib import Path
 
import torch
 
from model.gpt import GPT, GPTConfig
from train.data import load_shakespeare, get_batch, decode
 
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

def lr_at(step, lr=1e-3, min_lr=1e-4, warmup=100, total=3000):
    
    if step < 0:
        raise ValueError("ERROR: current step must be greater than or equal to 0")
    if total <= warmup:
        raise ValueError("ERROR: total steps must be greater than warmup")
    if step < warmup:
        return lr * (step + 1) / warmup
    
    p = min(1.0, (step - warmup) / (total - warmup))
    
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))

def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, .95)):
    
    decay, no_decay = [],[]
    
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    
    groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(groups, lr=lr, betas=betas)

@torch.no_grad
def evaluate(model, splits, block_size, batch_size, n_batches=20, seed = 99):
    
    model_training = model.training
    
    model.eval()
    out = {}
    for name, data in splits.items():
        
        gen = torch.Generator().manual_seed(seed)
        losses = []
        
        for _ in range(n_batches):
            
            x, y = get_batch(data, block_size, batch_size, gen)
            
            loss = model(x,y)['loss'].item()
            
            losses.append(loss)
        
        out[name] = sum(losses) / len(losses)
    
    if model_training:
        model.train()
        
    return out
    

def train(
    n_layer=4,
    n_head=4,
    d_model=128,
    block_size=128,
    dropout=0.1,
    norm="pre",
    tie_weights=True,
    batch_size=32,
    steps=3000,
    lr=1e-3,
    lr_min=1e-4,
    warmup=100,
    weight_decay=0.1,
    grad_clip=1.0,
    eval_every=250,
    eval_batches=20,
    sample_at=(0, 500, 1500, 3000),
    seed=1337,
    corpus=None,
    tag="run",
):
    ARTIFACTS.mkdir(exist_ok=True)
    torch.manual_seed(seed)
 
    train_data, val_data, stoi, itos = load_shakespeare(corpus)
    vocab_size = len(stoi)
    splits = {"train": train_data, "val": val_data}
 
    cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                    n_head=n_head, d_model=d_model, dropout=dropout, norm=norm,
                    tie_weights=tie_weights)
    model = GPT(cfg)
    opt = configure_optimizer(model, lr=lr, weight_decay=weight_decay)
 
    print(f"tokens   : {len(train_data):,} train / {len(val_data):,} val, vocab {vocab_size}")
    print(f"params   : {model.n_params():,} ({model.n_params(non_embedding=True):,} non-embedding)")
    print(f"floors   : ln(V) = {math.log(vocab_size):.4f}   "
          f"(unigram/bigram/trigram: run `python data/get_corpus.py --stats`)")
 
    history, evals, samples = [], [], {}
    best_val = float("inf")
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
 
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            e = evaluate(model, splits, block_size, batch_size, n_batches=eval_batches)
            evals.append({"step": step, "train": e["train"], "val": e["val"]})
            print(f"step {step:5d}  train {e['train']:.4f}  val {e['val']:.4f}  "
                  f"lr {lr_at(step, lr, lr_min, warmup, steps):.2e}  {time.time()-t0:.0f}s",
                  flush=True)
            if e["val"] < best_val:                       # early stopping = keep the best
                best_val = e["val"]
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "step": step, "val": best_val}, ARTIFACTS / f"{tag}_best.pt")
 
        if step in sample_at:
            samples[step] = sample(model, stoi, itos, n_tokens=300, temperature=0.8)
 
        if step == steps:
            break

        # Getting the current learning rate
        curr_lr = lr_at(step, lr=lr, min=lr_min, warmup=warmup, total_steps=steps)
        
        # setting the learning rate within the optimizer
        for p in opt.param_groups:
            
            p['lr'] = curr_lr
        
        # Getting the training data
        x, y = get_batch(train_data, block_size=block_size, batch_size=batch_size, generator=g)
        
        # Running Model
        loss = model(x, targets = y)['loss']
        
        # Zeroing the gradient to prevent accumulation
        opt.zero_grad(set_to_none=True)
        
        # Running the backpropogation
        loss.backward()
        
        # Clipping the gradients
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters, grad_clip)
        
        opt.step()
        
        history.append({'step': step, 'loss': loss.item(), 'lr': curr_lr, 'gnorm': gnorm})
        
    # ---- artifacts -------------------------------------------------------- #
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ARTIFACTS / f"{tag}_final.pt")
    (ARTIFACTS / f"{tag}_history.json").write_text(json.dumps(
        {"history": history, "evals": evals, "config": cfg.__dict__,
         "samples": samples, "wall_seconds": time.time() - t0}, indent=2))
    (ARTIFACTS / f"{tag}_samples.txt").write_text(
        "\n\n".join(f"----- step {k} -----\n{v}" for k, v in sorted(samples.items())))
    save_attention(model, stoi, itos, ARTIFACTS / f"{tag}_attn_final.pt")
 
    gn = [h["gnorm"] for h in history]
    print(f"\nwall clock {time.time()-t0:.0f}s")
    print(f"mean loss, first 10 steps {sum(h['loss'] for h in history[:10])/10:.4f}"
          f"  last 100 {sum(h['loss'] for h in history[-100:])/100:.4f}")
    print(f"grad norm: step0 {gn[0]:.2f}  step100 {gn[min(100, len(gn)-1)]:.2f}  "
          f"final {gn[-1]:.2f}  max {max(gn):.2f}")
    print(f"best val {best_val:.4f}")
    print(f"artifacts -> {ARTIFACTS}")
    return model, history, evals

@torch.no_grad()
def sample(model, stoi, itos, n_tokens=300, temperature=0.8, prompt="\n", top_k=None):
    was_training = model.training
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = model.generate(idx, n_tokens, temperature=temperature, top_k=top_k)
    if was_training:
        model.train()
    return decode(out[0].tolist(), itos)
 
 
@torch.no_grad()
def save_attention(model, stoi, itos, path,
                   text="First Citizen:\nBefore we proceed any further, hear me speak."):

    was_training = model.training
    model.eval()
    idx = torch.tensor([[stoi[c] for c in text if c in stoi]], dtype=torch.long)
    idx = idx[:, : model.cfg.block_size]
    out = model(idx, return_atten=True)
    attn = torch.stack([w[0] for w in out["attn"]])          # (L, H, T, T)
    torch.save({"attn": attn, "text": text, "ids": idx[0]}, path)
    if was_training:
        model.train()
    return attn
 
 
if __name__ == "__main__":
    import argparse
 
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm", choices=["pre", "post"], default="pre")
    p.add_argument("--no_tie", action="store_true")
    p.add_argument("--tag", default="run")
    a = p.parse_args()
 
    train(n_layer=a.n_layer, n_head=a.n_head, d_model=a.d_model,
          block_size=a.block_size, dropout=a.dropout, norm=a.norm,
          tie_weights=not a.no_tie, batch_size=a.batch_size, steps=a.steps,
          lr=a.lr, warmup=a.warmup, tag=a.tag,
          sample_at=(0, a.steps // 6, a.steps // 2, a.steps))