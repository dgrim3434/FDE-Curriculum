from pathlib import Path
 
import torch
 
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "input.txt"

def load_shakespeare(path=None, split=0.9, device=None):
    """Read the corpus, build the character vocabulary, return the two splits."""
    path = Path(path) if path is not None else DEFAULT_CORPUS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python data/get_corpus.py"
        )
 
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
 
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    n = int(split * len(data))
    return data[:n], data[n:], stoi, itos
 
 
def get_batch(data, block_size, batch_size, generator=None):

    if len(data) < block_size + 2:
        raise ValueError(f"data has {len(data)} tokens, need at least {block_size + 2}")
 
    # -1 reserves room for the +1 shift, so i + block_size never runs past the end
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    return x, y
 
 
def encode(s, stoi):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)
 
 
def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)