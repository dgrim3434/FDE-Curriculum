from metadata.store import load_chunks
from pathlib import Path
import numpy as np
import json
import csv
import sys

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
OUT = RESULTS / "floor_audit"

N = 30
PASS_MARK = 24
SEED = 0


def config_ids():
    
    ids = {}
    
    for path in (RESULTS / "configs").glob("*.json"):
        
        c = json.loads(path.read_text(encoding='utf-8'))
        ids[c['config']['name']] = c['config_id']
    
    return ids

def sample(name):
    
    ids = config_ids()
    
    if name not in ids:
        raise ValueError(f"ERROR: no results for configuration {name}")
    
    chunks = load_chunks(CACHE / f"chunks_{ids[name]}.jsonl")
    
    rng = np.random.default_rng(SEED)
    picks = sorted(rng.choice(len(chunks), size=min(N, len(chunks)), replace=False).tolist())
    
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        
        writer = csv.writer(f)
        writer.writerow(['row', 'chunk_id', 'n_tokens', 'text', 'interpretable', 'note'])
        
        for row, i in enumerate(picks, start=1):
            
            c = chunks[i]
            writer.writerow([row, c.chunk_id, c.n_tokens, c.text, '', ''])
    
    print(f"wrote {path}: fill 'interpretable' with Y or N for every row, then run: score {name}")

def score(name):
    
    path = OUT / f"{name}.csv"
    
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    
    marks = [r['interpretable'].strip().upper() for r in rows]
    
    if any(m not in ('Y', 'N') for m in marks):
        missing = [r['row'] for r, m in zip(rows, marks) if m not in ('Y', 'N')]
        raise ValueError(f"ERROR: rows without a Y/N mark: {missing}")
    
    yes = marks.count('Y')
    
    result = {
        'config': name,
        'n': len(marks),
        'interpretable': yes,
        'rate': yes / len(marks),
        'pass_mark': PASS_MARK,
        'passes': yes >= PASS_MARK,
        'seed': SEED,
        'failures': [{'chunk_id': r['chunk_id'], 'note': r['note']} for r, m in zip(rows, marks) if m == 'N'],
    }
    
    (OUT / f"{name}_score.json").write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    
    mode, name = sys.argv[1], sys.argv[2]
    
    if mode == 'sample':
        sample(name)
    elif mode == 'score':
        score(name)
    else:
        raise ValueError("usage: floor_audit sample|score <config name>")