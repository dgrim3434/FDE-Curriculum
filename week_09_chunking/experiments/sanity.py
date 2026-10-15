from data.constants import SCIFACT_PATH
from retrieval.data import load_beir
from retrieval.index import make_embed_fn, embed_with_cache, rank_documents
from retrieval.metrics import evaluate_run, mean_scores
from chunking.base import Chunk
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUT = ROOT / "results/sanity"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 10
WEEK8_NDCG = None
TOLERANCE = 0.005


def week8_reproduction(corpus, queries, qrels, embed):
    
    # Week 8's setting:
    chunks = []
    
    for doc_id in sorted(corpus):
        
        doc = corpus[doc_id]
        text = doc['text']
        
        chunk = Chunk(doc_id, 0, 0, len(text), text, 0, {'embed_text': doc['title'] + " " + text})
        chunks.append(chunk)
    
    C = embed_with_cache(chunks, embed, CACHE / "sanity_week8_minilm.npz")
    
    qids = sorted(qrels)
    Q = embed([queries[q] for q in qids])
    
    run = rank_documents(Q, C, chunks, qids, K)
    
    per_query = evaluate_run(run, qrels, K)
    
    return mean_scores(per_query), chunks

def random_baseline(corpus, qrels, seed=0):
    
    rng = np.random.default_rng(seed)
    doc_ids = sorted(corpus)
    
    run = {}
    
    for qid in sorted(qrels):
        
        order = rng.permutation(len(doc_ids))[:K]
        run[qid] = [doc_ids[i] for i in order]
    
    return mean_scores(evaluate_run(run, qrels, K))


if __name__ == "__main__":
    
    corpus, queries, qrels = load_beir(SCIFACT_PATH)
    print(f"corpus {len(corpus)}, queries {len(queries)}, qrels {len(qrels)}")
    
    model = SentenceTransformer(MODEL, device='cuda')
    embed = make_embed_fn(model, prefix="", batch_size=64)
    
    repro, _ = week8_reproduction(corpus, queries, qrels, embed)
    rand = random_baseline(corpus, qrels)
    
    match = None
    
    if WEEK8_NDCG is not None:
        match = abs(repro['ndcg'] - WEEK8_NDCG) <= TOLERANCE
    
    results = {
        'week8_reproduction': {
            'settings': {'model': MODEL, 'document_text': 'title + " " + text', 'prefix': None,
                         'chunks_per_document': 1, 'k': K},
            'scores': repro,
            'week8_ndcg': WEEK8_NDCG,
            'tolerance': TOLERANCE,
            'match': match,
        },
        'random_baseline': {'seed': 0, 'scores': rand},
    }
    
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity.json").write_text(json.dumps(results, indent=2), encoding='utf-8')
    
    print(json.dumps(results, indent=2))