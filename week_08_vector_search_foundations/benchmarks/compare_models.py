from embed import embed_texts
from search.metrics import evaluate
from data_loading import load_corpus, load_queries, load_qrels
from data.constants import scifact_path
from search.brute_force import BruteForceIndex
import numpy as np
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

GRID = [
    {"tag": "e5", "model": "intfloat/e5-base-v2", "q_prefix": "query: ", "p_prefix": "passage: "},
    {"tag": "e5_noprefix", "model": "intfloat/e5-base-v2", "q_prefix": "", "p_prefix": ""},
    {"tag": "bge", "model": "BAAI/bge-base-en-v1.5", "q_prefix": "Represent this sentence for searching relevant passages: ", "p_prefix": ""},
    {"tag": "bge_noprefix", "model": "BAAI/bge-base-en-v1.5", "q_prefix": "", "p_prefix": ""},
    {"tag": "minilm", "model": "sentence-transformers/all-MiniLM-L6-v2", "q_prefix": "", "p_prefix": ""},
    {"tag": "mpnet", "model": "sentence-transformers/all-mpnet-base-v2", "q_prefix": "", "p_prefix": ""}
]

SEED = 0

def run_experiment():
    
    corpus = load_corpus(scifact_path / "corpus.jsonl")
    test = load_qrels(scifact_path / "qrels/test.tsv")
    queries = load_queries(scifact_path / "queries.jsonl")
    
    test_ids = sorted(test.keys())
    
    query_ids = test_ids
    query_text = [queries[idx] for idx in query_ids]
    corpus_text = [c[1] for c in corpus]
    corpus_ids = np.array([c[0] for c in corpus])
    
    corpus_ids_to_position = {corpus_ids[i]:i for i in range(len(corpus_ids))}
    
    summary_results = []
    ndcgs = {}
    
    rng = np.random.default_rng(seed=SEED)
    rand_run = {qid: rng.choice(corpus_ids, size=100, replace=False).tolist() for qid in query_ids}   
    means, per_query = evaluate(rand_run, test)
    record = {'tag': 'rand_baseline', 'model': None, 'q_prefix': None, 'p_prefix': None, 'dim': None, 'index_MB': None,
              'passages_per_second': None, 'tokens_per_second': None, 'mean_tokens': None,
              'truncation_rate': None}
    record.update(means)
    
    summary_results.append(record)
    
    for g in GRID:
        
        rng = np.random.default_rng(seed=SEED)
        score_distributions = {}
        corp, corp_info = embed_texts(model_name=g['model'], texts=corpus_text, prefix=g['p_prefix'])
        bf = BruteForceIndex(corp)
        
        query, query_info = embed_texts(model_name=g['model'], texts=query_text, prefix=g['q_prefix'])
        
        # returns (300, 100) There are 300 queries and 100 docs
        indicies = bf.search_batch(Q=query, k=100)[0]
        
        doc_indicies = corpus_ids[indicies]
        
        run_indicies = doc_indicies.tolist()
        run = {query_ids[i]: run_indicies[i] for i in range(len(query_text))}
        
        means, per_query = evaluate(run, test)
        
        d = corp_info['dim']
        index_mb = float((len(corp) * d * 4) / 1e6)
        
        record = {'tag': g['tag'], 'model': g['model'], 'q_prefix': g['q_prefix'], 'p_prefix': g['p_prefix'], 'dim': d, 'index_MB': index_mb,
                  'passages_per_second': corp_info['passages_per_sec'], 'tokens_per_second': corp_info['tokens_per_sec'], 'mean_tokens': corp_info['mean_tokens'],
                  'truncation_rate': corp_info['truncation_rate']}
        
        per = {}
        for k, v in per_query.items():
            per[k] = v['ndcg@10']
        
        ndcgs[g['tag']] = per
        
        record.update(means)
        
        summary_results.append(record)

        # Shapes: query: 300, D E: N, D we need 300, N then take then N relevant indicies
        scores = query @ bf.E.T
        
        for i, qid in enumerate(query_ids):
            score_distributions[qid] = {}
            rel_doc_ids = list(test[qid].keys())
            mapped = []
            
            for rel_id in rel_doc_ids:
                mapped.append(corpus_ids_to_position.get(rel_id))
            
            mapped = np.array(mapped)
            q_score = scores[i][mapped]
            
            score_distributions[qid]['relevant'] = q_score.tolist()

            allowed = np.ones_like(scores[i], dtype=bool)
            allowed[mapped] = False
            pool = np.flatnonzero(allowed)
            
            picked = rng.choice(pool, size=100, replace=False)
            
            irr_scores = scores[i][picked]
            
            score_distributions[qid]['irrelevant'] = irr_scores.tolist()
        
        
        (RESULTS / f"score_dist_{g['tag']}.json").write_text(json.dumps(score_distributions, indent=2), encoding='utf-8')
        np.save(f"data/embeddings/scifact_{g['tag']}_corpus.npy", corp)
        np.save(f"data/embeddings/scifact_{g['tag']}_queries.npy", query)
    

    
    model_comparison = {}
    model_comparison['config'] = {'dataset': 'scifact', 'n_docs': len(corpus_text), 'n_queries': len(query_ids), 'search': 'exact brute force', 'k_max': 100, 'batch_size': 256,
                                  'seed': SEED}  
    model_comparison['model_summaries'] = summary_results
    
    (RESULTS / "compare_models.json").write_text(json.dumps(model_comparison, indent=2), encoding='utf-8')
    (RESULTS / "compare_models_per_query.json").write_text(json.dumps(ndcgs), encoding='utf-8')
    
        
if __name__ == "__main__":
    
    run_experiment()