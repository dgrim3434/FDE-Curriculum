import numpy as np
import math

def index_recall(approx_ids, exact_ids, k):
    
    if len(approx_ids) != len(exact_ids):
        
        raise ValueError ("ERROR: approx_ids and exact_ids must be the same length")
    
    results = []
    for i in range(len(approx_ids)):
        
        top_k_exact = set(exact_ids[i][:k])
        top_k_approx = set(approx_ids[i][:k])
        
        common = top_k_exact & top_k_approx
        
        results.append(float(len(common) / k))
    
    return np.mean(results), np.array(results)

def recall_at_k(ranked, relevant, k):
    
    rank = set(ranked[:k])
    rel = set([k for k, v in relevant.items() if v > 0])
    
    common = rank & rel
    
    return float(len(common) / len(rel))

def precision_at_k(ranked, relevant, k):
    
    rank = set(ranked[:k])
    rel = set([k for k, v in relevant.items() if v > 0])
    
    common = rank & rel
    
    return float(len(common) / len(rank))

def mrr_at_k(ranked, relevant, k):
    
    rev = set([k for k,v in relevant.items() if v > 0])
    
    for i in range(k):
        
        if ranked[i] in rev:
            return float(1 / (i + 1))
    
    return 0.0

def ndcg_at_k(ranked, relevant, k):

    best_order = [(key, v) for key, v in relevant.items()]
    best_order.sort(key=lambda x: -x[1])
    
    best = 0
    dcg = 0
    
    for i in range(k):
        
        if i < len(best_order):
            best += best_order[i][1] / math.log2(i + 2)
        
        if i < len(ranked):
            dcg += relevant.get(ranked[i], 0) / math.log2(i + 2)
    
    return dcg / best if best > 0 else 0
    
def evaluate(run, qrels, k_values = [1,10, 100]):
    
    results = {}
    means = {}
    
    tracker = {}
    for k in k_values:
        tracker[f'recall@{k}'] = []
        tracker[f'precision@{k}'] = []
        tracker[f'ndcg@{k}'] = []
        tracker[f'mrr@{k}'] = []
        
    for qid, gold in qrels.items():
        
        if qid not in run:
            raise ValueError(f"ERROR: qid: {qid} is in the golden set but couldn't be found in the run")
        
        ranked = run[qid]
        
        result = {}
        for k in k_values:
            recall = recall_at_k(ranked, gold, k)
            result[f'recall@{k}'] = recall
            tracker[f'recall@{k}'].append(recall)
            
            precision = precision_at_k(ranked, gold, k)
            result[f'precision@{k}'] = precision
            tracker[f'precision@{k}'].append(precision)
            
            ndcg = ndcg_at_k(ranked, gold, k)
            result[f'ndcg@{k}'] = ndcg
            tracker[f'ndcg@{k}'].append(ndcg)
            
            mrr = mrr_at_k(ranked, gold, k)
            result[f'mrr@{k}'] = mrr
            tracker[f'mrr@{k}'].append(mrr)
            
        results[qid] = result
    
    for k, v in tracker.items():
        means[k] = float(np.mean(v))
    
    return means, results
        
    