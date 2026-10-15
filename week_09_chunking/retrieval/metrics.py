import math

def recall_at_k(ranked, qrels_q, k):
    
    if qrels_q is None or len(qrels_q) == 0:
        raise ValueError("ERROR: Query Has no relevant documents")
    
    preds = set(ranked[:k])
    gold = set([k for k,v in qrels_q.items() if v > 0])

    if len(gold) == 0:
        raise ValueError("ERROR: Query Has no relevant documents")
    
    common = preds & gold
    
    return len(common) / len(gold)

def precision_at_k(ranked, qrels_q, k):
    
    preds = set(ranked[:k])
    gold = set([k for k,v in qrels_q.items() if v > 0])
    
    common = preds & gold
    
    return len(common) / k

def reciprocal_rank(ranked, qrels_q, k):
    
    gold = set([k for k, v in qrels_q.items() if v > 0])
    
    for i in range(k):
        
        if i >= len(ranked):
            return 0
        if ranked[i] in gold:
            
            return 1 / (i + 1)
    
    return 0

def ndcg_at_k(ranked, qrels_q, k, gain="linear"):

    gains = set(['linear', 'exponential'])
    
    if gain not in gains:
        raise ValueError("ERROR: Invalid gain")
    if qrels_q is None or len(qrels_q) == 0:
        raise ValueError("ERROR: Query Has no relevant documents")
    
    scores = [s for _, s in qrels_q.items() if s > 0]
    
    if len(scores) == 0:
        raise ValueError("ERROR: Query Has no relevant documents")
    
    scores.sort(key= lambda x: -x)
    
    curr = 0
    best = 0
    for i in range(k):
        
        if gain == "linear":
            
            if i < len(ranked):
                curr += qrels_q.get(ranked[i], 0) / math.log2(i + 2)
            if i < len(scores):
                best += scores[i] / math.log2(i + 2)
        if gain == "exponential":
            
            if i < len(ranked):
                
                if ranked[i] in qrels_q:
                    
                    curr += (2 ** qrels_q[ranked[i]] - 1) / math.log(i + 2)
            if i < len(scores):
                
                best += (2 ** scores[i] - 1) / math.log(i + 2)
    
    return float(curr / best)


def evaluate_run(run, qrels, k, gain="linear") -> dict:

    results = {}
    
    for id, gold in qrels.items():
        
        if id not in run:
            results[id] = {'recall': 0.0, 'precision': 0.0, 'rr': 0.0, 'ndcg': 0.0}
            continue
        
        preds = run[id]
        
        prec = precision_at_k(preds, gold, k)
        recall = recall_at_k(preds, gold, k)
        rr = reciprocal_rank(preds, gold, k)
        ndcg = ndcg_at_k(preds, gold, k, gain=gain)
        
        results[id] = {'recall': recall, 'precision': prec, 'rr': rr, 'ndcg': ndcg}
    
    return results

def mean_scores(per_query) -> dict:
    
    prec = 0
    recall = 0
    mrr = 0
    ndcg = 0
    n = len(per_query)
    for _, scores in per_query.items():
        
        prec += scores['precision']
        recall += scores['recall']
        mrr += scores['rr']
        ndcg += scores['ndcg']
    
    if n == 0:
        return {'n': 0}

    precision = float(prec / n)
    recall = float(recall / n)
    mrr = float(mrr / n)
    ndcg = float(ndcg / n)
    
    return {'n': n, 'precision': precision, 'recall': recall, 'rr': mrr, 'ndcg': ndcg}

def merge_spans(spans) -> list[tuple[int, int]]:
    
    if len(spans) == 0:
        return []
    disjoint = []
    spans_sort = spans.copy()
    spans_sort.sort(key=lambda x: x[0])
    
    
    curr_start = spans_sort[0][0]
    curr_end = spans_sort[0][1]
    
    for span in spans_sort[1:]:
        
        if curr_end >= span[0]:
            curr_end = max(curr_end, span[1])
        else:
            disjoint.append(tuple([curr_start, curr_end]))
            
            curr_start = span[0]
            curr_end = span[1]
    
    disjoint.append(tuple([curr_start, curr_end]))
    
    return disjoint

def overlap_length(a, b) -> int:
    
    a_dis = merge_spans(a)
    b_dis = merge_spans(b)
    
    if len(a_dis) == 0 or len(b_dis) == 0:
        return 0
    
    i, j = 0,0
    count = 0
    
    while i < len(a_dis) and j < len(b_dis):
        
        count += max(0, min(a_dis[i][1], b_dis[j][1]) - max(a_dis[i][0], b_dis[j][0]))
        
        if a_dis[i][1] < b_dis[j][1]:
            i += 1
        elif a_dis[i][1] == b_dis[j][1]:
            i += 1
            j += 1
        else:
            j += 1

    return count

def evidence_at_budget(ranked_chunks, gold_spans, budget) -> dict:
    
    if gold_spans is None or len(gold_spans) == 0:
        raise ValueError("ERROR: No valid spans provided")
    
    total_tokens = 0
    selected = []
    for chunk in ranked_chunks:
        
        if total_tokens + chunk[2] > budget:
            
            break
        
        total_tokens += chunk[2]
        selected.append([chunk[0], chunk[1]])
    
    hit = overlap_length(selected, gold_spans)
    
    gold_chars = sum([t[1] - t[0] for t in merge_spans(gold_spans)])
    recall = float(hit / gold_chars) if gold_chars > 0 else None
    
    pred_chars = sum([t[1] - t[0] for t in merge_spans(selected)]) 
    precision = float(hit / pred_chars) if pred_chars > 0 else None
    tot_sel = len(selected)

    return {'evidence_recall': recall, 'context_precision': precision, 'n_selected': tot_sel, 'tokens_used': total_tokens}