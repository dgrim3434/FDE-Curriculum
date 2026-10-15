from collections import defaultdict, Counter
from chunking.base import Chunk
import numpy as np
import re

def by_doc(chunks) -> dict[str, list[Chunk]]:
    
    groups = defaultdict(list)
    
    for c in chunks:
        
        groups[c.doc_id].append(c)
    
    for doc_id in groups:
        
        groups[doc_id].sort(key=lambda x: x.chunk_index)

    return groups

def size_stats(chunks, budget, max_seq_length, floor) -> dict:
    
    results = {}
    
    tokens = np.array([c.n_tokens for c in chunks])
    
    if len(tokens) == 0:
        return {'n': 0}
    
    results['n'] = len(tokens)
    results['min'] = int(np.min(tokens))
    results['max'] = int(np.max(tokens))
    results['mean'] = float(np.mean(tokens))
    p5, p50, p95 = np.percentile(tokens, [5,50, 95])
    results['p5'], results['p50'], results['p95'] = float(p5), float(p50), float(p95)
    results['cv'] = float(tokens.std() / tokens.mean()) if tokens.mean() > 0 else None
    results['fill_mean'] = (tokens / budget).mean()
    results['above_ceiling_rate'] = float((tokens + 2 > max_seq_length).mean())
    results['below_floor_rate'] = float((tokens < floor).mean())

    return results

def single_chunk_doc_rate(chunks) -> float | None:
    
    if len(chunks) == 0:
        return None
    
    counts = Counter([c.doc_id for c in chunks])
    single = 0
    
    for count in counts.values():
        
        if count == 1:
            single += 1
    
    return float(single / len(counts))

DANGLING = {"it", "its", "this", "these", "those", "they", "their", "such",
            "however", "moreover", "therefore", "furthermore", "he", "she"}

def boundary_stats(chunks, docs, sentence_spans) -> dict:
    
    
    
    results = {}
    
    
    doc_groups = by_doc(chunks)
    
    aligned_end = 0
    aligned_start = 0
    mid_word = 0
    dangling_starts = 0
    cuts = 0
    
    for doc, cs in doc_groups.items():
        
        if len(cs) > 0:
            cuts += 1
        starts = set([s[0] for s in sentence_spans[doc]])
        ends = set([s[1] for s in sentence_spans[doc]])
        
        index = 0
        
        while index < len(cs):
            
            chunk = cs[index]
            
            if index != 0:
                start = chunk.start
                
                if start in starts:
                    aligned_start +=1
                
                m = re.match(r"\W*(\w+)", chunk.text)
                first = m.group(1).casefold() if m else None
            
                if first in DANGLING:
                    dangling_starts += 1
            
            if index != len(cs) -1:
                
                end = chunk.end
                
                if end in ends:
                    aligned_end += 1
                
                if docs[doc][end].isalnum() and docs[doc][end - 1].isalnum():
                    mid_word += 1
            
            index += 1
    
    internal_cuts = len(chunks) - cuts
    results['n_internal_cuts'] = internal_cuts
    
    if internal_cuts == 0:
        results['sentence_aligned_end_rate'] = None
        results['sentence_aligned_start_rate'] = None
        results['mid_word_rate'] = None
        results['dangling_start_rate'] = None
    else:
        results['sentence_aligned_end_rate'] = float(aligned_end / internal_cuts)
        results['sentence_aligned_start_rate'] = float(aligned_start / internal_cuts)
        results['mid_word_rate'] = float(mid_word / internal_cuts)
        results['dangling_start_rate'] = float(dangling_starts / internal_cuts)
    
    return results

def union_length(chunks):
    
    if len(chunks) == 0:
        return None
    
    chunk_copy = chunks.copy()
    
    chunk_copy.sort(key=lambda x: x.start)
    
    total_toks = 0
    curr_start = chunk_copy[0].start
    curr_end = chunk_copy[0].end
    
    for chunk in chunk_copy[1:]:
        
        if curr_end >= chunk.start:
            curr_end = chunk.end
        else:
            
            total_toks += curr_end - curr_start
            curr_start = chunk.start
            curr_end = chunk.end
    
    total_toks += curr_end - curr_start
    
    return total_toks    
        
   
def overlap_stats(chunks, docs, sentence_spans) -> dict:
    
    overlap = []
    n_straddling = 0
    staddle_protected = 0
    total_length = 0
    union_len = 0
    doc_groups = by_doc(chunks)
    
    for doc, cs in doc_groups.items():
        ends = set([s[1] for s in sentence_spans[doc]])
        
        index = 0
        
        union_len += union_length(cs)
        while index < len(cs):
            
            chunk = cs[index]
            total_length += len(chunk.text)
            
            if index != 0:
                overlap.append(max(0, cs[index - 1].end - chunk.start))
                
            if index != len(cs) -1:
                
                if chunk.end not in ends:
                    n_straddling += 1
                    
                    start, end = [(s[0], s[1]) for s in sentence_spans[doc] if s[0] < chunk.end and s[1] > chunk.end][0]
                    
                    if cs[index + 1].start <= start and cs[index + 1].end >= end:
                        staddle_protected += 1
            
            index += 1
                
    results = {}      
    results['mean_overlap_chars'] = float(np.mean(overlap)) if len(overlap) > 0 else None
    results['max_overlap_chars'] = int(np.max(overlap)) if len(overlap) > 0 else None
    results['redundancy_ratio'] = float(total_length / union_len)
    results['n_straddling'] = n_straddling
    results['straddle_protection_rate'] = float(staddle_protected / n_straddling) if n_straddling > 0 else None
    
    return results

def chunk_pieces(chunk, doc_text, spans) -> list[str]:
    
    c_start = chunk.start
    c_end = chunk.end
    
    sorted_spans = spans.copy()
    sorted_spans.sort(key= lambda x: x[0])
    
    peices = []
    
    for span in sorted_spans:
        
        if span[0] > c_end:
            return peices
        
        if span[1] > c_start:
            
            text = doc_text[max(c_start, span[0]): min(c_end, span[1])]
            
            if len(text.strip()) == 0:
                continue
            peices.append(text.strip())
    
    return peices
    
def representation_fit(peice_vecs, chunk_vec) -> float:
    
    return float(np.mean(peice_vecs @ chunk_vec))

def coherence_scores(chunks, docs, sentence_spans, embed_fn) -> list[float]:
    
    flat = []
    ranges = []
        
    for chunk in chunks:
        start = len(flat)
            
        peices = chunk_pieces(chunk, docs[chunk.doc_id], sentence_spans[chunk.doc_id])
            
        if len(peices) >= 2:
            flat.extend(peices)
            end = len(flat)
            
            ranges.append([start, end])
        else:
            ranges.append(None)
    
    if len(flat) == 0:
        return []
    
    
    P = embed_fn(flat)
    C = embed_fn([c.text for c in chunks])
    
    
    results = []
    
    for i in range(len(C)):
        
        if ranges[i] is None:
            results.append(float("nan"))
            continue
        
        s = ranges[i][0]
        e = ranges[i][1]
        
        peice_vecs = P[s:e, :]
        chunk_vec = C[i]
        
        results.append(representation_fit(peice_vecs, chunk_vec))
    
    return results

def coherence_summary(scores) -> dict:
    
    results = {}
    n = len(scores)
    non_nan = [s for s in scores if not np.isnan(s)]
    n_nan = n - len(non_nan)
    
    results['n'], results['n_nan'] = n, n_nan
    
    if n != n_nan:
        mean = float(np.mean(non_nan))
        p5, p50, p95 = np.percentile(non_nan, [5, 50, 95])
    
        ps = {'mean': mean, 'p5': float(p5), 'p50': float(p50), 'p95': float(p95)}
    else:
        ps = {'mean': None, 'p5': None, 'p50': None, 'p95': None}
        
        
    results.update(ps)
    
    return results

def random_chunk_baseline(chunks, docs, sentence_spans, embed_fn, n=300, seed=0) -> dict:
    
    rng = np.random.default_rng(seed=seed)

    ks = []
    
    doc_groups = by_doc(chunks)
    
    for doc, cs in doc_groups.items():
        
        for c in cs:
            
            peices = len(chunk_pieces(c, docs[doc], sentence_spans[doc]))
            
            if peices >= 2:
                
                ks.append(peices)
    
    if len(ks) == 0:
        return coherence_summary([])
    
    docs_ids = [k for k,_ in docs.items()]
    docs_ids = [k for k in docs_ids if len(sentence_spans[k]) > 0]
    rand_chunks = []
    rand_peices = []
    chunk_indicies = []
    for i in range(n):
        
        k = int(rng.choice(ks))

        if k > len(docs_ids):
            continue
        rand_docs = rng.choice(docs_ids, size=k, replace=False)
        
        peices = []
        
        for doc in rand_docs:
            
            span = sentence_spans[doc]

            sent = int(rng.choice(range(0, len(span))))
            
            start, end = span[sent][0], span[sent][1]
            
            peices.append(docs[doc][start: end])
        
        text = " ".join(peices)
        
        rand_chunks.append(text)
        
        s = len(rand_peices)
        rand_peices.extend(peices)
        e = len(rand_peices)
        
        chunk_indicies.append([s, e])
    
    if len(rand_peices) == 0:
        return coherence_summary([])
    
    P = embed_fn(rand_peices)
    C = embed_fn(rand_chunks)
    
    scores = []
    
    for i in range(len(C)):
        
        s, e = chunk_indicies[i]
        
        peice_vecs = P[s:e, :]
        chunk_vec = C[i]
        
        scores.append(representation_fit(peice_vecs, chunk_vec))
    
    
    return coherence_summary(scores)

def analyze(chunks, docs, sentence_spans, budget, max_seq_length, floor, embed_fn=None) -> dict:
    
    results = {}
    
    r1 = size_stats(chunks, budget, max_seq_length=max_seq_length, floor=floor)
    
    results['size'] = r1
    
    results['single_chunk_doc_rate'] = single_chunk_doc_rate(chunks)
    
    r2 = boundary_stats(chunks, docs, sentence_spans)
    
    results['boundary'] = r2
    
    r3 = overlap_stats(chunks, docs, sentence_spans)
    
    results['overlap'] = r3
    
    if embed_fn is None:
        
        results['coherence'] = None
        return results
    
    scores = coherence_scores(chunks, docs, sentence_spans, embed_fn)
    
    coh_summary = coherence_summary(scores)
    
    results['coherence'] = coh_summary
    
    return results
    
    