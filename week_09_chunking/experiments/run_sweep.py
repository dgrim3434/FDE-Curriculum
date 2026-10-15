from data.constants import SCIFACT_PATH, QASPER_PATH
from retrieval.data import load_beir
from retrieval.qasper import render_paper, qasper_doc_fields, qasper_questions
from retrieval.index import make_embed_fn, embed_with_cache, rank_documents, rank_chunks, texts_for_embedding
from retrieval.metrics import evaluate_run, mean_scores, evidence_at_budget
from chunking.base import make_chunk, count_tokens
from chunking.fixed import FixedSizeChunker
from chunking.sentence import SentenceChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker, percentile_threshold
from chunking.splitter import make_spacy_splitter
from metadata.structure import detect_headings, build_sections
from metadata.attach import attach_metadata, set_embed_text
from metadata.store import save_chunks
from analysis.analyzer import analyze, coherence_scores, coherence_summary, random_chunk_baseline
from experiments.bounds import load_spans
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import hashlib
import spacy
import json
import time
import csv
import sys

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

RETRIEVAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COHERENCE_MODEL = "sentence-transformers/all-mpnet-base-v2"

K = 10
MAX_SEQ = 256
SIZES = [64, 96, 128, 192, 254]
BUDGETS = [256, 512, 1024]
FLOOR = {'scifact': 58, 'qasper': 52}
OVERLAP_P90 = 52                               
PREFIX_CEILING = 230                          
PERCENTILES = [50, 60, 70, 80, 85, 90, 95, 98]
COHERENCE_SAMPLE = {'scifact': 500, 'qasper': 60}


def cfg(name, corpus, strategy, size, overlap=0, min_tokens=0, window=1, prefix='none'):
    
    return {'name': name, 'corpus': corpus, 'strategy': strategy, 'size': size, 'overlap': overlap,
            'min_tokens': min_tokens, 'window': window, 'prefix': prefix}

def config_id(c):
    
    key = {k: v for k, v in c.items() if k != 'name'}
    
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode('utf-8')).hexdigest()[:10]

STAGE1 = (
    [cfg('A0_whole', 'scifact', 'whole', None)]
    + [cfg(f'A1_fixed_{s}', 'scifact', 'fixed', s) for s in SIZES]
    + [cfg('A2_sentence_128', 'scifact', 'sentence', 128),
       cfg('A2_semantic_128', 'scifact', 'semantic', 128, min_tokens=58, window=2)]
    + [cfg(f'B1_fixed_{s}', 'qasper', 'fixed', s) for s in SIZES]
    + [cfg('B2_sentence_128', 'qasper', 'sentence', 128),
       cfg('B2_recursive_128', 'qasper', 'recursive', 128, min_tokens=52),
       cfg('B2_semantic_128', 'qasper', 'semantic', 128, min_tokens=52, window=2)]
)

def stage2_configs(strategy, size):
    
    min_tokens = 52 if strategy in ('recursive', 'semantic') else 0
    window = 2 if strategy == 'semantic' else 1
    psize = min(size, PREFIX_CEILING)
    
    return [
        cfg(f'B3_fixed_{size}_o{OVERLAP_P90}', 'qasper', 'fixed', size, overlap=OVERLAP_P90),
        cfg(f'B4_{strategy}_{psize}_none', 'qasper', strategy, psize, min_tokens=min_tokens, window=window),
        cfg(f'B4_{strategy}_{psize}_title', 'qasper', strategy, psize, min_tokens=min_tokens, window=window, prefix='title'),
    ]


def sample_docs(doc_ids, n, seed=0):
    
    rng = np.random.default_rng(seed)
    
    return sorted(rng.choice(sorted(doc_ids), size=min(n, len(doc_ids)), replace=False).tolist())

def load_data():
    
    corpus, queries, qrels = load_beir(SCIFACT_PATH)
    sci_docs = {d: corpus[d]['text'] for d in sorted(corpus)}
    sci_fields = {d: {'title': corpus[d]['title'], 'authors': None, 'date': None, 'date_raw': None,
                      'provenance': {'title': 'source_field', 'authors': 'not_found', 'date': 'not_found'}}
                  for d in sci_docs}
    
    papers = json.loads((QASPER_PATH / "qasper-test-v0.3.json").read_text(encoding='utf-8'))
    dev = json.loads((QASPER_PATH / "qasper-dev-v0.3.json").read_text(encoding='utf-8'))
    
    q_docs = {pid: render_paper(papers[pid])[0] for pid in sorted(papers)}
    q_dev_docs = {pid: render_paper(dev[pid])[0] for pid in sorted(dev)}
    q_fields = {pid: qasper_doc_fields(papers[pid]) for pid in sorted(papers)}
    
    q_items = []
    clusters = {}
    accounting = {}
    
    for pid in sorted(papers):
        
        questions, stats = qasper_questions(papers[pid], q_docs[pid])
        
        for k, v in stats.items():
            accounting[k] = accounting.get(k, 0) + v
        
        for q in questions:
            q_items.append((pid, q))
            clusters[q['question_id']] = pid
    
    (RESULTS / "qasper_test_accounting.json").write_text(json.dumps(accounting, indent=2), encoding='utf-8')
    (RESULTS / "qasper_test_clusters.json").write_text(json.dumps(clusters, indent=2), encoding='utf-8')
    print(f"qasper test accounting: {accounting}")
    
    return {
        'scifact': {'docs': sci_docs, 'spans': load_spans(CACHE / "sentences_scifact.json"),
                    'fields': sci_fields, 'queries': queries, 'qrels': qrels,
                    'tuning_docs': sci_docs,                         # size-matching uses no labels
                    'sample': sample_docs(list(sci_docs), COHERENCE_SAMPLE['scifact'])},
        'qasper': {'docs': q_docs, 'spans': load_spans(CACHE / "sentences_qasper_test.json"),
                   'fields': q_fields, 'items': q_items,
                   'tuning_docs': q_dev_docs,
                   'tuning_spans': load_spans(CACHE / "sentences_qasper_dev.json"),
                   'sample': sample_docs(list(q_docs), COHERENCE_SAMPLE['qasper'])},
    }

def make_cached_splitter(data, nlp):
    
    by_text = {}
    
    for docs, spans in [(data['scifact']['docs'], data['scifact']['spans']),
                        (data['qasper']['docs'], data['qasper']['spans']),
                        (data['qasper']['tuning_docs'], data['qasper']['tuning_spans'])]:
        
        for d, text in docs.items():
            by_text[text] = spans[d]
    
    fallback = make_spacy_splitter(nlp)
    
    def split(text):
        
        if text in by_text:
            return by_text[text]
        
        return fallback(text)
    
    return split


def tune_semantic(docs, tok, splitter, seg_embed, size, window, min_tokens):
    
    fixed = FixedSizeChunker(tok, size, 0)
    target = float(np.mean([c.n_tokens for d in sorted(docs) for c in fixed.chunk(docs[d], d)]))
    
    ch = SemanticChunker(tok, splitter, seg_embed, budget=size, threshold=None, window=window, min_tokens=min_tokens)
    prepared = {d: ch.prepare(docs[d]) for d in sorted(docs)}
    all_d = np.concatenate([prepared[d][1] for d in sorted(docs) if len(prepared[d][1]) > 0])
    
    records = []
    
    for p in PERCENTILES:
        
        ch.threshold = percentile_threshold(all_d, p)
        sizes = [c.n_tokens for d in sorted(docs) for c in ch.chunk_prepared(docs[d], d, *prepared[d])]
        records.append({'percentile': p, 'threshold': float(ch.threshold), 'mean_tokens': float(np.mean(sizes))})
    
    best = min(records, key=lambda r: abs(r['mean_tokens'] - target))
    matched = abs(best['mean_tokens'] - target) / target <= 0.05
    
    return best['percentile'], {'target_mean_fixed': target, 'records': records, 'chosen': best,
                                'within_5_percent': bool(matched)}

def build_chunker(c, tok, splitter):
    
    if c['strategy'] == 'fixed':
        return FixedSizeChunker(tok, c['size'], c['overlap'])
    
    if c['strategy'] == 'sentence':
        return SentenceChunker(tok, splitter, c['size'], c['overlap'])
    
    if c['strategy'] == 'recursive':
        return RecursiveChunker(tok, splitter, c['size'], c['min_tokens'], False)
    
    raise ValueError(f"ERROR: unknown strategy {c['strategy']}")

def chunk_corpus(c, corpus, tok, splitter, seg_embed):
    
    docs = corpus['docs']
    order = sorted(docs)
    extra = {}
    prepared = None
    
    if c['strategy'] == 'semantic':
        
        p, tuning = tune_semantic(corpus['tuning_docs'], tok, splitter, seg_embed, c['size'], c['window'], c['min_tokens'])
        
        ch = SemanticChunker(tok, splitter, seg_embed, budget=c['size'], threshold=None,
                             window=c['window'], min_tokens=c['min_tokens'])
        prepared = {d: ch.prepare(docs[d]) for d in order}
        all_d = np.concatenate([prepared[d][1] for d in order if len(prepared[d][1]) > 0])
        ch.threshold = percentile_threshold(all_d, p)          # per-corpus threshold at the tuned percentile
        
        extra = {'semantic_percentile': p, 'semantic_threshold': float(ch.threshold), 'semantic_tuning': tuning}
    
    elif c['strategy'] != 'whole':
        ch = build_chunker(c, tok, splitter)
    
    chunks = []
    ranges = {}
    
    for d in order:
        
        text = docs[d]
        
        if c['strategy'] == 'whole':
            one = make_chunk(text, d, 0, 0, len(text), tok)
            cs = [one] if one is not None else []
        elif c['strategy'] == 'semantic':
            cs = ch.chunk_prepared(text, d, *prepared[d])
        else:
            cs = ch.chunk(text, d)
        
        ranges[d] = (len(chunks), len(chunks) + len(cs))
        chunks.extend(cs)
    
    return chunks, ranges, extra

def add_prefix(c, chunks, ranges, corpus):
    
    for d, (s, e) in ranges.items():
        
        if s == e:
            continue
        
        text = corpus['docs'][d]
        sections = build_sections(detect_headings(text), len(text))
        attach_metadata(chunks[s:e], corpus['fields'][d], sections)
    
    set_embed_text(chunks, c['prefix'])


def mean_ignoring_none(per_query):
    
    keys = list(next(iter(per_query.values())).keys())
    out = {'n': len(per_query)}
    
    for k in keys:
        
        vals = [v[k] for v in per_query.values() if v[k] is not None]
        out[k] = float(np.mean(vals)) if len(vals) > 0 else None
    
    return out

def evaluate_scifact(chunks, C, corpus, embed):
    
    qids = sorted(corpus['qrels'])
    Q = embed([corpus['queries'][q] for q in qids])
    
    run = rank_documents(Q, C, chunks, qids, K)
    per_query = evaluate_run(run, corpus['qrels'], K)
    
    return per_query, mean_scores(per_query)

def evaluate_qasper(chunks, C, ranges, corpus, embed):
    
    items = corpus['items']
    Qv = embed([q['question'] for _, q in items])
    
    per_query = {}
    
    for (pid, q), qvec in zip(items, Qv):
        
        s, e = ranges[pid]
        order = rank_chunks(qvec, C[s:e])
        ranked = [(chunks[s + i].start, chunks[s + i].end, chunks[s + i].n_tokens) for i in order]
        
        rec = {}
        
        for B in BUDGETS:
            
            best = max((evidence_at_budget(ranked, gold, B) for gold in q['gold']),
                       key=lambda r: r['evidence_recall'])
            rec[f'er@{B}'] = best['evidence_recall']
            rec[f'cp@{B}'] = best['context_precision']
        
        per_query[q['question_id']] = rec
    
    return per_query, mean_ignoring_none(per_query)

def coherence_sample(chunks, ranges, corpus, coh_embed):
    
    sample_chunks = []
    
    for d in corpus['sample']:
        s, e = ranges[d]
        sample_chunks.extend(chunks[s:e])
    
    scores = coherence_scores(sample_chunks, corpus['docs'], corpus['spans'], coh_embed)
    pairs = [[c.n_tokens, None if np.isnan(sc) else float(sc)] for c, sc in zip(sample_chunks, scores)]
    
    return coherence_summary(scores), pairs, sample_chunks

def write_summary_row(c, cid, n_chunks, scores, coherence, truncation):
    
    path = RESULTS / f"summary_{c['corpus']}.csv"
    row = {'name': c['name'], 'config_id': cid, 'corpus': c['corpus'], 'strategy': c['strategy'],
           'size': c['size'], 'overlap': c['overlap'], 'prefix': c['prefix'], 'n_chunks': n_chunks,
           'truncation_rate': round(truncation, 4),
           'coherence_mean': None if coherence['mean'] is None else round(coherence['mean'], 4)}
    
    for k, v in scores.items():
        row[k] = v if (v is None or k == 'n') else round(v, 4)
    
    new = not path.exists()
    
    with open(path, 'a', newline='', encoding='utf-8') as f:
        
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        
        if new:
            writer.writeheader()
        
        writer.writerow(row)

def run_config(c, data, tok, splitter, embed, coh_embed):
    
    cid = config_id(c)
    run_path = RESULTS / "runs" / f"{cid}.json"
    
    if run_path.exists():
        print(f"skip (done): {c['name']} {cid}")
        return
    
    t0 = time.time()
    corpus = data[c['corpus']]
    
    chunks, ranges, extra = chunk_corpus(c, corpus, tok, splitter, embed)
    
    if c['prefix'] != 'none':
        add_prefix(c, chunks, ranges, corpus)
    
    save_chunks(chunks, CACHE / f"chunks_{cid}.jsonl")
    C = embed_with_cache(chunks, embed, CACHE / f"vec_{cid}.npz")
    
    truncation = float(np.mean([count_tokens(tok, t) + 2 > MAX_SEQ for t in texts_for_embedding(chunks)]))
    
    if c['corpus'] == 'scifact':
        per_query, scores = evaluate_scifact(chunks, C, corpus, embed)
    else:
        per_query, scores = evaluate_qasper(chunks, C, ranges, corpus, embed)
    
    budget = c['size'] if c['size'] is not None else MAX_SEQ - 2
    analysis = analyze(chunks, corpus['docs'], corpus['spans'], budget, MAX_SEQ, FLOOR[c['corpus']])
    coherence, pairs, sample_chunks = coherence_sample(chunks, ranges, corpus, coh_embed)
    analysis['coherence'] = coherence
    
    baseline_path = RESULTS / "analysis" / f"baseline_{c['corpus']}.json"
    
    if c['strategy'] == 'fixed' and c['size'] == 128 and c['overlap'] == 0 and not baseline_path.exists():
        baseline = random_chunk_baseline(sample_chunks, corpus['docs'], corpus['spans'], coh_embed, n=300, seed=0)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    
    for folder in ['configs', 'runs', 'analysis']:
        (RESULTS / folder).mkdir(parents=True, exist_ok=True)
    
    (RESULTS / "configs" / f"{cid}.json").write_text(json.dumps(
        {'config': c, 'config_id': cid, 'n_chunks': len(chunks), 'truncation_rate': truncation,
         'retrieval_model': RETRIEVAL_MODEL, 'coherence_model': COHERENCE_MODEL, **extra}, indent=2), encoding='utf-8')
    (RESULTS / "analysis" / f"{cid}.json").write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    (RESULTS / "analysis" / f"{cid}_coherence_pairs.json").write_text(json.dumps(pairs), encoding='utf-8')
    run_path.write_text(json.dumps({'config_id': cid, 'name': c['name'], 'scores': scores,
                                    'per_query': per_query}, indent=2), encoding='utf-8')
    
    write_summary_row(c, cid, len(chunks), scores, coherence, truncation)
    print(f"done: {c['name']} {cid}  chunks={len(chunks)}  {scores}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    
    stage = sys.argv[1]
    
    if stage == 'stage1':
        configs = STAGE1
    elif stage == 'stage2':
        configs = stage2_configs(sys.argv[2], int(sys.argv[3]))
    else:
        raise ValueError("usage: run_sweep stage1 | stage2 <strategy> <size>")
    
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    model = SentenceTransformer(RETRIEVAL_MODEL, device='cuda')
    tok = model.tokenizer
    embed = make_embed_fn(model, prefix="", batch_size=128)
    coh_embed = make_embed_fn(SentenceTransformer(COHERENCE_MODEL, device='cuda'), prefix="", batch_size=128)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    
    data = load_data()
    splitter = make_cached_splitter(data, nlp)
    
    for c in configs:
        run_config(c, data, tok, splitter, embed, coh_embed)