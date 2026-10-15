from chunking.base import trim_span
from chunking.recursive import PAR_SEP
from data.constants import SCIFACT_PATH, QASPER_PATH
from retrieval.qasper import render_paper
from retrieval.data import load_corpus
from pathlib import Path
from metadata.structure import detect_headings, build_sections
from metadata.attach import build_embed_text
from chunking.base import Chunk, count_tokens
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
import json

SAVE_PATH = Path(__file__).resolve().parent.parent / "cache"
CONFIG = Path(__file__).resolve().parent.parent / "results/env.json"
BOUNDS_PATH = Path(__file__).resolve().parent.parent / "results/bounds"

SCI_DATA = SCIFACT_PATH / "corpus.jsonl"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SWEEP = [64, 96, 128, 192, 254]


def sentence_bounds(texts, doc_ids, nlp, save_path):
    
    doc_splits = nlp.pipe(texts, batch_size=32)
    
    spans_by_doc = {}
    
    for doc_id, text, parsed in zip(doc_ids, texts, doc_splits):
        
        spans = []
        
        for sent in parsed.sents:
            
            s, e = trim_span(text, sent.start_char, sent.end_char)
            
            if s < e:
                
                spans.append((s, e))
            
        spans_by_doc[doc_id] = spans
    
    
    (save_path).write_text(json.dumps(spans_by_doc), encoding='utf-8')
    print(f"done: {save_path.name}, {len(spans_by_doc)} docs")
    return spans_by_doc

def load_spans(path):
    
    loaded = json.loads(path.read_text(encoding='utf-8'))
    
    return {doc_id: [tuple(span) for span in spans] for doc_id, spans in loaded.items()}

def cached_sentence_bounds(texts, doc_ids, nlp, save_path):
    
    if save_path.exists():
        print(f"cached: {save_path.name}")
        return load_spans(save_path)
    
    return sentence_bounds(texts, doc_ids, nlp, save_path)

def run_setence_bounds():
    
    sci = load_corpus(SCI_DATA)
    
    sci_doc_ids = sorted([k for k, _ in sci.items()])
    sci_texts = [sci[i]['text'] for i in sci_doc_ids]
    
    qas_test = json.loads((QASPER_PATH / "qasper-test-v0.3.json").read_text(encoding='utf-8'))
    qas_dev = json.loads((QASPER_PATH / "qasper-dev-v0.3.json").read_text(encoding='utf-8'))
    
    q_test_ids = sorted([k for k,_ in qas_test.items()])
    q_dev_ids = sorted([k for k,_ in qas_dev.items()])
    
    q_test_text = [render_paper(qas_test[i])[0] for i in q_test_ids]
    q_dev_text = [render_paper(qas_dev[i])[0] for i in q_dev_ids]
    
    nlp = spacy.load("en_core_web_sm", disable = ["ner", "lemmatizer"])
    
    sci_fact_spans = cached_sentence_bounds(sci_texts, sci_doc_ids, nlp, SAVE_PATH / "sentences_scifact.json")
    q_test_spans = cached_sentence_bounds(q_test_text, q_test_ids, nlp, SAVE_PATH / "sentences_qasper_test.json")
    q_dev_spans = cached_sentence_bounds(q_dev_text, q_dev_ids, nlp, SAVE_PATH / "sentences_qasper_dev.json")
    
    return {
        'scifact': (sci_doc_ids, sci_texts, sci_fact_spans),
        'qasper_test': (q_test_ids, q_test_text, q_test_spans),
        'qasper_dev': (q_dev_ids, q_dev_text, q_dev_spans),
    }

def upper_bound(model):

    config = json.loads((CONFIG).read_text(encoding='utf-8'))
    
    models = config['models']
    
    if model not in models:
        raise ValueError("ERROR: Model not supported")
    
    m = SentenceTransformer(model, device='cpu')
    tok = m.tokenizer
    
    max_seq = models[model]['max_seq_length']
    ceiling = max_seq - 2
    
    qas_test = json.loads((QASPER_PATH / "qasper-test-v0.3.json").read_text(encoding='utf-8'))
    papers = list(qas_test.values())
    
    prefix = count_prefixes(papers, tok)
    
    return {
        'max_seq_length': {'value': max_seq, 'evidence': 'read'},
        'special_tokens': {'value': 2, 'evidence': 'read'},
        'ceiling': {'value': ceiling, 'evidence': 'computed'},
        'prefix_tokens': {'value': prefix, 'evidence': 'measured'},
        'budget_with_prefix': {
            'value': {mode: ceiling - int(np.ceil(stats['p95'])) for mode, stats in prefix.items()},
            'evidence': 'computed',
        }
    }
    
    
def count_prefixes(papers, tok):
    lengths = {"title": [], "title_section": []}

    for paper in papers:
        title = paper["title"]

        
        dummy = Chunk("x", 0, 0, 0, "", 0, metadata={"title": title, "section_path": []})
        lengths["title"].append(count_tokens(tok, build_embed_text(dummy, mode="title")))

        doc_text = render_paper(paper)[0]
        for sec in build_sections(detect_headings(doc_text), len(doc_text)):
            dummy = Chunk("x", 0, 0, 0, "", 0, metadata={"title": title, "section_path": sec.path})
            lengths["title_section"].append(count_tokens(tok, build_embed_text(dummy, mode="title_section")))

    final = {}
    for mode, vals in lengths.items():
        p50, p95 = np.percentile(vals, [50, 95])
        final[mode] = {"p50": float(p50), "p95": float(p95), "max": int(max(vals)), "n": len(vals)}
    return final

def document_truncation_rates(texts, ceiling, tok):
    
    tokens = []
    truncated = 0
    past_ceiling = 0
    for text in texts:
        
        toks = count_tokens(tok, text)
        
        if toks > ceiling:
            truncated += 1

            past_ceiling += toks - ceiling
            
        tokens.append(toks)
    
    p50, p90, p99 = np.percentile(np.array(tokens), [50, 90, 99])
    max_toks = np.max(tokens)
    
    truncation_rate = float(truncated / len(tokens))
    past_ceiling = float(past_ceiling / sum(tokens))
    
    return {
        'n': len(tokens),
        'p50': float(p50),
        'p90': float(p90),
        'p99': float(p99),
        'max': int(max_toks),
        'truncation_rate': truncation_rate,
        'past_ceiling_ratio': past_ceiling
    }
    
def floor(sentence_spans, doc_texts, tok):
    
    tokens = []
    
    for doc_id, text in doc_texts.items():
        
        doc_spans = sentence_spans[doc_id]
        
        for span in doc_spans:
            
            tokens.append(count_tokens(tok, text[span[0]:span[1]]))
    
    p50, p90, p99 = np.percentile(np.array(tokens), [50, 90, 99])
    
    return {
        'n': len(tokens),
        'p50': float(p50),
        'p90': float(p90),
        'p99': float(p99)
    }

def measure_structure(texts):
    
    n = len(texts)
    blank_lines = 0
    for text in texts:
        
        if PAR_SEP.search(text) is not None:
            blank_lines +=1
    
    return float(blank_lines / n)

def calculate_budget(sweep: list[int]) -> dict:
    
    w = 4096
    i = 300
    q = 50
    a = 500
    h = 20
    r = 1.0
    
    available = w - (i + q + a)
    
    ks = {}
    for c in sweep:
        k = available // (r * c + h)

        ks[str(c)] = int(k)
    
    return {
        'assumptions': {'window': w, 'instructions': i, 'question': q, 'answer_reserve': a,
                        'header_per_chunk': h, 'r': r, 'r_note': 'assumed, not measured'},
        'available': available,
        'k_max': ks,
        'evidence': 'computed'
    }
        
        
if __name__ == "__main__":
    
    corpora = run_setence_bounds()
    
    m = SentenceTransformer(MODEL, device='cpu')
    tok = m.tokenizer
    
    hard = upper_bound(MODEL)
    ceiling = hard['ceiling']['value']
    
    bounds = {
        'hard_ceiling': hard,
        'document_lengths': {'value': {}, 'evidence': 'measured'},
        'sentence_lengths': {'value': {}, 'evidence': 'measured'},
        'structure': {'value': {}, 'evidence': 'measured'},
        'budget': calculate_budget(SWEEP),
    }
    
    for name in ['scifact', 'qasper_test']:
        
        ids, texts, spans = corpora[name]
        
        bounds['document_lengths']['value'][name] = document_truncation_rates(texts, ceiling, tok)
        bounds['sentence_lengths']['value'][name] = floor(spans, dict(zip(ids, texts)), tok)
        bounds['structure']['value'][name] = measure_structure(texts)
    
    BOUNDS_PATH.mkdir(parents=True, exist_ok=True)
    (BOUNDS_PATH / "bounds.json").write_text(json.dumps(bounds, indent=2), encoding='utf-8')
    
    print(json.dumps(bounds, indent=2))