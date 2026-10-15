from metadata.store import load_chunks
from metadata.structure import detect_headings
from experiments.bounds import load_spans
from data.constants import QASPER_PATH
from retrieval.qasper import render_paper
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

WINDOW = 4000                      # characters of the example paper shown in the span chart
CEILING = 254
BUDGET = 128
FLOOR = {'qasper': 52, 'scifact': 58}

QASPER_PANELS = [('fixed', 'B1_fixed_128'), ('sentence', 'B2_sentence_128'),
                 ('recursive', 'B2_recursive_128'), ('semantic', 'B2_semantic_128'),
                 ('fixed + overlap 52', 'B3_fixed_128_o52')]
SCIFACT_SERIES = [('fixed', 'A1_fixed_128'), ('sentence', 'A2_sentence_128'), ('semantic', 'A2_semantic_128')]


def config_ids():
    
    ids = {}
    
    for path in (RESULTS / "configs").glob("*.json"):
        
        c = json.loads(path.read_text(encoding='utf-8'))
        ids[c['config']['name']] = c['config_id']
    
    return ids

def available(series, ids):
    
    out = []
    
    for label, name in series:
        
        if name in ids and (CACHE / f"chunks_{ids[name]}.jsonl").exists():
            out.append((label, name, load_chunks(CACHE / f"chunks_{ids[name]}.jsonl")))
        else:
            print(f"not available yet: {name}")
    
    return out

def by_doc(chunks):
    
    groups = {}
    
    for c in chunks:
        groups.setdefault(c.doc_id, []).append(c)
    
    return groups

def mid_word(text, pos):
    
    return 0 < pos < len(text) and text[pos - 1].isalnum() and text[pos].isalnum()

def pick_paper(fixed_chunks):
    
    counts = {d: len(cs) for d, cs in by_doc(fixed_chunks).items()}
    ordered = sorted(counts, key=lambda d: (counts[d], d))
    
    return ordered[len(ordered) // 2]                  # the median paper by chunk count


# ---------------------------------------------------------------- figure 1: span chart
def span_chart(panels, doc_id, doc_text, spans):
    
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 1.5 * len(panels) + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    
    heading_starts = [h.start for h in detect_headings(doc_text) if h.start < WINDOW]
    sentence_ends = [e for _, e in spans[doc_id] if e < WINDOW]
    
    for ax, (label, name, chunks) in zip(axes, panels):
        
        for e in sentence_ends:
            ax.axvline(e, color='0.82', lw=0.6, zorder=0)
        
        for h in heading_starts:
            ax.axvline(h, color='0.35', lw=1.0, ls='--', zorder=0)
        
        vis = [c for c in by_doc(chunks).get(doc_id, []) if c.start < WINDOW]
        
        for k, c in enumerate(vis):
            
            start, end = c.start, min(c.end, WINDOW)
            ax.barh(k % 2, end - start, left=start, height=0.55,
                    color=['#4C72B0', '#55A868'][k % 2], edgecolor='black', lw=0.4, zorder=2)
            
            if c.end < WINDOW and mid_word(doc_text, c.end):
                ax.plot(c.end, 1.55, marker='v', color='red', ms=6, zorder=3)
        
        for prev, nxt in zip(vis, vis[1:]):
            
            if nxt.start < prev.end:
                ax.axvspan(nxt.start, min(prev.end, WINDOW), color='red', alpha=0.25, zorder=1)
        
        ax.set_yticks([])
        ax.set_ylim(-0.6, 1.9)
        ax.set_ylabel(f"{label}\n({len(vis)} chunks)", rotation=0, ha='right', va='center', fontsize=9)
    
    axes[-1].set_xlim(0, WINDOW)
    axes[-1].set_xlabel("character position in the document")
    fig.suptitle(f"Chunk spans, Qasper paper {doc_id}, first {WINDOW} characters, budget {BUDGET} tokens\n"
                 "bars = chunks (alternating rows)  |  light lines = sentence ends  |  dashed = headings  |  "
                 "red shading = overlap  |  red marker = mid-word cut", fontsize=10)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "span_chart.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: size distribution
def size_distribution(qasper_series, scifact_series):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    bins = np.arange(0, 272, 8)
    
    for ax, series, corpus in [(axes[0], qasper_series, 'qasper'), (axes[1], scifact_series, 'scifact')]:
        
        for label, name, chunks in series:
            
            tokens = np.array([c.n_tokens for c in chunks])
            p5, p50, p95 = np.percentile(tokens, [5, 50, 95])
            ax.hist(tokens, bins=bins, histtype='step', lw=1.6, density=True,
                    label=f"{label}: n={len(tokens)}, p5/p50/p95 = {p5:.0f}/{p50:.0f}/{p95:.0f}")
        
        ax.axvline(CEILING, color='black', ls='--', lw=1.2, label=f"ceiling {CEILING} (beyond = truncated)")
        ax.axvline(BUDGET, color='0.4', ls=':', lw=1.2, label=f"budget {BUDGET}")
        ax.axvline(FLOOR[corpus], color='tab:red', ls='-.', lw=1.2, label=f"floor {FLOOR[corpus]} (p90 sentence)")
        
        ax.set_title(f"{corpus}: chunk sizes across the whole corpus, budget {BUDGET}")
        ax.set_xlabel("chunk size (MiniLM tokens, without special tokens)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "size_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- the numbers for part 4 of the write-up
def phase2_stats(names, ids):
    
    stats = {}
    
    for name in names:
        
        if name not in ids:
            continue
        
        a = json.loads((RESULTS / "analysis" / f"{ids[name]}.json").read_text(encoding='utf-8'))
        size, boundary, overlap = a['size'], a['boundary'], a['overlap']
        
        stats[name] = {
            'n_chunks': size['n'], 'p5': size['p5'], 'p50': size['p50'], 'p95': size['p95'], 'cv': size['cv'],
            'below_floor_rate': size['below_floor_rate'], 'above_ceiling_rate': size['above_ceiling_rate'],
            'single_chunk_doc_rate': a['single_chunk_doc_rate'],
            'n_internal_cuts': boundary['n_internal_cuts'],
            'sentence_aligned_end_rate': boundary['sentence_aligned_end_rate'],
            'mid_word_rate': boundary['mid_word_rate'],
            'dangling_start_rate': boundary['dangling_start_rate'],
            'mean_overlap_chars': overlap['mean_overlap_chars'],
            'redundancy_ratio': overlap['redundancy_ratio'],
            'straddle_protection_rate': overlap['straddle_protection_rate'],
            'coherence_mean': a['coherence']['mean'] if a['coherence'] else None,
        }
    
    return stats


if __name__ == "__main__":
    
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ids = config_ids()
    
    qasper_panels = available(QASPER_PANELS, ids)
    scifact_series = available(SCIFACT_SERIES, ids)
    
    papers = json.loads((QASPER_PATH / "qasper-test-v0.3.json").read_text(encoding='utf-8'))
    spans = load_spans(CACHE / "sentences_qasper_test.json")
    
    doc_id = pick_paper(qasper_panels[0][2])
    doc_text = render_paper(papers[doc_id])[0]
    
    span_chart(qasper_panels, doc_id, doc_text, spans)
    size_distribution([p for p in qasper_panels if p[1] != 'B3_fixed_128_o52'], scifact_series)
    
    stats = {'example_paper': doc_id, 'window_chars': WINDOW,
             'configs': phase2_stats([n for _, n in QASPER_PANELS + SCIFACT_SERIES], ids)}
    (RESULTS / "phase2_stats.json").write_text(json.dumps(stats, indent=2), encoding='utf-8')
    
    print(f"example paper: {doc_id}")
    print(json.dumps(stats['configs'], indent=2))