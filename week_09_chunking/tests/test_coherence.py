import json
import math

import numpy as np
import pytest

from analysis.analyzer import (analyze, chunk_pieces, coherence_scores, coherence_summary,
                               random_chunk_baseline, representation_fit)
from chunking.base import Chunk


class StubEmbedder:
    """Topic-word counts (cat/dog/fish/bird + other), normalized. Counts calls; rejects empty input."""
    VOCAB = ["cat", "dog", "fish", "bird"]

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        if len(texts) == 0:
            raise ValueError("embed_fn called with an empty list")
        self.calls += 1
        rows = []
        for t in texts:
            words = [w.strip(".,").casefold() for w in t.split()]
            v = np.array([words.count(w) for w in self.VOCAB]
                         + [sum(w not in self.VOCAB for w in words)], dtype=float)
            if not v.any():
                v[-1] = 1.0
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)


def mk(doc_id, idx, doc, start, end):
    return Chunk(doc_id, idx, start, end, doc[start:end], len(doc[start:end].split()))


def spans_of(doc, sentences):
    out, pos = [], 0
    for s in sentences:
        i = doc.index(s, pos)
        out.append((i, i + len(s)))
        pos = i + len(s)
    return out


DOC = "Cat cat cat. Dog dog dog. Fish fish fish."        # sentences at [0,12) [13,25) [26,41)
SPANS = spans_of(DOC, ["Cat cat cat.", "Dog dog dog.", "Fish fish fish."])


# ---------------------------------------------------------------- chunk_pieces
def test_precondition_spans():
    assert SPANS == [(0, 12), (13, 25), (26, 41)]


def test_pieces_whole_sentences():
    assert chunk_pieces(mk("d", 0, DOC, 0, 25), DOC, SPANS) == ["Cat cat cat.", "Dog dog dog."]


def test_pieces_clipped_at_both_edges():
    assert chunk_pieces(mk("d", 0, DOC, 4, 20), DOC, SPANS) == ["cat cat.", "Dog dog"]


def test_pieces_chunk_inside_one_sentence():
    assert chunk_pieces(mk("d", 0, DOC, 14, 22), DOC, SPANS) == ["og dog d"]


def test_pieces_no_empty_piece_when_next_sentence_starts_at_chunk_end():
    doc = "One.Two."
    assert chunk_pieces(mk("d", 0, doc, 0, 4), doc, [(0, 4), (4, 8)]) == ["One."]


def test_pieces_are_stripped():
    untrimmed = [(0, 13), (12, 25)]                       # spans that include surrounding spaces
    assert chunk_pieces(mk("d", 0, DOC, 0, 25), DOC, untrimmed) == ["Cat cat cat.", "Dog dog dog."]


# ---------------------------------------------------------------- representation_fit
def test_representation_fit_known_values():
    assert representation_fit(np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    r = np.sqrt(0.5)
    assert representation_fit(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([r, r])) == pytest.approx(r)


# ---------------------------------------------------------------- coherence_scores
def test_scores_values_and_input_order():
    one_topic = "Cat cat cat. Cat cat."
    two_topic = "Cat cat cat. Dog dog dog."
    single = "Fish fish fish."
    docs = {"a": one_topic, "b": two_topic, "c": single}
    spans = {"a": spans_of(one_topic, ["Cat cat cat.", "Cat cat."]),
             "b": spans_of(two_topic, ["Cat cat cat.", "Dog dog dog."]),
             "c": [(0, len(single))]}
    chunks = [mk("b", 0, two_topic, 0, len(two_topic)),     # deliberately NOT grouped or sorted
              mk("c", 0, single, 0, len(single)),
              mk("a", 0, one_topic, 0, len(one_topic))]
    scores = coherence_scores(chunks, docs, spans, StubEmbedder())
    assert len(scores) == 3
    assert scores[0] == pytest.approx(np.sqrt(0.5))           # 50/50 two-topic chunk
    assert isinstance(scores[1], float) and math.isnan(scores[1])   # single piece → NaN
    assert scores[2] == pytest.approx(1.0)                   # one topic


def test_scores_embed_at_most_twice():
    emb = StubEmbedder()
    chunks = [mk("d", 0, DOC, 0, 25), mk("d", 1, DOC, 26, 41), mk("d", 2, DOC, 0, 41)]
    coherence_scores(chunks, {"d": DOC}, {"d": SPANS}, emb)
    assert emb.calls <= 2


def test_scores_all_single_piece_never_embeds_empty_list():
    chunks = [mk("d", 0, DOC, 0, 12), mk("d", 1, DOC, 13, 25)]
    scores = coherence_scores(chunks, {"d": DOC}, {"d": SPANS}, StubEmbedder())
    assert all(math.isnan(s) for s in scores)


# ---------------------------------------------------------------- coherence_summary
def test_summary_excludes_nan():
    s = coherence_summary([1.0, float("nan"), 0.5, float("nan")])
    assert (s["n"], s["n_nan"]) == (4, 2)
    assert s["mean"] == pytest.approx(0.75)
    p5, p50, p95 = np.percentile([0.5, 1.0], [5, 50, 95])
    assert (s["p5"], s["p50"], s["p95"]) == (pytest.approx(p5), pytest.approx(p50), pytest.approx(p95))
    json.dumps(s)


@pytest.mark.parametrize("scores", [[float("nan"), float("nan")], []])
def test_summary_all_nan_or_empty(scores):
    s = coherence_summary(scores)
    assert s["n"] == len(scores) and s["n_nan"] == len(scores)
    assert s["mean"] is None and s["p5"] is None and s["p50"] is None and s["p95"] is None


# ---------------------------------------------------------------- random_chunk_baseline
def topic_corpus():
    """Four single-topic documents (both sentences 3 words, so any cross-document pair
    scores exactly 1/sqrt(2)), plus one document with no sentences (never eligible)."""
    texts = {t: f"{t.capitalize()} {t} {t}. {t.capitalize()} {t} {t}." for t in ["cat", "dog", "fish", "bird"]}
    docs = dict(texts)
    spans = {t: spans_of(d, [f"{t.capitalize()} {t} {t}.", f"{t.capitalize()} {t} {t}."]) for t, d in texts.items()}
    chunks = [mk(t, 0, d, 0, len(d)) for t, d in texts.items()]          # each: 2 pieces, one topic
    docs["empty"] = ""
    spans["empty"] = []
    return chunks, docs, spans


def test_baseline_is_reproducible():
    chunks, docs, spans = topic_corpus()
    a = random_chunk_baseline(chunks, docs, spans, StubEmbedder(), n=100, seed=7)
    b = random_chunk_baseline(chunks, docs, spans, StubEmbedder(), n=100, seed=7)
    assert a == b


def test_baseline_uses_different_documents_and_is_below_real():
    chunks, docs, spans = topic_corpus()
    base = random_chunk_baseline(chunks, docs, spans, StubEmbedder(), n=300, seed=0)
    real = coherence_summary(coherence_scores(chunks, docs, spans, StubEmbedder()))
    assert base["n"] == 300
    # two sentences from two DIFFERENT topic documents → exactly 1/sqrt(2) every time
    assert base["mean"] == pytest.approx(np.sqrt(0.5))
    assert real["mean"] == pytest.approx(1.0)


def test_baseline_skips_draws_needing_more_documents_than_exist():
    doc = "Cat cat. Cat cat cat. Cat cat cat cat."
    docs = {"only": doc, "other": "Dog dog. Dog dog dog."}
    spans = {"only": spans_of(doc, ["Cat cat.", "Cat cat cat.", "Cat cat cat cat."]),
             "other": spans_of(docs["other"], ["Dog dog.", "Dog dog dog."])}
    chunks = [mk("only", 0, doc, 0, len(doc))]                # 3 pieces, but only 2 documents exist
    s = random_chunk_baseline(chunks, docs, spans, StubEmbedder(), n=50, seed=0)
    assert s["n"] == 0 and s["mean"] is None


# ---------------------------------------------------------------- analyze
def test_analyze_shape_without_embedder():
    out = analyze([mk("d", 0, DOC, 0, 25), mk("d", 1, DOC, 26, 41)], {"d": DOC}, {"d": SPANS},
                  budget=64, max_seq_length=256, floor=2)
    assert set(out) == {"size", "single_chunk_doc_rate", "boundary", "overlap", "coherence"}
    assert out["coherence"] is None
    assert out["size"]["n"] == 2
    json.dumps(out)


def test_analyze_with_embedder():
    out = analyze([mk("d", 0, DOC, 0, 25), mk("d", 1, DOC, 26, 41)], {"d": DOC}, {"d": SPANS},
                  budget=64, max_seq_length=256, floor=2, embed_fn=StubEmbedder())
    assert out["coherence"]["n"] == 2 and out["coherence"]["n_nan"] == 1
    assert out["coherence"]["mean"] == pytest.approx(np.sqrt(0.5))
    json.dumps(out)