import re

import numpy as np
import pytest

from chunking.base import Chunker
from chunking.semantic import SemanticChunker, cut_after, gap_similarities, percentile_threshold
from tests.helpers import run, stub_splitter
from tests.invariants import assert_shared_invariants


# ---------------------------------------------------------------- stub embedder
class StubEmbedder:
    """Stand-in for a sentence-transformers model: no weights, no network.
    Each text becomes a normalized vector of topic-word counts, so topic shifts are known exactly."""
    VOCAB = ["cat", "dog", "fish", "bird"]

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        rows = []
        for t in texts:
            words = re.findall(r"\w+", t.lower())
            v = np.array([words.count(w) for w in self.VOCAB]
                         + [sum(w not in self.VOCAB for w in words)], dtype=float)
            if not v.any():
                v[-1] = 1.0
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)


def tsent(word, n):
    """A sentence of exactly n tokens (n >= 2) about one topic: (n-1) copies of `word` + '.'"""
    return " ".join([word] * (n - 1)) + "."


def tdoc(*parts):
    """parts: (word, n_tokens) pairs → a document of topic sentences."""
    return " ".join(tsent(w, n) for w, n in parts)


def make(tok, emb=None, budget=200, threshold=0.5, window=1, min_tokens=0):
    return SemanticChunker(tok, stub_splitter, emb or StubEmbedder(), budget=budget,
                           threshold=threshold, window=window, min_tokens=min_tokens)


# ---------------------------------------------------------------- gap_similarities
def test_gap_similarities_neighbors():
    E = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(gap_similarities(E, window=1), [1.0, 0.0])


def test_gap_similarities_window_2_by_hand():
    E = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    # gap0: [1,0] vs mean(e1,e2)=[.5,.5]→norm → 0.7071
    # gap1: mean(e0,e1)=[1,0] vs mean(e2,e3)=[0,1] → 0
    # gap2: mean(e1,e2)→norm [.7071,.7071] vs e3=[0,1] → 0.7071
    assert np.allclose(gap_similarities(E, window=2), [0.70710678, 0.0, 0.70710678])


@pytest.mark.parametrize("window", [1, 2, 3, 5])
def test_gap_similarities_always_n_minus_1(window):
    rng = np.random.default_rng(0)
    E = rng.normal(size=(9, 6))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    assert gap_similarities(E, window=window).shape == (8,)


def test_gap_similarities_window_1_equals_rowwise_dot():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(12, 8))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    assert np.allclose(gap_similarities(E, window=1), np.sum(E[:-1] * E[1:], axis=1))


def test_gap_similarities_single_row_is_empty():
    assert gap_similarities(np.array([[1.0, 0.0]]), window=1).shape == (0,)


# ---------------------------------------------------------------- percentile_threshold + cut_after
SIMS = np.array([0.62, 0.58, 0.21, 0.66, 0.60, 0.18, 0.55])      # the Section 5 worked example


def test_percentile_threshold_worked_example():
    t = percentile_threshold(1 - SIMS, 75)
    assert isinstance(t, float)
    assert t == pytest.approx(0.62)


def test_cut_after_worked_example():
    d = 1 - SIMS
    cuts = cut_after(d, percentile_threshold(d, 75))
    assert cuts == [2, 5] and isinstance(cuts, list)


def test_percentile_rule_is_scale_invariant_and_absolute_rule_is_not():
    """Your Section 5 checkpoint in code: a second model's scale = 0.6*s + 0.38."""
    d1 = 1 - SIMS
    d2 = 1 - (0.6 * SIMS + 0.38)
    assert cut_after(d2, percentile_threshold(d2, 75)) == cut_after(d1, percentile_threshold(d1, 75))
    assert cut_after(d1, 0.5) != cut_after(d2, 0.5)          # negative control: [2, 5] vs [5]


# ---------------------------------------------------------------- interface and config
def test_is_a_chunker(tok):
    assert isinstance(make(tok), Chunker)


@pytest.mark.parametrize("kwargs", [
    {"budget": 0}, {"budget": -3}, {"window": 0}, {"min_tokens": -1}, {"min_tokens": 201},
])
def test_invalid_config_raises(tok, kwargs):
    with pytest.raises(ValueError):
        make(tok, **kwargs)


def test_chunk_without_threshold_raises(tok):
    ch = make(tok, threshold=None)
    with pytest.raises(ValueError):
        ch.chunk(tdoc(("cat", 10), ("dog", 10)), "d1")


# ---------------------------------------------------------------- embedding calls
def test_empty_doc_returns_empty_and_never_embeds(tok):
    emb = StubEmbedder()
    assert run(make(tok, emb), "") == []
    assert run(make(tok, emb), "   \n ") == []
    assert emb.calls == 0


def test_single_sentence_is_one_chunk_without_embedding(tok):
    emb = StubEmbedder()
    out = run(make(tok, emb), "  " + tsent("cat", 12) + "  ")
    assert len(out) == 1 and out[0].n_tokens == 12
    assert emb.calls == 0


def test_embeds_once_per_document(tok):
    emb = StubEmbedder()
    run(make(tok, emb), tdoc(("cat", 10), ("cat", 10), ("dog", 10), ("dog", 10)))
    assert emb.calls == 1


# ---------------------------------------------------------------- cuts at topic changes
def test_one_topic_change_one_cut(tok):
    doc = tdoc(("cat", 10), ("cat", 10), ("cat", 10), ("dog", 10), ("dog", 10), ("dog", 10))
    out = run(make(tok), doc)
    assert [c.n_tokens for c in out] == [30, 30]
    assert out[0].end == len(tdoc(("cat", 10), ("cat", 10), ("cat", 10)))


def test_three_topics_two_cuts(tok):
    doc = tdoc(("cat", 10), ("cat", 10), ("dog", 10), ("dog", 10), ("fish", 10), ("fish", 10))
    assert [c.n_tokens for c in run(make(tok), doc)] == [20, 20, 20]


def test_no_cuts_keeps_whole_document(tok):
    doc = tdoc(*[("cat", 10)] * 5)
    out = run(make(tok), doc)
    assert [c.n_tokens for c in out] == [50]
    assert out[0].start == 0 and out[0].end == len(doc)


def test_window_2_still_cuts_only_at_the_topic_change(tok):
    doc = tdoc(("cat", 10), ("cat", 10), ("cat", 10), ("dog", 10), ("dog", 10), ("dog", 10))
    assert [c.n_tokens for c in run(make(tok, window=2), doc)] == [30, 30]


# ---------------------------------------------------------------- size enforcement
def test_oversized_segment_is_packed(tok):
    doc = tdoc(*[("cat", 30)] * 8)                           # one topic, 240 tokens, budget 100
    out = run(make(tok, budget=100), doc)
    assert [c.n_tokens for c in out] == [90, 90, 60]
    assert out[0].start == 0 and out[-1].end == len(doc)
    assert_shared_invariants(doc, out, tok, allow_overlap=False)


def test_undersized_merges_into_more_similar_left_neighbor(tok):
    """'cat cat dog.' is closer to the cat block (distance 0.106) than the dog block (0.553)."""
    left = tdoc(("cat", 10), ("cat", 10), ("cat", 10))
    doc = left + " cat cat dog. " + tdoc(("dog", 10), ("dog", 10), ("dog", 10))
    out = run(make(tok, threshold=0.05, min_tokens=10), doc)
    assert [c.n_tokens for c in out] == [34, 30]
    assert out[0].end == len(left + " cat cat dog.")


def test_undersized_merges_into_more_similar_right_neighbor(tok):
    """Mirror image: 'dog dog cat.' is closer to the dog block, so it merges right."""
    left = tdoc(("cat", 10), ("cat", 10), ("cat", 10))
    doc = left + " dog dog cat. " + tdoc(("dog", 10), ("dog", 10), ("dog", 10))
    out = run(make(tok, threshold=0.05, min_tokens=10), doc)
    assert [c.n_tokens for c in out] == [30, 34]
    assert out[1].start == len(left) + 1


def test_undersized_kept_when_no_merge_fits(tok):
    doc = (tdoc(("cat", 10), ("cat", 10), ("cat", 10)) + " cat cat dog. "
           + tdoc(("dog", 10), ("dog", 10), ("dog", 10)))
    out = run(make(tok, budget=30, threshold=0.05, min_tokens=10), doc)
    assert [c.n_tokens for c in out] == [30, 4, 30]


def test_undersized_at_document_start_merges_right(tok):
    doc = "cat cat dog. " + tdoc(("dog", 10), ("dog", 10), ("dog", 10))
    out = run(make(tok, threshold=0.05, min_tokens=10), doc)
    assert [c.n_tokens for c in out] == [34]


# ---------------------------------------------------------------- two-pass API
def test_chunk_equals_prepare_then_chunk_prepared(tok):
    doc = tdoc(("cat", 12), ("cat", 9), ("dog", 20), ("fish", 15), ("fish", 7))
    ch = make(tok)
    sentences, distances = ch.prepare(doc)
    assert len(sentences) == 5 and distances.shape == (4,)
    assert ch.chunk_prepared(doc, "d1", sentences, distances) == run(ch, doc)


def test_corpus_threshold_two_pass(tok):
    """Pass 1 collects distances from every doc; pass 2 chunks with ONE corpus threshold."""
    docs = [tdoc(("cat", 10), ("cat", 10), ("dog", 10)),
            tdoc(("fish", 10), ("fish", 10), ("fish", 10), ("bird", 10))]
    ch = make(tok, threshold=None)
    prepared = [ch.prepare(d) for d in docs]
    all_d = np.concatenate([d for _, d in prepared])
    ch.threshold = percentile_threshold(all_d, 50)
    outs = [ch.chunk_prepared(doc, f"d{i}", s, d) for i, (doc, (s, d)) in enumerate(zip(docs, prepared))]
    assert [[c.n_tokens for c in o] for o in outs] == [[20, 10], [30, 10]]


# ---------------------------------------------------------------- shared invariants on a mixed document
MIXED = tdoc(("cat", 12), ("cat", 30), ("dog", 8), ("dog", 25), ("fish", 40), ("cat", 15),
             ("bird", 60), ("bird", 9), ("dog", 120))


@pytest.mark.parametrize("window", [1, 2])
@pytest.mark.parametrize("min_tokens", [0, 10])
@pytest.mark.parametrize("budget", [32, 64, 200])
def test_shared_invariants_mixed_doc(tok, window, min_tokens, budget):
    out = run(make(tok, budget=budget, window=window, min_tokens=min_tokens), MIXED, "doc-mixed")
    assert_shared_invariants(MIXED, out, tok, doc_id="doc-mixed", allow_overlap=False)
    assert all(c.n_tokens <= budget for c in out)


def test_deterministic(tok):
    ch = make(tok, window=2, min_tokens=10)
    assert run(ch, MIXED) == run(ch, MIXED)