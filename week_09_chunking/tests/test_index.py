import numpy as np
import pytest

from chunking.base import Chunk
from retrieval.index import (doc_layout, embed_with_cache, maxp, rank_chunks, rank_documents,
                             texts_for_embedding)


def chunk(doc_id, idx, start=0, text="x", meta=None):
    return Chunk(doc_id, idx, start, start + len(text), text, 1, dict(meta or {}))


# chunks a a a b c c c c   (start values are character offsets, NOT list positions)
LAYOUT = [chunk("a", 0, 0), chunk("a", 1, 412), chunk("a", 2, 830),
          chunk("b", 0, 0),
          chunk("c", 0, 0), chunk("c", 1, 377), chunk("c", 2, 700), chunk("c", 3, 990)]


class StubEmbed:
    """Deterministic vectors from text length; counts calls."""
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        v = np.array([[len(t), 1.0] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- texts_for_embedding
def test_texts_prefer_embed_text_and_fall_back_to_text():
    c1 = chunk("d", 0, text="Body one.", meta={"embed_text": "Title: T\nBody one."})
    c2 = chunk("d", 1, text="Body two.")                       # no embed_text key at all
    assert texts_for_embedding([c1, c2]) == ["Title: T\nBody one.", "Body two."]


# ---------------------------------------------------------------- embed_with_cache
def test_cache_embeds_once_then_reads_from_disk(tmp_path):
    chunks = [chunk("d", 0, text="short"), chunk("d", 1, text="a longer text")]
    path = tmp_path / "cache" / "cfg.npz"

    first = StubEmbed()
    v1 = embed_with_cache(chunks, first, path)
    assert first.calls == 1 and path.exists()

    second = StubEmbed()
    v2 = embed_with_cache(chunks, second, path)
    assert second.calls == 0                                     # served from the cache
    assert np.array_equal(v1, v2)


def test_cache_detects_changed_chunks(tmp_path):
    chunks = [chunk("d", 0, text="short"), chunk("d", 1, text="a longer text")]
    path = tmp_path / "cfg.npz"
    embed_with_cache(chunks, StubEmbed(), path)
    with pytest.raises(ValueError):
        embed_with_cache(list(reversed(chunks)), StubEmbed(), path)


# ---------------------------------------------------------------- doc_layout
def test_doc_layout_uses_list_positions():
    doc_ids, doc_starts = doc_layout(LAYOUT)
    assert doc_ids == ["a", "b", "c"]
    assert list(doc_starts) == [0, 3, 4]


def test_doc_layout_rejects_non_contiguous():
    with pytest.raises(ValueError):
        doc_layout([chunk("a", 0), chunk("b", 0), chunk("a", 1)])


def test_doc_layout_empty():
    doc_ids, doc_starts = doc_layout([])
    assert doc_ids == [] and len(doc_starts) == 0


# ---------------------------------------------------------------- maxp
def test_maxp_matches_naive_loop():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(4, 10))
    starts = np.array([0, 3, 4, 7])
    ends = list(starts[1:]) + [S.shape[1]]
    naive = np.stack([S[:, s:e].max(axis=1) for s, e in zip(starts, ends)], axis=1)
    assert np.allclose(maxp(S, starts), naive)


def test_maxp_section_8a_example():
    S = np.array([[0.41, 0.63, 0.22, 0.58, 0.40, 0.39, 0.37, 0.55]])   # A A A | B | C C C C
    assert np.allclose(maxp(S, np.array([0, 3, 4])), [[0.63, 0.58, 0.55]])


# ---------------------------------------------------------------- rank_documents
C_IDENTITY = np.eye(8)                                            # S = Q @ I.T = Q


def test_rank_documents_order_and_k():
    Q = np.array([[0.41, 0.63, 0.22, 0.58, 0.40, 0.39, 0.37, 0.55],   # MaxP: a .63, b .58, c .55
                  [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00, 0.00]])  # MaxP: c .90, b .10, a 0
    run = rank_documents(Q, C_IDENTITY, LAYOUT, ["q1", "q2"], k=3)
    assert run == {"q1": ["a", "b", "c"], "q2": ["c", "b", "a"]}

    run2 = rank_documents(Q, C_IDENTITY, LAYOUT, ["q1", "q2"], k=2)
    assert run2 == {"q1": ["a", "b"], "q2": ["c", "b"]}


def test_rank_documents_k_larger_than_number_of_docs():
    Q = np.array([[0.41, 0.63, 0.22, 0.58, 0.40, 0.39, 0.37, 0.55]])
    assert rank_documents(Q, C_IDENTITY, LAYOUT, ["q1"], k=10) == {"q1": ["a", "b", "c"]}


def test_rank_documents_ties_are_stable():
    Q = np.zeros((1, 8))                                          # every document scores 0
    assert rank_documents(Q, C_IDENTITY, LAYOUT, ["q1"], k=3) == {"q1": ["a", "b", "c"]}


# ---------------------------------------------------------------- rank_chunks
def test_rank_chunks_order():
    C_paper = np.eye(4)
    q_vec = np.array([0.2, 0.9, 0.1, 0.5])
    assert rank_chunks(q_vec, C_paper) == [1, 3, 0, 2]


def test_rank_chunks_ties_are_stable():
    assert rank_chunks(np.zeros(3), np.eye(3)) == [0, 1, 2]