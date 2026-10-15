import math

import pytest

from chunking.base import Chunker
from chunking.fixed import FixedSizeChunker
from tests.invariants import assert_shared_invariants


def cats(n):
    """n words that are each exactly one token, so token counts are known exactly."""
    return " ".join(["cat"] * n)


REAL_DOC = (
    "Background: Statin therapy reduces LDL-cholesterol (LDL-C) and cardiovascular events.\r\n"
    "Methods: In a randomized, double-blind trial, 4,162 patients (mean age 61.3 ± 9.8 years) "
    "received atorvastatin 80 mg or pravastatin 40 mg daily.\n\n"
    "Results: After 24 months, LDL-C fell by 51% vs 22% (p < 0.001); the hazard ratio for death "
    "or myocardial infarction was 0.84 (95% CI, 0.74–0.95). Pembrolizumab-naïve subgroups showed "
    "similar effects.\nConclusions: Intensive lipid lowering provided greater protection — "
    "especially in patients with acute coronary syndromes. "
) * 3


# ---------------------------------------------------------------- interface and config
def test_is_a_chunker(tok):
    assert isinstance(FixedSizeChunker(tok, 64), Chunker)


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (64, 64), (64, 100), (64, -1)])
def test_invalid_config_raises(tok, size, overlap):
    with pytest.raises(ValueError):
        FixedSizeChunker(tok, size, overlap)


def test_returns_a_list(tok):
    assert isinstance(FixedSizeChunker(tok, 64).chunk(cats(10), "d1"), list)


def test_empty_and_whitespace_docs_return_empty_list(tok):
    ch = FixedSizeChunker(tok, 64)
    assert ch.chunk("", "d1") == []
    assert ch.chunk("   \n\t  ", "d1") == []


def test_short_doc_is_one_trimmed_chunk(tok):
    out = FixedSizeChunker(tok, 64).chunk("  Statins lowered LDL.  ", "d1")
    assert len(out) == 1
    assert out[0].text == "Statins lowered LDL."
    assert out[0].start == 2


# ---------------------------------------------------------------- exact window arithmetic
def test_precondition_cat_is_one_token(tok):
    assert len(tok(cats(600), add_special_tokens=False)["input_ids"]) == 600


def test_600_tokens_no_overlap_keeps_short_tail(tok):
    out = FixedSizeChunker(tok, 256, 0).chunk(cats(600), "d1")
    assert [c.n_tokens for c in out] == [256, 256, 88]


def test_600_tokens_with_overlap(tok):
    out = FixedSizeChunker(tok, 256, 56).chunk(cats(600), "d1")
    assert [c.n_tokens for c in out] == [256, 256, 200]


def test_realized_overlap_equals_configured(tok):
    doc = cats(600)
    out = FixedSizeChunker(tok, 256, 56).chunk(doc, "d1")
    for prev, nxt in zip(out, out[1:]):
        shared = doc[nxt.start:prev.end]
        assert len(tok(shared, add_special_tokens=False)["input_ids"]) == 56


@pytest.mark.parametrize("n,size,overlap,expected", [
    (512, 256, 0, 2),     # exact fit: no third window
    (456, 256, 56, 2),    # 2nd window reaches n: stop (no all-overlap 3rd window)
    (457, 256, 56, 3),
    (256, 256, 0, 1),
    (10, 256, 0, 1),
])
def test_window_count_stops_at_end(tok, n, size, overlap, expected):
    assert len(FixedSizeChunker(tok, size, overlap).chunk(cats(n), "d1")) == expected


def test_window_count_formula(tok):
    for n in [1, 50, 255, 256, 257, 600]:
        for size, overlap in [(64, 0), (64, 16), (100, 99), (256, 0)]:
            stride = size - overlap
            expected = 1 if n <= size else math.ceil((n - size) / stride) + 1
            out = FixedSizeChunker(tok, size, overlap).chunk(cats(n), "d1")
            assert len(out) == expected, (n, size, overlap)


def test_first_and_last_chunks_reach_document_edges(tok):
    doc = "   " + cats(300) + "  \n"
    out = FixedSizeChunker(tok, 64, 16).chunk(doc, "d1")
    assert out[0].start == 3
    assert out[-1].end == len(doc.rstrip())


# ---------------------------------------------------------------- shared invariants on realistic text
@pytest.mark.parametrize("size,overlap", [(16, 0), (32, 8), (64, 16), (128, 0), (254, 0)])
def test_shared_invariants_real_text(tok, size, overlap):
    out = FixedSizeChunker(tok, size, overlap).chunk(REAL_DOC, "doc-real")
    assert len(out) >= 2
    assert_shared_invariants(REAL_DOC, out, tok, doc_id="doc-real", allow_overlap=overlap > 0)
    if overlap > 0:
        assert all(n.start < p.end for p, n in zip(out, out[1:])), "configured overlap not realized"


def test_original_casing_and_accents_preserved(tok):
    doc = "Hello WORLD — Café naïve résumé."
    out = FixedSizeChunker(tok, 64).chunk(doc, "d1")
    assert len(out) == 1 and out[0].text == doc


def test_deterministic(tok):
    ch = FixedSizeChunker(tok, 32, 8)
    assert ch.chunk(REAL_DOC, "d") == ch.chunk(REAL_DOC, "d")