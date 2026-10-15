import json

import numpy as np
import pytest

from analysis.analyzer import boundary_stats, by_doc, overlap_stats, single_chunk_doc_rate, size_stats
from chunking.base import Chunk


def mk(doc_id, idx, doc, start, end, n_tokens=None):
    text = doc[start:end]
    return Chunk(doc_id, idx, start, end, text, n_tokens if n_tokens is not None else len(text.split()))


def spans_of(doc, sentences):
    out = []
    for s in sentences:
        i = doc.index(s)
        out.append((i, i + len(s)))
    return out


# ---------------------------------------------------------------- by_doc
def test_by_doc_groups_and_sorts():
    c = [Chunk("b", 1, 0, 1, "x", 1), Chunk("a", 0, 0, 1, "x", 1), Chunk("b", 0, 0, 1, "x", 1)]
    g = by_doc(c)
    assert set(g) == {"a", "b"}
    assert [x.chunk_index for x in g["b"]] == [0, 1]


# ---------------------------------------------------------------- size_stats
SIZES = [10, 20, 30, 40, 100]


def size_chunks():
    return [Chunk(f"d{i}", 0, 0, 1, "x", n) for i, n in enumerate(SIZES)]


def test_size_stats_values():
    s = size_stats(size_chunks(), budget=100, max_seq_length=101, floor=25)
    assert s["n"] == 5 and s["min"] == 10 and s["max"] == 100
    assert s["mean"] == pytest.approx(40.0)
    assert (s["p5"], s["p50"], s["p95"]) == (pytest.approx(12.0), pytest.approx(30.0), pytest.approx(88.0))
    assert s["cv"] == pytest.approx(np.sqrt(1000) / 40)          # population std 31.62 / mean 40
    assert s["fill_mean"] == pytest.approx(0.4)
    assert s["above_ceiling_rate"] == pytest.approx(0.2)         # 100 + 2 = 102 > 101
    assert s["below_floor_rate"] == pytest.approx(0.4)           # 10 and 20 < 25


def test_size_stats_exactly_at_ceiling_is_not_above():
    s = size_stats(size_chunks(), budget=100, max_seq_length=102, floor=0)
    assert s["above_ceiling_rate"] == 0.0                         # 100 + 2 = 102 is NOT > 102


def test_size_stats_is_json_serializable():
    s = size_stats(size_chunks(), budget=100, max_seq_length=101, floor=25)
    json.dumps(s)                                                 # numpy number types would raise here
    assert all(isinstance(v, (int, float)) for v in s.values())


def test_size_stats_empty():
    assert size_stats([], budget=100, max_seq_length=256, floor=10) == {"n": 0}


# ---------------------------------------------------------------- single_chunk_doc_rate
def test_single_chunk_doc_rate():
    c = [Chunk("a", 0, 0, 1, "x", 1),
         Chunk("b", 0, 0, 1, "x", 1), Chunk("b", 1, 1, 2, "x", 1), Chunk("b", 2, 2, 3, "x", 1),
         Chunk("c", 0, 0, 1, "x", 1)]
    assert single_chunk_doc_rate(c) == pytest.approx(2 / 3)
    assert single_chunk_doc_rate([]) is None


# ---------------------------------------------------------------- boundary_stats
D1_SENTS = ["Alpha one two.", "Beta three four.", "Gamma five six.", "Delta seven eight."]
D1 = " ".join(D1_SENTS)

D2_SENTS = ["The drug lowered mortality.", "It reduced strokes too."]
D2 = " ".join(D2_SENTS)

D3 = "Only one chunk here."


def boundary_fixture():
    s1 = spans_of(D1, D1_SENTS)
    s2 = spans_of(D2, D2_SENTS)
    s3 = spans_of(D3, [D3])
    chunks = [
        # D1: clean cuts at sentence boundaries, no overlap
        mk("d1", 0, D1, s1[0][0], s1[1][1]),
        mk("d1", 1, D1, s1[2][0], s1[2][1]),
        mk("d1", 2, D1, s1[3][0], s1[3][1]),
        # D2: a mid-word cut ("lowe|red"), then a dangling start ("It ...")
        mk("d2", 0, D2, 0, 13),                       # "The drug lowe"
        mk("d2", 1, D2, 13, s2[0][1]),                # "red mortality."
        mk("d2", 2, D2, s2[1][0], s2[1][1]),          # "It reduced strokes too."
        # D3: a single chunk contributes no cuts
        mk("d3", 0, D3, 0, len(D3)),
    ]
    docs = {"d1": D1, "d2": D2, "d3": D3, "unused": "A document that produced no chunks."}
    spans = {"d1": s1, "d2": s2, "d3": s3, "unused": [(0, 35)]}
    return chunks, docs, spans


def test_precondition_mid_word_fixture():
    assert D2[0:13] == "The drug lowe" and D2[13] == "r"


def test_boundary_stats_internal_cuts_only():
    b = boundary_stats(*boundary_fixture())
    assert b["n_internal_cuts"] == 4                  # d1: 2, d2: 2, d3: 0, unused: 0
    assert b["sentence_aligned_end_rate"] == pytest.approx(3 / 4)
    assert b["sentence_aligned_start_rate"] == pytest.approx(3 / 4)
    assert b["mid_word_rate"] == pytest.approx(1 / 4)
    assert b["dangling_start_rate"] == pytest.approx(1 / 4)


def test_dangling_start_after_leading_punctuation():
    doc = 'Results were clear. "However, the effect faded."'
    cut = doc.index('"However')
    chunks = [mk("d", 0, doc, 0, cut - 1), mk("d", 1, doc, cut, len(doc))]
    b = boundary_stats(chunks, {"d": doc}, {"d": [(0, cut - 1), (cut, len(doc))]})
    assert b["dangling_start_rate"] == 1.0


def test_boundary_stats_no_cuts_gives_none():
    doc = "Just one sentence."
    b = boundary_stats([mk("d", 0, doc, 0, len(doc))], {"d": doc}, {"d": [(0, len(doc))]})
    assert b["n_internal_cuts"] == 0
    for k in ["sentence_aligned_end_rate", "sentence_aligned_start_rate", "mid_word_rate", "dangling_start_rate"]:
        assert b[k] is None


# ---------------------------------------------------------------- overlap_stats
X = "x" * 300


def protected_doc():
    """Chunks [0,100) [80,180) [200,260); sentences [0,90) [90,120) [120,200) [200,260).
    Cut at 100: sentence [90,120) straddles and is inside chunk [80,180) → protected.
    Cut at 180: sentence [120,200) straddles; not inside [200,260) → NOT protected."""
    chunks = [mk("p", 0, X, 0, 100), mk("p", 1, X, 80, 180), mk("p", 2, X, 200, 260)]
    spans = [(0, 90), (90, 120), (120, 200), (200, 260)]
    return chunks, spans


def unprotected_doc():
    """Chunks [0,100) [80,180); sentences [0,60) [60,120) [120,180).
    Cut at 100: sentence [60,120) straddles; starts before 80 → NOT protected."""
    chunks = [mk("u", 0, X, 0, 100), mk("u", 1, X, 80, 180)]
    spans = [(0, 60), (60, 120), (120, 180)]
    return chunks, spans


def test_overlap_protected_doc():
    chunks, spans = protected_doc()
    o = overlap_stats(chunks, {"p": X}, {"p": spans})
    assert o["mean_overlap_chars"] == pytest.approx(10.0)       # pairs: 100-80 = 20, max(0, 180-200) = 0
    assert o["max_overlap_chars"] == 20
    assert o["redundancy_ratio"] == pytest.approx(260 / 240)    # chunk chars 260, union of chunk spans 240
    assert o["n_straddling"] == 2
    assert o["straddle_protection_rate"] == pytest.approx(0.5)


def test_overlap_across_two_docs():
    pc, ps = protected_doc()
    uc, us = unprotected_doc()
    o = overlap_stats(pc + uc, {"p": X, "u": X}, {"p": ps, "u": us})
    assert o["mean_overlap_chars"] == pytest.approx(40 / 3)     # [20, 0] + [20]
    assert o["redundancy_ratio"] == pytest.approx(460 / 420)    # (260 + 200) / (240 + 180)
    assert o["n_straddling"] == 3
    assert o["straddle_protection_rate"] == pytest.approx(1 / 3)


def test_no_overlap_redundancy_is_exactly_one():
    chunks, docs, spans = boundary_fixture()
    o = overlap_stats(chunks, docs, spans)
    assert o["redundancy_ratio"] == pytest.approx(1.0)
    assert o["max_overlap_chars"] == 0


def test_nothing_straddles_gives_none():
    chunks = [mk("d", 0, X, 0, 50), mk("d", 1, X, 50, 100)]
    o = overlap_stats(chunks, {"d": X}, {"d": [(0, 50), (50, 100)]})
    assert o["n_straddling"] == 0
    assert o["straddle_protection_rate"] is None


def test_all_single_chunk_docs_give_none_for_overlap():
    chunks = [mk("a", 0, X, 0, 50), mk("b", 0, X, 0, 70)]
    o = overlap_stats(chunks, {"a": X, "b": X}, {"a": [(0, 50)], "b": [(0, 70)]})
    assert o["mean_overlap_chars"] is None and o["max_overlap_chars"] is None
    assert o["redundancy_ratio"] == pytest.approx(1.0)


def test_overlap_stats_is_json_serializable():
    chunks, spans = protected_doc()
    json.dumps(overlap_stats(chunks, {"p": X}, {"p": spans}))


def test_analyzer_does_not_modify_chunks():
    chunks, docs, spans = boundary_fixture()
    before = [(c.start, c.end, c.text, dict(c.metadata)) for c in chunks]
    boundary_stats(chunks, docs, spans)
    overlap_stats(chunks, docs, spans)
    size_stats(chunks, budget=64, max_seq_length=256, floor=4)
    assert [(c.start, c.end, c.text, dict(c.metadata)) for c in chunks] == before