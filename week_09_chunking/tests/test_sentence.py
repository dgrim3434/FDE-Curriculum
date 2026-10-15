import re
import threading

import pytest

from chunking.base import Chunker, trim_span
from chunking.sentence import SentenceChunker
from tests.invariants import assert_shared_invariants


# ---------------------------------------------------------------- helpers
def sent(n):
    """A sentence of exactly n tokens (n >= 2): (n-1) words 'cat' + a final '.'"""
    return " ".join(["cat"] * (n - 1)) + "."


def doc_of(*lengths):
    """A document made of sentences with the given token counts, joined by spaces."""
    return " ".join(sent(n) for n in lengths)


def stub_splitter(text):
    """Stand-in for spaCy: a sentence ends at each '.'; spans include the leading space."""
    return [(m.start(), m.end()) for m in re.finditer(r"[^.]+(?:\.|$)", text)]


def sentence_spans(text):
    """Trimmed, non-empty sentence spans: what the chunker should be working with."""
    spans = [trim_span(text, s, e) for s, e in stub_splitter(text)]
    return [(s, e) for s, e in spans if s < e]


USE_TIMEOUT = False      # set to False while debugging; set back to True for the final run


def run(chunker, doc, doc_id="d1", seconds=10):
    """Call chunker.chunk with a time limit, so an infinite loop fails instead of hanging.

    With USE_TIMEOUT = False it calls chunk() directly on the main thread,
    so breakpoints and stepping work (but a real infinite loop will hang: Ctrl+C to stop).
    """
    if not USE_TIMEOUT:
        return chunker.chunk(doc, doc_id)

    box = {}

    def target():
        try:
            box["out"] = chunker.chunk(doc, doc_id)
        except BaseException as e:          # re-raised in the test thread below
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "chunk() did not return within the time limit (infinite loop?)"
    if "err" in box:
        raise box["err"]
    return box["out"]


REAL_DOC = (
    "Background: Statin therapy reduces LDL-cholesterol and cardiovascular events. "
    "Methods: In a randomized trial, 4,162 patients received atorvastatin 80 mg or "
    "pravastatin 40 mg daily.\r\nResults: LDL-C fell by 51% vs 22% (p < 0.001); the hazard "
    "ratio was 0.84 (95% CI, 0.74–0.95).\n\nConclusions: Intensive lipid lowering provided "
    "greater protection — especially after acute coronary syndromes. "
) * 3


# ---------------------------------------------------------------- preconditions
def test_precondition_sentence_token_counts(tok):
    for n in [2, 40, 120, 300]:
        assert len(tok(sent(n), add_special_tokens=False)["input_ids"]) == n


def test_precondition_stub_splitter():
    doc = doc_of(5, 7, 3)
    assert [doc[s:e] for s, e in sentence_spans(doc)] == [sent(5), sent(7), sent(3)]


# ---------------------------------------------------------------- interface and config
def test_is_a_chunker(tok):
    assert isinstance(SentenceChunker(tok, stub_splitter, budget=64), Chunker)


@pytest.mark.parametrize("budget,overlap", [(0, 0), (-10, 0), (64, -1)])
def test_invalid_config_raises(tok, budget, overlap):
    with pytest.raises(ValueError):
        SentenceChunker(tok, stub_splitter, budget=budget, overlap_sentences=overlap)


def test_empty_and_whitespace_docs_return_empty_list(tok):
    ch = SentenceChunker(tok, stub_splitter, budget=64)
    assert run(ch, "") == []
    assert run(ch, "   \n\t ") == []


def test_single_sentence_is_one_trimmed_chunk(tok):
    doc = "   " + sent(10) + "   "
    out = run(SentenceChunker(tok, stub_splitter, budget=64), doc)
    assert len(out) == 1
    assert out[0].text == sent(10)
    assert out[0].n_tokens == 10


# ---------------------------------------------------------------- exact packing behaviour
def test_trace_from_section_3(tok):
    """Your Section 3 re-ask: A=120, B=60, C=150, D=300, E=40; budget 200; overlap 1."""
    doc = doc_of(120, 60, 150, 300, 40)
    out = run(SentenceChunker(tok, stub_splitter, budget=200, overlap_sentences=1), doc)
    A, B, C, D, E = sentence_spans(doc)

    assert [c.n_tokens for c in out] == [180, 150, 200, 100, 40]
    assert (out[0].start, out[0].end) == (A[0], B[1])       # [A, B]
    assert (out[1].start, out[1].end) == C                   # overlap B dropped: B + C > 200
    assert out[2].start == D[0] and out[3].end == D[1]       # fallback pieces cover D exactly
    assert (out[4].start, out[4].end) == E                   # no overlap carried after a fallback


def test_no_overlap_packing(tok):
    doc = doc_of(50, 50, 50, 50, 50)
    out = run(SentenceChunker(tok, stub_splitter, budget=120, overlap_sentences=0), doc)
    assert [c.n_tokens for c in out] == [100, 100, 50]
    assert_shared_invariants(doc, out, tok, allow_overlap=False)


def test_one_sentence_overlap_repeats_last_sentence(tok):
    doc = doc_of(50, 50, 50, 50, 50)
    out = run(SentenceChunker(tok, stub_splitter, budget=120, overlap_sentences=1), doc)
    spans = sentence_spans(doc)
    assert [c.n_tokens for c in out] == [100, 100, 100, 100]
    for k, (prev, nxt) in enumerate(zip(out, out[1:])):
        # the next chunk starts exactly at the previous chunk's last sentence
        assert nxt.start == spans[k + 1][0]
        assert prev.end == spans[k + 1][1]


def test_overlap_drops_oldest_sentences_until_it_fits(tok):
    """Overlap 2 would give 80+80+80 = 240 > 170, so the oldest overlap sentence is dropped."""
    doc = doc_of(80, 80, 80)
    out = run(SentenceChunker(tok, stub_splitter, budget=170, overlap_sentences=2), doc)
    s0, s1, s2 = sentence_spans(doc)
    assert [c.n_tokens for c in out] == [160, 160]
    assert (out[0].start, out[0].end) == (s0[0], s1[1])
    assert (out[1].start, out[1].end) == (s1[0], s2[1])


@pytest.mark.parametrize("lengths,budget,overlap,expected", [
    ((150, 100), 200, 1, [150, 100]),             # the C / D2 case: must terminate
    ((100, 100, 100), 150, 3, [100, 100, 100]),   # overlap larger than the chunk
    ((190, 190, 190), 200, 1, [190, 190, 190]),
])
def test_overlap_always_makes_progress(tok, lengths, budget, overlap, expected):
    doc = doc_of(*lengths)
    out = run(SentenceChunker(tok, stub_splitter, budget=budget, overlap_sentences=overlap), doc)
    assert [c.n_tokens for c in out] == expected
    starts = [c.start for c in out]
    assert all(a < b for a, b in zip(starts, starts[1:]))


# ---------------------------------------------------------------- fallback
def test_fallback_pieces_point_into_original_document(tok):
    doc = doc_of(20, 300, 20)
    out = run(SentenceChunker(tok, stub_splitter, budget=100), doc)
    _, D, _ = sentence_spans(doc)
    pieces = [c for c in out if c.start >= D[0] and c.end <= D[1]]
    assert [c.n_tokens for c in pieces] == [100, 100, 100]
    assert pieces[0].start == D[0] and pieces[-1].end == D[1]
    assert_shared_invariants(doc, out, tok, allow_overlap=False)


def test_every_chunk_within_budget(tok):
    doc = doc_of(30, 45, 12, 80, 25, 60, 300, 20, 99, 101)
    for overlap in [0, 1, 2]:
        out = run(SentenceChunker(tok, stub_splitter, budget=100, overlap_sentences=overlap), doc)
        assert all(c.n_tokens <= 100 for c in out), overlap


# ---------------------------------------------------------------- boundaries
@pytest.mark.parametrize("overlap", [0, 1])
def test_boundaries_fall_on_sentence_boundaries(tok, overlap):
    doc = doc_of(30, 45, 12, 80, 25, 60, 300, 20)
    out = run(SentenceChunker(tok, stub_splitter, budget=100, overlap_sentences=overlap), doc)
    spans = sentence_spans(doc)
    starts = {s for s, _ in spans}
    ends = {e for _, e in spans}
    oversized = [(s, e) for s, e in spans
                 if len(tok(doc[s:e], add_special_tokens=False)["input_ids"]) > 100]
    for c in out:
        inside_fallback = any(s <= c.start and c.end <= e for s, e in oversized)
        if not inside_fallback:
            assert c.start in starts and c.end in ends, \
                f"chunk {c.chunk_index} [{c.start}:{c.end}] cuts inside a sentence"


# ---------------------------------------------------------------- shared invariants on realistic text
@pytest.mark.parametrize("budget", [16, 32, 64, 128])
@pytest.mark.parametrize("overlap", [0, 1, 2])
def test_shared_invariants_real_text(tok, budget, overlap):
    out = run(SentenceChunker(tok, stub_splitter, budget=budget, overlap_sentences=overlap),
              REAL_DOC, doc_id="doc-real")
    assert len(out) >= 2
    assert_shared_invariants(REAL_DOC, out, tok, doc_id="doc-real", allow_overlap=overlap > 0)


def test_deterministic(tok):
    ch = SentenceChunker(tok, stub_splitter, budget=32, overlap_sentences=1)
    assert run(ch, REAL_DOC) == run(ch, REAL_DOC)