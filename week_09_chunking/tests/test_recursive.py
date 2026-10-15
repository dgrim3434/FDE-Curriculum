import pytest

from chunking.base import Chunker, trim_span
from chunking.recursive import RecursiveChunker
from chunking.sentence import SentenceChunker
from tests.helpers import cats, para, run, sent, stub_splitter
from tests.invariants import assert_shared_invariants


def make(tok, budget=200, min_tokens=0, across=False):
    return RecursiveChunker(tok, stub_splitter, budget=budget, min_tokens=min_tokens,
                            merge_across_paragraphs=across)


def para_spans(doc):
    """Trimmed spans of the paragraphs of a doc built with '\\n\\n' between paragraphs."""
    spans, pos = [], 0
    for p in doc.split("\n\n"):
        s, e = trim_span(doc, pos, pos + len(p))
        if s < e:
            spans.append((s, e))
        pos += len(p) + 2
    return spans


REAL_DOC = "\n\n".join([(
    "Introduction\n\n"
    "Statin therapy reduces LDL-cholesterol and cardiovascular events. Its benefit in elderly "
    "patients remains uncertain.\n\n"
    "Methods\n\n"
    "In a randomized trial, 4,162 patients received atorvastatin 80 mg or pravastatin 40 mg daily.\n"
    "Follow-up lasted 24 months.\n"
    "- Primary outcome: death or myocardial infarction\n"
    "- Secondary outcome: stroke\n\n"
    "Results\n\n"
    "LDL-C fell by 51% vs 22% (p < 0.001). The hazard ratio was 0.84 (95% CI, 0.74–0.95). "
    "Effects were similar in every subgroup.\r\n\r\n"
    "Conclusions\n\n"
    "Intensive lipid lowering provided greater protection — especially after acute coronary syndromes."
)] * 2)


# ---------------------------------------------------------------- interface and config
def test_is_a_chunker(tok):
    assert isinstance(make(tok), Chunker)


@pytest.mark.parametrize("budget,min_tokens", [(0, 0), (-5, 0), (200, -1), (200, 201)])
def test_invalid_config_raises(tok, budget, min_tokens):
    with pytest.raises(ValueError):
        RecursiveChunker(tok, stub_splitter, budget=budget, min_tokens=min_tokens)


def test_empty_and_whitespace_docs_return_empty_list(tok):
    assert run(make(tok), "") == []
    assert run(make(tok), "  \n\n \t \n ") == []


def test_short_doc_is_one_trimmed_chunk(tok):
    out = run(make(tok), "\n\n  " + para(10, 12) + "  \n\n")
    assert len(out) == 1 and out[0].text == para(10, 12) and out[0].n_tokens == 22


# ---------------------------------------------------------------- your Section 4 checkpoint trace
TRACE_DOC = "\n\n".join([para(60), para(90, 110, 60), para(50)])   # P1, P2 (3 sentences), P3


def test_trace_merge_within_paragraphs(tok):
    out = run(make(tok, budget=200, across=False), TRACE_DOC)
    assert [c.n_tokens for c in out] == [60, 200, 60, 50]


def test_trace_merge_across_paragraphs(tok):
    out = run(make(tok, budget=200, across=True), TRACE_DOC)
    assert [c.n_tokens for c in out] == [150, 170, 50]


def test_trace_sub_floor_group_may_cross_a_paragraph(tok):
    """P1 = 60 < min_tokens = 64, so it merges forward even with across=False."""
    out = run(make(tok, budget=200, min_tokens=64, across=False), TRACE_DOC)
    assert [c.n_tokens for c in out] == [150, 170, 50]


# ---------------------------------------------------------------- headings
def test_heading_attaches_to_following_paragraph(tok):
    doc = "Methods\n\n" + para(40, 40) + "\n\n" + para(60)
    out = run(make(tok, budget=200, min_tokens=10, across=False), doc)
    assert [c.n_tokens for c in out] == [81, 60]
    assert out[0].start == 0 and out[0].text.startswith("Methods")


def test_without_min_tokens_heading_is_orphaned(tok):
    """Negative control: with no floor, the heading stays alone."""
    doc = "Methods\n\n" + para(40, 40) + "\n\n" + para(60)
    out = run(make(tok, budget=200, min_tokens=0, across=False), doc)
    assert out[0].text == "Methods"


# ---------------------------------------------------------------- paragraph structure
def test_paragraphs_under_budget_are_never_split_or_merged(tok):
    doc = "\n\n".join(para(n) for n in [30, 45, 12, 80, 25])
    out = run(make(tok, budget=200, across=False), doc)
    assert [(c.start, c.end) for c in out] == para_spans(doc)


def test_across_merge_keeps_boundaries_on_paragraph_edges(tok):
    doc = "\n\n".join(para(n) for n in [30, 45, 12, 80, 25, 90, 70])
    out = run(make(tok, budget=200, across=True), doc)
    spans = para_spans(doc)
    starts, ends = {s for s, _ in spans}, {e for _, e in spans}
    assert all(c.start in starts and c.end in ends for c in out)
    assert [c.n_tokens for c in out] == [192, 160]


def test_oversized_paragraph_splits_only_at_sentence_boundaries(tok):
    doc = para(90, 110, 60)
    out = run(make(tok, budget=200), doc)
    assert [c.n_tokens for c in out] == [200, 60]
    assert out[0].end == len(para(90, 110))
    assert out[1].start == len(para(90, 110)) + 1


def test_line_level_splits_before_sentence_level(tok):
    doc = "\n".join(cats(80) for _ in range(3))           # one paragraph, three 80-token lines
    out = run(make(tok, budget=200), doc)
    assert [c.n_tokens for c in out] == [160, 80]
    assert out[0].end == len(cats(80) + "\n" + cats(80))


def test_windows_line_endings_detect_paragraphs(tok):
    doc = para(60) + "\r\n\r\n" + para(50)
    assert [c.n_tokens for c in run(make(tok, across=False), doc)] == [60, 50]
    assert [c.n_tokens for c in run(make(tok, across=True), doc)] == [110]


# ---------------------------------------------------------------- fallback
def test_fallback_when_no_separators_at_all(tok):
    doc = cats(450)                                        # no blank line, no newline, no period
    out = run(make(tok, budget=200), doc)
    assert [c.n_tokens for c in out] == [200, 200, 50]
    assert out[0].start == 0 and out[-1].end == len(doc)


def test_fallback_pieces_point_into_original_document(tok):
    doc = "\n\n".join([para(20), cats(450), para(20)])
    out = run(make(tok, budget=200, across=False), doc)
    big_start = len(para(20)) + 2
    big_end = big_start + len(cats(450))
    assert [c.n_tokens for c in out] == [20, 200, 200, 50, 20]
    assert out[1].start == big_start and out[3].end == big_end
    assert_shared_invariants(doc, out, tok)


# ---------------------------------------------------------------- oracle: equals sentence chunking on unstructured text
@pytest.mark.parametrize("budget", [120, 200])
@pytest.mark.parametrize("across", [False, True])
def test_matches_sentence_chunker_when_no_structure(tok, budget, across):
    """No blank lines, no newlines, no oversized sentence: recursive reduces to sentence packing (Section 4.5)."""
    doc = para(30, 45, 12, 80, 25, 60, 99, 101, 20)
    rec = run(make(tok, budget=budget, across=across), doc)
    sen = run(SentenceChunker(tok, stub_splitter, budget=budget, overlap_sentences=0), doc)
    assert [(c.start, c.end) for c in rec] == [(c.start, c.end) for c in sen]


# ---------------------------------------------------------------- shared invariants on realistic structured text
@pytest.mark.parametrize("budget", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("across", [False, True])
@pytest.mark.parametrize("min_tokens", [0, 8])
def test_shared_invariants_real_text(tok, budget, across, min_tokens):
    out = run(make(tok, budget=budget, min_tokens=min_tokens, across=across), REAL_DOC, "doc-real")
    assert len(out) >= 2
    assert_shared_invariants(REAL_DOC, out, tok, doc_id="doc-real", allow_overlap=False)
    assert all(c.n_tokens <= budget for c in out), "a chunk exceeds the budget"


def test_deterministic(tok):
    ch = make(tok, budget=32, min_tokens=8)
    assert run(ch, REAL_DOC) == run(ch, REAL_DOC)