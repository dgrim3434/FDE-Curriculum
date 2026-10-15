import random

import pytest

from chunking.base import Chunk, Chunker, count_tokens, make_chunk, trim_span


# ---------------------------------------------------------------- Chunk
def test_chunk_metadata_defaults_to_empty_dict():
    c = Chunk("d1", 3, 10, 20, "x" * 10, 2)
    assert isinstance(c.metadata, dict)
    assert c.metadata == {}


def test_chunk_metadata_not_shared_between_instances():
    a = Chunk("d", 0, 0, 1, "a", 1)
    b = Chunk("d", 1, 1, 2, "b", 1)
    a.metadata["section"] = "Methods"
    assert b.metadata == {}, "every Chunk must get its own metadata dict"


def test_chunk_id_property():
    c = Chunk("doc42", 7, 0, 1, "a", 1)
    assert c.chunk_id == "doc42::7"          # a property: accessed without ()


# ---------------------------------------------------------------- count_tokens
def test_count_tokens_excludes_special_tokens(tok):
    assert count_tokens(tok, "hello world") == 2


@pytest.mark.parametrize("text", [
    "Statins lowered LDL cholesterol by 23%.",
    "Pembrolizumab (anti-PD-1) therapy was well tolerated.",
    "café — naïve résumé",
])
def test_count_tokens_matches_tokenizer_without_specials(tok, text):
    expected = len(tok(text, add_special_tokens=False)["input_ids"])
    assert count_tokens(tok, text) == expected
    assert count_tokens(tok, text) + 2 == len(tok(text)["input_ids"])   # the 2 specials


def test_count_tokens_empty_and_whitespace(tok):
    assert count_tokens(tok, "") == 0
    assert count_tokens(tok, "   \n\t ") == 0


# ---------------------------------------------------------------- trim_span
@pytest.mark.parametrize("doc,span,expected", [
    ("  abc  ", (0, 7), (2, 5)),
    ("abc", (0, 3), (0, 3)),                # end == len(doc): must not index past the end
    ("ab  cd  ef", (2, 8), (4, 6)),         # interior span
    ("\n\t x \r\n", (0, 7), (3, 4)),        # tabs and \r\n count as whitespace
])
def test_trim_span_cases(doc, span, expected):
    assert trim_span(doc, *span) == expected


def test_trim_span_all_whitespace_collapses():
    s, e = trim_span("a    b", 1, 5)
    assert s == e and 1 <= s <= 5


def test_trim_span_whitespace_only_document():
    s, e = trim_span("   ", 0, 3)
    assert s == e


def test_trim_span_empty_span():
    s, e = trim_span("abcdef", 3, 3)
    assert s == e


def test_trim_span_matches_strip_on_random_spans():
    """Property test: the trimmed span's text equals the stripped original slice."""
    rng = random.Random(0)
    for _ in range(2000):
        doc = "".join(rng.choice("ab \n\t") for _ in range(rng.randint(1, 30)))
        s = rng.randint(0, len(doc))
        e = rng.randint(s, len(doc))
        ts, te = trim_span(doc, s, e)
        assert s <= ts <= te <= e, (doc, s, e, ts, te)
        assert doc[ts:te] == doc[s:e].strip(), (doc, s, e, ts, te)


# ---------------------------------------------------------------- make_chunk
def test_make_chunk_trims_and_slices(tok):
    doc = "xx   hello world   yy"
    c = make_chunk(doc, "d9", 4, 2, 19, tok)
    assert (c.start, c.end, c.text) == (5, 16, "hello world")
    assert c.doc_id == "d9" and c.chunk_index == 4
    assert c.n_tokens == 2
    assert c.metadata == {}


def test_make_chunk_whitespace_span_returns_none(tok):
    assert make_chunk("a     b", "d", 0, 1, 6, tok) is None


def test_make_chunk_text_is_the_original_slice(tok):
    doc = "Mortality fell 23%.  It was significant.\nNo deaths occurred."
    c = make_chunk(doc, "d", 0, 0, len(doc), tok)
    assert c.text == doc                      # double space and newline preserved


# ---------------------------------------------------------------- Chunker
def test_chunker_base_raises():
    with pytest.raises(NotImplementedError):
        Chunker().chunk("some text", "d1")