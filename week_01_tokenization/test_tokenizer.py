"""test_tokenizer.py — pytest suite for the BPE tokenizer."""

import random
import pytest
from tokenizer import Tokenizer, merge

TRAINING_TEXT = """
Hello, My name is Dylan Grim. I have recently decided to take things to the extreme.
I am working on becoming the best FDE in the game and that is going to take a lot of work
but I am completely here for it and am not scared about anything since God has my back and
will continue to guide me. This is my training Corpus which I will use to test my tokenizer for
week one.
"""

@pytest.fixture(scope="module")
def tok():
    """Trained once for the whole file. Only for tests that do NOT mutate it."""
    t = Tokenizer()
    t.train(512, TRAINING_TEXT)
    return t

@pytest.fixture
def fresh_tok():
    """Tokenizer per test. For tests that mutate state."""
    t = Tokenizer()
    t.train(512, TRAINING_TEXT)
    return t

def test_merge_overlapping_pairs():
    """'aaa' must consume both elements on a match, not reprocess."""
    assert merge((1, 1), [1, 1, 1], 99) == [99, 1]


def test_merge_index_guard():
    """Last element equals pair[0] with nothing after it -> must not IndexError."""
    assert merge((1, 2), [1], 99) == [1]
    assert merge((1, 2), [5, 1], 99) == [5, 1]


def test_merge_basic():
    assert merge((1, 2), [1, 2], 99) == [99]
    assert merge((5, 6), [1, 2, 3], 99) == [1, 2, 3]
    assert merge((1, 2), [], 99) == []


# ------------------------------------------------------------- round trip

def test_roundtrip_property(tok):
    """decode(encode(x)) == x for random strings across the full Unicode range."""
    rng = random.Random(0)                      # seeded -> reproducible failures
    for _ in range(1000):
        n = rng.randint(0, 200)
        s = "".join(chr(rng.randint(0, 0x10FFFF)) for _ in range(n))
        s = "".join(c for c in s if not (0xD800 <= ord(c) <= 0xDFFF))
        assert tok.decode(tok.encode(s)) == s, repr(s)


EDGE_CASES = [
    pytest.param("",                        id="empty"),
    pytest.param(" ",                       id="single_space"),
    pytest.param("  ",                      id="two_spaces"),
    pytest.param("    ",                    id="four_spaces"),
    pytest.param("\n",                      id="newline"),
    pytest.param("\t\t",                    id="tabs"),
    pytest.param("a",                       id="single_char"),
    pytest.param("aaaaaaaa",                id="repeated_char"),
    pytest.param("é",                       id="latin1_2byte"),
    pytest.param("中文测试",                 id="cjk_3byte"),
    pytest.param("🙂",                      id="emoji_4byte"),
    pytest.param("👨‍👩‍👧",                   id="zwj_sequence"),
    pytest.param("café",                    id="mixed_ascii_accent"),
    pytest.param("def f(x):\n    return x", id="code_indented"),
    pytest.param("1234567890",              id="digits"),
    pytest.param("a" * 100_000,             id="very_long_100k"),
]


@pytest.mark.parametrize("s", EDGE_CASES)
def test_edge_case_roundtrip(tok, s):
    assert tok.decode(tok.encode(s)) == s


@pytest.mark.parametrize("s", EDGE_CASES)
def test_token_ids_in_range(tok, s):
    ids = tok.encode(s)
    assert all(0 <= i < len(tok.vocab) for i in ids), f"out-of-range id in {ids}"


# ------------------------------------------------------------- invariants

def test_vocab_merges_consistency(tok):
    for (p0, p1), idx in tok.merged.items():
        assert tok.vocab[idx] == tok.vocab[p0] + tok.vocab[p1]


def test_save_load_identity(tok, tmp_path):
    prefix = str(tmp_path / "test")          # tmp_path: pytest-provided temp dir
    tok.save(prefix)

    tok2 = Tokenizer()
    tok2.load(prefix + ".model")

    assert tok2.merged == tok.merged
    assert tok2.vocab == tok.vocab
    assert tok2.SPLIT_PATTERN == tok.SPLIT_PATTERN

    sample = "Hello My Name is Dylan"
    assert tok2.encode(sample) == tok.encode(sample)


# --------------------------------------------------------- special tokens

def test_special_tokens(fresh_tok):
    fresh_tok.add_special(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    text = "<|im_start|>Hi, I am using special tokens<|im_end|>"

    ids_all = fresh_tok.encode(text, allowed_special="all")
    ids_none = fresh_tok.encode(text, allowed_special="none")

    assert fresh_tok.decode(ids_all) == text
    assert fresh_tok.decode(ids_none) == text

    # 'all' honors them as single IDs; 'none' encodes them as literal text
    assert len(ids_all) < len(ids_none)
    assert fresh_tok.special_tokens["<|im_start|>"] in ids_all
    assert fresh_tok.special_tokens["<|im_start|>"] not in ids_none


def test_special_tokens_raise_policy(fresh_tok):
    fresh_tok.add_special(["<|endoftext|>"])
    with pytest.raises(ValueError):
        fresh_tok.encode("hello <|endoftext|> world", allowed_special="none_raise")