"""test_golden.py — assert the tokenizer still behaves exactly as recorded."""
import hashlib
import json
import pathlib

from tokenizer import Tokenizer

GOLDEN = json.loads(pathlib.Path("tests/golden_tokens.json").read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def test_model_file_unchanged():
    """Fails the instant the .model file's bytes differ from the recording."""
    actual = sha256_file(GOLDEN["model_file"])
    assert actual == GOLDEN["model_sha256"], (
        f"\nMODEL FILE CHANGED\n"
        f"  expected sha256: {GOLDEN['model_sha256']}\n"
        f"  actual   sha256: {actual}\n"
        f"  If this change is intentional: retrain the model, rebuild every\n"
        f"  index built with this tokenizer, then re-run generate_golden.py."
    )


def test_metadata_unchanged():
    tok = Tokenizer()
    tok.load(GOLDEN["model_file"])
    assert len(tok.vocab) == GOLDEN["vocab_size"]
    assert len(tok.merges) == GOLDEN["num_merges"]
    assert tok.pattern == GOLDEN["pattern"], "pre-tokenizer regex changed"


def test_tokenization_unchanged():
    """Fails if ANY recorded input now produces different token IDs."""
    tok = Tokenizer()
    tok.load(GOLDEN["model_file"])

    failures = []
    for text, expected in GOLDEN["cases"].items():
        actual = tok.encode(text, allowed_special="none")
        if actual != expected:
            failures.append(
                f"  {text!r}\n"
                f"      expected: {expected}\n"
                f"      actual:   {actual}"
            )

    assert not failures, (
        f"\nTOKENIZATION CHANGED for {len(failures)} of "
        f"{len(GOLDEN['cases'])} cases:\n" + "\n".join(failures)
    )