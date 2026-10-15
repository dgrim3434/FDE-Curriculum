import re
import threading

USE_TIMEOUT = False      # set to False while debugging; back to True for the final run


def cats(n):
    """n tokens, no punctuation, no newlines."""
    return " ".join(["cat"] * n)


def sent(n):
    """A sentence of exactly n tokens (n >= 2): (n-1) words 'cat' + a final '.'"""
    return " ".join(["cat"] * (n - 1)) + "."


def para(*lengths):
    """A paragraph made of sentences with the given token counts."""
    return " ".join(sent(n) for n in lengths)


def stub_splitter(text):
    """Stand-in for spaCy: a sentence ends at each '.'; spans include the leading space."""
    return [(m.start(), m.end()) for m in re.finditer(r"[^.]+(?:\.|$)", text)]


def run(chunker, doc, doc_id="d1", seconds=10):
    """Call chunker.chunk with a time limit, so an infinite loop fails instead of hanging."""
    if not USE_TIMEOUT:
        return chunker.chunk(doc, doc_id)
    box = {}

    def target():
        try:
            box["out"] = chunker.chunk(doc, doc_id)
        except BaseException as e:
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "chunk() did not return within the time limit (infinite loop?)"
    if "err" in box:
        raise box["err"]
    return box["out"]