def assert_shared_invariants(doc_text, chunks, tok, doc_id="d1", allow_overlap=False):
    """The contract every chunker must satisfy (Section 3.1)."""
    assert isinstance(chunks, list), "chunk() must return a list"

    # chunk_index is 0..n-1 with no gaps
    assert [c.chunk_index for c in chunks] == list(range(len(chunks))), \
        "chunk_index must be 0, 1, 2, ... with no gaps"

    for c in chunks:
        assert c.doc_id == doc_id
        # (1) span invariant
        assert doc_text[c.start:c.end] == c.text, \
            f"chunk {c.chunk_index}: doc_text[start:end] != text"
        # (7) no empty chunks, no leading/trailing whitespace
        assert c.text, f"chunk {c.chunk_index} is empty"
        assert not c.text[0].isspace() and not c.text[-1].isspace(), \
            f"chunk {c.chunk_index} is not whitespace-trimmed"
        # (8) recorded count = fresh re-tokenization, WITHOUT special tokens
        assert c.n_tokens == len(tok(c.text, add_special_tokens=False)["input_ids"]), \
            f"chunk {c.chunk_index}: n_tokens does not match re-tokenized text"

    # (2) starts strictly increase
    starts = [c.start for c in chunks]
    assert all(a < b for a, b in zip(starts, starts[1:])), "chunk starts must strictly increase"

    # (4) overlap only when configured
    if not allow_overlap:
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.start >= prev.end, \
                f"chunks {prev.chunk_index} and {nxt.chunk_index} overlap but overlap is off"

    # (3) coverage: every non-whitespace character is inside some chunk
    covered = [False] * len(doc_text)
    for c in chunks:
        for i in range(c.start, c.end):
            covered[i] = True
    missing = [i for i, ch in enumerate(doc_text) if not ch.isspace() and not covered[i]]
    assert not missing, f"{len(missing)} non-whitespace characters are in no chunk (first: {missing[:5]})"