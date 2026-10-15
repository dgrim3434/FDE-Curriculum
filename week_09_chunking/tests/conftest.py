import re

import pytest

_TOKEN = re.compile(r"\w+|[^\w\s]")


class StubTokenizer:
    """Test stand-in for a Hugging Face fast tokenizer: no model files, no network.

    One token per word (\\w+) or punctuation mark, with exact character offsets.
    Supports the call signature the chunkers use:
        tok(text, add_special_tokens=..., return_offsets_mapping=..., verbose=...)
    """
    CLS, SEP = 101, 102

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False,
                 verbose=True, **kwargs):
        offsets = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
        ids = [1000 + i for i in range(len(offsets))]
        if add_special_tokens:
            ids = [self.CLS] + ids + [self.SEP]
            offsets = [(0, 0)] + offsets + [(0, 0)]
        enc = {"input_ids": ids}
        if return_offsets_mapping:
            enc["offset_mapping"] = offsets
        return enc


@pytest.fixture(scope="session")
def tok():
    return StubTokenizer()