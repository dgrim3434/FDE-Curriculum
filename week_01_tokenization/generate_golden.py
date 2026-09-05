"""Records the Current Tokenizers Behavior as Reference. Version Control"""
import hashlib
import json
import pathlib
from datetime import datetime, timezone

from tokenizer import Tokenizer

MODEL_PREFIX = "models/code_4096"
GOLDEN_PATH = pathlib.Path("tests/golden_tokens.json")

CASES = [
    "",
    " ",
    "    ",
    "\n",
    "hello world",
    "Hello World",
    " return",
    "return",
    "def process(self, x):\n    return self._cache[x]",
    "The quick brown fox jumps over the lazy dog.",
    "1234",
    "1235",
    "12345",
    "café",
    "中文测试",
    "🙂",
    "aaaaaaaa",
    "TX-88421-QQ",
]

def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def find_bytes(obj, path="golden"):
    if isinstance(obj, bytes):
        print(f"  BYTES VALUE at {path}: {obj[:60]!r}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, bytes):
                print(f"  BYTES KEY at {path}: {k!r}")
            find_bytes(v, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_bytes(v, f"{path}[{i}]")

def main():
    tok = Tokenizer()
    tok.load(MODEL_PREFIX + ".model")
    
    golden = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_file": MODEL_PREFIX + ".model",
        "model_sha256": sha256_file(MODEL_PREFIX + ".model"),
        "vocab_size": len(tok.vocab),
        "num_merges": len(tok.merged),
        "pattern": tok.SPLIT_PATTERN,
        "cases": {text: tok.encode(text, allowed_special="none") for text in CASES},
    }
    find_bytes(golden)
    for key, value in golden.items():
        print(f"  {key:<15} {type(value).__name__}")
    GOLDEN_PATH.parent.mkdir(exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print(f"wrote {GOLDEN_PATH}")
    print(f"  model sha256: {golden['model_sha256']}")
    print(f"  vocab_size:   {golden['vocab_size']}")
    print(f"  cases:        {len(CASES)}")


if __name__ == "__main__":
    main()