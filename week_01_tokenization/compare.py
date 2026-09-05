import json, pathlib
import tiktoken
from tokenizer import Tokenizer

TEST_SETS = {}

for name in ['code_test', 'prose_test']:
    TEST_SETS[name] = pathlib.Path(f"corpus/{name}.txt").read_text(encoding="utf-8")


tokenizers = {}

for label, path, V in [
    ("mine-code-1024",  "models/code_1024.model",  1024),
    ("mine-code-4096",  "models/code_4096.model",  4096),
    ("mine-prose-1024", "models/prose_1024.model", 1024),
    ("mine-prose-4096", "models/prose_4096.model", 4096),
]:
    t = Tokenizer()
    t.load(path)
    tokenizers[label] = (t, V, True)

for enc_name in ['gpt2', 'cl100k_base', 'o200k_base']:
    enc = tiktoken.get_encoding(enc_name)
    tokenizers[enc_name] = (enc, enc.n_vocab, False)

def count_tokens(tok, is_mine, text):
    
    if is_mine:
        return len(tok.encode(text))
    return len(tok.encode_ordinary(text))

def token_bytes(tok, is_mine, token_id):
    
    if is_mine:
        return tok.vocab[token_id]
    return tok.decode_single_token_bytes(token_id)

def encode_ids(tok, is_mine, text):
    if is_mine:
        return tok.encode(text)
    return tok.encode_ordinary(text)

rows = []
for label, (tok, V, is_mine) in tokenizers.items():
    row = {"tokenizer": label, "V": V}
    for set_name, text in TEST_SETS.items():
        n = count_tokens(tok, is_mine, text)
        row[set_name + "_chars"]  = len(text)
        row[set_name + "_tokens"] = n
        row[set_name + "_cpt"]    = len(text) / n          # chars per token
    row["domain_ratio"] = row["code_test_cpt"] / row["prose_test_cpt"]
    rows.append(row)

header = f"{'tokenizer':<18}{'V':>9}{'code c/t':>11}{'prose c/t':>12}{'ratio':>9}"
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r['tokenizer']:<18}{r['V']:>9,}{r['code_test_cpt']:>11.2f}"
          f"{r['prose_test_cpt']:>12.2f}{r['domain_ratio']:>9.2f}")

pathlib.Path("results").mkdir(exist_ok=True)
json.dump(rows, open("results/comparison.json", "w"), indent=2)


# ------------------------------------------------------- 5. BOUNDARY INSPECTION
SAMPLES = [
    "def process_batch(self, x):",
    "    return self._cache[key]",
    "The quick brown fox jumps over the lazy dog.",
    "pembrolizumab 12.5 mg/kg",
    "TX-88421-QQ",
    "1234 vs 1235",
]

SHOW = ["mine-code-4096", "mine-prose-4096", "gpt2", "cl100k_base"]

for sample in SAMPLES:
    print(f"\n{sample!r}")
    for label in SHOW:
        tok, V, is_mine = tokenizers[label]
        ids = encode_ids(tok, is_mine, sample)
        pieces = [token_bytes(tok, is_mine, i).decode("utf-8", errors="replace") for i in ids]
        print(f"  {label:<18} ({len(ids):>2}) " + " | ".join(pieces))


# ----------------------------------------------------------- 6. MERGE-LIST DIFF
code_tok, _, _  = tokenizers["mine-code-4096"]
prose_tok, _, _ = tokenizers["mine-prose-4096"]

print(f"\n{'rank':<6}{'code-trained':<24}{'prose-trained':<24}")
print("-" * 54)
for rank in range(50):
    tid = 256 + rank
    c = code_tok.vocab.get(tid,  b"").decode("utf-8", errors="replace")
    p = prose_tok.vocab.get(tid, b"").decode("utf-8", errors="replace")
    print(f"{rank:<6}{c!r:<24}{p!r:<24}")


# ------------------------------------- 7. VERIFY THE LESSON'S CLAIMS YOURSELF
gpt2_enc = tokenizers["gpt2"][0]
cl_enc   = tokenizers["cl100k_base"][0]

print("\n--- leading whitespace ---")
for s in [" return", "return", " the", "the"]:
    print(f"  {s!r:<10} gpt2={gpt2_enc.encode(s)}  cl100k={cl_enc.encode(s)}")

print("\n--- digit grouping ---")
for s in ["1234", "1235", "12345", "999", "1000"]:
    print(f"  {s!r:<8} gpt2={gpt2_enc.encode(s)}  cl100k={cl_enc.encode(s)}")

print("\n--- indentation ---")
s = "def f(x):\n    return x"
print("  gpt2  :", [gpt2_enc.decode_single_token_bytes(i) for i in gpt2_enc.encode(s)])
print("  cl100k:", [cl_enc.decode_single_token_bytes(i)   for i in cl_enc.encode(s)])
    
