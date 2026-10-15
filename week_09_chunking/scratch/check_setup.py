"""Week 9 setup check. Run from week09_chunking:  python scratch\check_setup.py"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import spacy
import sentence_transformers
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",   # primary retrieval + segmentation model
    "sentence-transformers/all-mpnet-base-v2",  # independent model for coherence (avoids the circularity)
    "BAAI/bge-base-en-v1.5",                    # second model for the 2x2 chunker-vs-embedder block
]

env = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "sentence_transformers": sentence_transformers.__version__,
    "spacy": spacy.__version__,
    "numpy": np.__version__,
    "models": {},
}

print("== models ==")
for name in MODELS:
    m = SentenceTransformer(name)
    info = {
        "max_seq_length": m.max_seq_length,                         # the limit encode() enforces
        "dim": m.get_sentence_embedding_dimension(),
        "tokenizer_is_fast": m.tokenizer.is_fast,                    # must be True for offset mapping
        "tokenizer_model_max_length": m.tokenizer.model_max_length,  # the number the warning quotes
    }
    env["models"][name] = info
    print(name, info)
    print("   pooling:", m[1])

print("\n== spaCy ==")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
env["spacy_model"] = f"en_core_web_sm {nlp.meta['version']}"
text = "Patients improved (p < 0.05; see Fig. 2). Smith et al. found similar results."
for s in nlp(text).sents:
    assert text[s.start_char:s.end_char] == s.text
    print(f"  [{s.start_char}:{s.end_char}] {s.text!r}")

print("\n== SciFact ==")
sf = DATA / "scifact"
corpus = [json.loads(line) for line in open(sf / "corpus.jsonl", encoding="utf-8")]
queries = [json.loads(line) for line in open(sf / "queries.jsonl", encoding="utf-8")]
qrels_lines = (sf / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()
test_qids = {line.split("\t")[0] for line in qrels_lines[1:]}
words = [len(d["text"].split()) for d in corpus]
print("  qrels header:", qrels_lines[0])
print("  corpus docs:", len(corpus), "| all queries:", len(queries), "| test queries:", len(test_qids))
print("  first doc keys:", list(corpus[0].keys()))
print("  train qrels present:", (sf / "qrels" / "train.tsv").exists())
print("  words per doc p50 / p90 / max:", np.percentile(words, [50, 90]).round(1), max(words))
print("  docs containing a blank line:", sum("\n\n" in d["text"] for d in corpus))

print("\n== Qasper ==")
qd = DATA / "qasper"
for fname in ["qasper-dev-v0.3.json", "qasper-test-v0.3.json"]:
    papers = json.loads((qd / fname).read_text(encoding="utf-8"))
    n_q = sum(len(p["qas"]) for p in papers.values())
    print(f"  {fname}: {len(papers)} papers, {n_q} questions")

dev = json.loads((qd / "qasper-dev-v0.3.json").read_text(encoding="utf-8"))
pid, paper = next(iter(dev.items()))
ft = paper["full_text"]
first_sec = ft[0] if isinstance(ft, list) else {k: v[0] for k, v in ft.items()}
qa = paper["qas"][0]
ans = qa["answers"][0]
print("  paper id:", pid, "| keys:", list(paper.keys()))
print("  full_text type:", type(ft).__name__)
print("  first section:", repr(first_sec["section_name"]), "|", len(first_sec["paragraphs"]), "paragraphs")
print("  qa keys:", list(qa.keys()))
print("  answer keys:", list(ans.keys()), "| inner answer keys:", list(ans["answer"].keys()))
print("  example question:", qa["question"])
print("  example evidence:", ans["answer"]["evidence"][:1])

out = ROOT / "results" / "env.json"
out.write_text(json.dumps(env, indent=2), encoding="utf-8")
print("\nwrote", out)