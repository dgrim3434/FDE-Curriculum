from __future__ import annotations
 
import argparse
import collections
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
 
DATA_DIR = Path(__file__).resolve().parent

SOURCES = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt",
]

MIN_BYTES = 100_000
TIMEOUT = 30    

def fetch(urls: list[str]) -> str:
    """Try each URL in order. Returns decoded text, or raises RuntimeError."""
    errors = []
    for url in urls:
        try:
            print(f"  fetching {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "week04-corpus/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                raw = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            errors.append(f"{url} -> {type(e).__name__}: {e}")
            print(f"    failed: {type(e).__name__}")
            continue
 
        if len(raw) < MIN_BYTES:
            errors.append(f"{url} -> only {len(raw)} bytes, expected >= {MIN_BYTES}")
            print(f"    failed: only {len(raw)} bytes (probably an error page)")
            continue
 
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            errors.append(f"{url} -> not valid UTF-8: {e}")
            print("    failed: not valid UTF-8")
            continue
 
        print(f"    ok, {len(raw):,} bytes")
        return text
 
    raise RuntimeError(
        "Could not download the corpus. Tried:\n  "
        + "\n  ".join(errors)
        + "\n\nIf you are offline or behind a proxy, download the file manually and save it as "
        + f"{DATA_DIR / 'input.txt'}"
    )
 
 
def normalize(text: str) -> str:
    """CRLF -> LF, strip a BOM. Both add phantom characters to the vocabulary."""
    if text.startswith("﻿"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")
 
def describe(text: str, split: float) -> dict:
    counts = collections.Counter(text)
    vocab = sorted(counts)
    n = len(text)
    n_train = int(split * n)
    return {
        "chars": n,
        "vocab_size": len(vocab),
        "vocab": vocab,
        "counts": counts,
        "n_train": n_train,
        "n_val": n - n_train,
    }

def report(info: dict, text: str) -> None:
    print("\n  characters      : {:,}".format(info["chars"]))
    print("  distinct chars  : {}".format(info["vocab_size"]))
    print("  vocabulary      : {!r}".format("".join(info["vocab"])))
    print("  train / val     : {:,} / {:,}".format(info["n_train"], info["n_val"]))
    top = info["counts"].most_common(5)
    print("  most common     : " + ", ".join(f"{c!r}x{k:,}" for c, k in top))
    print("\n  first 120 characters:")
    print("  " + "-" * 60)
    for line in text[:120].split("\n"):
        print("  | " + line)
    print("  " + "-" * 60)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download the Week 4 corpus into data/.")
    p.add_argument("--out", default="input.txt", help="filename inside data/ (default: input.txt)")
    p.add_argument("--url", default=None, help="download from this URL instead of the defaults")
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    p.add_argument("--stats", action="store_true", help="also compute the n-gram loss floors")
    p.add_argument("--split", type=float, default=0.9, help="train fraction (default: 0.9)")
    args = p.parse_args(argv)
 
    out_path = (DATA_DIR / args.out).resolve()
    if DATA_DIR not in out_path.parents and out_path.parent != DATA_DIR:
        print(f"refusing to write outside {DATA_DIR}", file=sys.stderr)
        return 2
 
    print(f"data directory : {DATA_DIR}")
    print(f"target file    : {out_path}")
 
    if out_path.exists() and not args.force:
        print(f"\n  {out_path.name} already exists ({out_path.stat().st_size:,} bytes). "
              f"Use --force to re-download.")
        text = out_path.read_text(encoding="utf-8")
    else:
        urls = [args.url] if args.url else SOURCES
        try:
            text = normalize(fetch(urls))
        except RuntimeError as e:
            print(f"\n{e}", file=sys.stderr)
            return 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8", newline="\n")
        print(f"\n  wrote {out_path} ({out_path.stat().st_size:,} bytes)")
 
    # verification: read back what is actually on disk, not what we think we wrote
    on_disk = out_path.read_text(encoding="utf-8")
    assert on_disk == text, "round-trip mismatch: what was written is not what was read back"
 
    info = describe(on_disk, args.split)
    report(info, on_disk)
 
    if info["vocab_size"] < 2:
        print("\n  ERROR: vocabulary has fewer than 2 characters.", file=sys.stderr)
        return 1
    if "\r" in on_disk:
        print("\n  WARNING: carriage returns survived normalization; your vocab has a phantom char.")
 
    print("\n  next: python -c \"from train.data import load_shakespeare; "
          "print(len(load_shakespeare()[0]))\"")
    return 0

if __name__ == "__main__":
    main()