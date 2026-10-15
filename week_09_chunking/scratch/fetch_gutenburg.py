"""Week 9 Deliverable 2 data: a small Project Gutenberg sample with catalog labels.

Run from week_09_chunking:  python scratch/fetch_gutenberg.py
Downloads 30 books (2 s apart) + the catalog CSV into data/gutenberg/.
"""
import csv
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gutenberg"
OUT.mkdir(parents=True, exist_ok=True)

BOOK_IDS = [1342, 5200, 2600,                 # shown below: plain / translated / multiple translators
            84, 11, 1661, 2701, 98, 1080, 74, 345, 174, 1952, 2542, 64317,
            768, 1260, 16328, 4300, 1232, 1400, 76, 205, 2554, 158, 1184, 120, 36, 3207, 6130]
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
HEADERS = {"User-Agent": "FDE-curriculum-week9 (educational; small sample)"}


def fetch(url, path):
    if path.exists():                      # rerunning the script never re-downloads
        return
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        path.write_bytes(r.read())
    time.sleep(2)


fetch(CATALOG_URL, OUT / "pg_catalog.csv")
for bid in BOOK_IDS:
    try:
        fetch(BOOK_URL.format(id=bid), OUT / f"pg{bid}.txt")
        print("ok", bid)
    except Exception as e:
        print("FAILED", bid, e)

# Raw catalog labels for the sampled books. Normalizing them is your evaluation code's job.
labels = {}
with open(OUT / "pg_catalog.csv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row["Text#"].isdigit() and int(row["Text#"]) in BOOK_IDS:
            labels[row["Text#"]] = {"title": row["Title"], "authors": row["Authors"],
                                    "issued": row["Issued"]}
(OUT / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nlabels for {len(labels)} books -> {OUT / 'labels.json'}")
for bid in BOOK_IDS[:3]:
    print(f"  {bid}: {labels.get(str(bid))}")

# What the front matter looks like: three books, first 25 lines each.
for bid in BOOK_IDS[:3]:
    text = (OUT / f"pg{bid}.txt").read_text(encoding="utf-8-sig", errors="replace")
    print(f"\n----- pg{bid}: first 25 lines -----")
    for line in text.splitlines()[:25]:
        print(repr(line))

# How Qasper names its sections (this decides how you render headings into doc_text).
dev = json.loads((ROOT / "data" / "qasper" / "qasper-dev-v0.3.json").read_text(encoding="utf-8"))
for pid in list(dev)[:3]:
    print(f"\n----- Qasper {pid}: section names -----")
    for sec in dev[pid]["full_text"]:
        print(repr(sec["section_name"]))