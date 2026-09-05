"""Build train/test splits for both Corpora (Generated Using AI)"""
import sysconfig
import pathlib
import random

import nltk
from nltk.corpus import gutenberg

CORPUS = pathlib.Path("corpus")
CORPUS.mkdir(exist_ok=True)

TRAIN_CHARS = 250_000
TEST_CHARS = 60_000
PROSE_GAP = 50_000
SEED = 0

def collect_python_files():
    
    stdlib = pathlib.Path(sysconfig.get_paths()['stdlib'])
    skip = {"test", "tests", "idlelib", "lib2to3", "site-packages"}
    
    files = []
    
    for p in sorted(stdlib.rglob("*.py")):
        if any(part in skip for part in p.parts):
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        if len(text) < 500:
            continue
        
        files.append((p, text))
    
    return files

def build_code_corpus():
    
    files = collect_python_files()
    
    print(f"[code] found {len(files)} usable .py files")
    
    random.Random(SEED).shuffle(files)
    
    train, test = [], []
    train_chars = test_chars = 0
    
    for path, text in files:
        if train_chars < TRAIN_CHARS:
            train.append((path, text))
            train_chars += len(text)
        elif test_chars < TEST_CHARS:
            test.append((path, text))
            test_chars += len(text)
        else:
            break
    
    train_paths = {p for p, _ in train}
    test_paths = {p for p, _ in test}
    
    assert not (train_paths & test_paths), "LEAKAGE: Detected"
    assert test, "Test is empty"
    
    train_text = "\n".join(t for _, t in train)
    test_text = "\n".join(t for _, t in test)
    
    (CORPUS / "code_train.txt").write_text(train_text, encoding='utf-8')
    (CORPUS / "code_test.txt").write_text(test_text, encoding='utf-8')
    
    print(f"[code] train: {len(train):>4} files, {len(train_text):>9,} chars")
    print(f"[code] test: {len(test):>4} files, {len(test_text):>9,} chars")
    
def build_prose_corpus():
    nltk.download("gutenberg", quiet=True)
    text = gutenberg.raw("melville-moby_dick.txt")
    
    stripped = text.lstrip()
    if stripped.startswith('['):
        text = stripped[stripped.index("]") + 1:]
    text = text.strip()
    
    needed = TRAIN_CHARS + PROSE_GAP + TEST_CHARS
    
    assert len(text) >= needed, \
        f"book has {len(text):,} chars, need {needed:,} - pick a longer one"
    
    train = text[:TRAIN_CHARS]
    start = TRAIN_CHARS + PROSE_GAP
    test = text[start: start + TEST_CHARS]
    
    rng = random.Random(SEED)
    for _ in range(50):
        i = rng.randrange(0, len(test) - 200)
        assert test[i:i + 200] not in train, "Leakage: test text found in train"
    
    (CORPUS/ "prose_train.txt").write_text(train, encoding="utf-8")
    (CORPUS/"prose_test.txt").write_text(test, encoding='utf-8')
    
    print(f"[prose] source: {len(text):,} chars")
    print(f"[prose] train:  {len(train):,} chars")
    print(f"[prose] test:   {len(test):,} chars (starting at offset {start:,})")

if __name__ == "__main__":
    
    build_code_corpus()
    build_prose_corpus()