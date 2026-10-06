"""
AI GENERATED DataSet Loader

Each loader returns a list of plain dicts, so nothing else in the project
touches Hugging Face objects. Entities and relations are stored as tuples,
ready to put into sets for precision / recall.
"""
from datasets import load_dataset

# CoNLL-2003 stores NER tags as integers. This is the standard order.
NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def bio_to_entities(tokens, tag_ids):
    """Turn per-token BIO tags into (text, type) pairs.

    B-X starts a new entity of type X. I-X continues it.
    An I-X that doesn't follow an entity of type X starts a new one.
    """
    entities, current, current_type = [], [], None
    for tok, tag_id in zip(tokens, tag_ids):
        tag = NER_TAGS[tag_id]
        if tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != current_type):
            if current:
                entities.append((" ".join(current), current_type))
            current, current_type = [tok], tag[2:]
        elif tag.startswith("I-"):
            current.append(tok)
        else:  # "O"
            if current:
                entities.append((" ".join(current), current_type))
            current, current_type = [], None
    if current:
        entities.append((" ".join(current), current_type))
    return entities


def load_conll2003(split="validation", n=None, min_tokens=5):
    """Sentences with gold (text, type) entities. Labels: PER, ORG, LOC, MISC."""
    ds = load_dataset("lhoestq/conll2003", split=split)
    rows = []
    for ex in ds:
        tokens = ex["tokens"]
        if len(tokens) < min_tokens or tokens[0] == "-DOCSTART-":
            continue
        rows.append({
            "id": f"conll03_{split}_{ex['id']}",
            "sentence": " ".join(tokens),
            "entities": bio_to_entities(tokens, ex["ner_tags"]),
        })
        if n is not None and len(rows) >= n:
            break
    return rows


def load_conll04(split="validation", n=None):
    """Sentences with gold entities and (subject, relation, object) triples.

    Labels: Peop, Loc, Org, Other. Relations: Work_For, Kill, OrgBased_In,
    Live_In, Located_In. Entity 'end' is exclusive, like a Python slice.
    """
    ds = load_dataset("DFKI-SLT/conll04", split=split)
    rows = []
    for i, ex in enumerate(ds):
        toks = ex["tokens"]
        # If this line raises a TypeError, print(ex["entities"]): older versions
        # of `datasets` return a dict of lists here instead of a list of dicts.
        ents = [(" ".join(toks[e["start"]:e["end"]]), e["type"]) for e in ex["entities"]]
        rels = [(ents[r["head"]][0], r["type"], ents[r["tail"]][0]) for r in ex["relations"]]
        rows.append({
            "id": f"conll04_{split}_{i:04d}",
            "sentence": " ".join(toks),
            "entities": ents,
            "relations": rels,
        })
        if n is not None and len(rows) >= n:
            break
    return rows


def load_ag_news(n=100, seed=0, split="test"):
    """News articles with a gold topic. Fixed seed, so the same n articles every run."""
    ds = load_dataset("fancyzhx/ag_news", split=split).shuffle(seed=seed).select(range(n))
    return [
        {"id": f"ag_{split}_{i:04d}", "text": ex["text"], "topic": AG_LABELS[ex["label"]]}
        for i, ex in enumerate(ds)
    ]