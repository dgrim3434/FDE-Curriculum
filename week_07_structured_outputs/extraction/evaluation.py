from collections import Counter, defaultdict
from extraction.repair import parse_with_repairs
import math


def to_pairs(value):
    
    if value is None:
        return set()

    entities = value['entities']
    return {(e['text'], e['type']) for e in entities}

def to_triples(value):
    
    if value is None:
        return set()
    
    entities = value['relations']
    return {(e['subject'], e['relation'], e['object']) for e in entities}

def gold_pairs(record):
    
    gold = record['gold']
    if gold is None:
        return {}
    
    entities = gold['entities']
    return {tuple(e) for e in entities}

def gold_triples(record):
    
    gold = record['gold']
    
    if gold is None:
        return {}
    
    relations = gold['relations']
    
    return {tuple(e) for e in relations}

def prf(golds, preds):
    common = 0
    pred = 0
    act = 0
    for g, p in zip(golds, preds):
        
        common += len(g & p)
        pred += len(p)
        act += len(g)
    
    prec = common / pred if pred > 0 else 0
    recall = common / act if act > 0 else 0
    
    f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
    
    return prec, recall, f1

def outcomes_counter(records):
    
    counts = defaultdict(Counter)
    
    for idx in range(len(records)):
        
        schema = records[idx]['schema']
        outcome = records[idx]['outcome']
        
        counts[schema][outcome] += 1
    
    return counts

def rule_hits(records):
    
    counter = Counter()
    
    for idx in range(len(records)):
        rules = records[idx]['rules_fired']
        
        flattened = {x for row in rules for x in row}
        
        counter.update(flattened)
    
    return counter



def flat_accuracy(records):
    

    tot = 0
    
    has_person, has_organization, has_location, has_misc = 0,0,0,0
    label_mapping = {'has_person': 'PER', 'has_organization': 'ORG', 'has_location': 'LOC', 'has_misc': 'MISC'}

    for idx in range(len(records)):
        
        row = records[idx]

        if row['value'] is None:
            continue
        tot += 1
        
        values = row['value']
        gold_types = {i[1] for i in gold_pairs(row)}
        
        for k, v in values.items():
            
            contained = label_mapping.get(k) in gold_types
            if v == contained:
                if k == 'has_person':
                    has_person += 1
                elif k == 'has_organization':
                    has_organization += 1
                elif k == 'has_location':
                    has_location += 1
                else:
                    has_misc += 1
    
    if tot == 0:
        return {}
    return {'has_person': float(has_person / tot), 'has_organization': float(has_organization / tot), 'has_location': float(has_location / tot), 'has_misc': float(has_misc / tot)}

def key_order(raw_text):
    
    res = parse_with_repairs(raw_text)
    
    if res.ok:
        if not isinstance(res.data, dict):
            return None
        
        return [k for k, _ in res.data.items()]
    
    return None

def followed_order(raw_text, expected_keys):
    
    res = key_order(raw_text)
    
    if res is None:
        return None
    
    return res == expected_keys

def topic_correct(value, gold_topic):
    
    if value is None:
        return False
    
    return value['topic'] == gold_topic

def smoosh(text):
    return text.lower().strip()

def quote_grounded(value, article):
    
    if value is None:
        return False

    evidence = smoosh(value['evidence'])
    if len(evidence) == 0:
        return False
    
    return  evidence in  smoosh(article) 

def paired_disagreements(correct_a, correct_b):
    
    set_a = set(correct_a.keys())
    set_b = set(correct_b.keys())
    
    common = set_a & set_b
    if len(common) == 0:
        raise ValueError("There are no common questions within the two sets")
    
    a = 0
    b = 0
    
    for k in common:
        
        if correct_a[k] == correct_b[k]:
            continue
        if correct_a[k]:
            a += 1
        if correct_b[k]:
            b += 1
    
    return a, b

def noise_margin(p, n):
    
    return 2 * math.sqrt(p * (1- p) / n)