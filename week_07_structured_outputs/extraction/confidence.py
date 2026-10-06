from dataclasses import dataclass
from extraction.extractor import ExtractionResult, extract
from collections import Counter, defaultdict

@dataclass
class SelfConsistancy:
    results: list[ExtractionResult]
    valid: int
    input_tokens: int
    output_tokens: int
    
    
def sample(text, schema, llm, n=5, temperature=0.7, base_seed=42, item_id=""):
    
    results = []
    input_tokens = 0
    output_tokens = 0
    valid = 0
    for i in range(n):
        
        curr_seed = base_seed + i
        
        resp = extract(text, schema, llm, item_id=item_id, temperature=temperature, seed=curr_seed)
        input_tokens += resp.input_tokens
        output_tokens += resp.output_tokens
        
        if resp.outcome != "failed":
            valid +=1
            results.append(resp.value)
    
    return SelfConsistancy(results=results, valid=valid, input_tokens=input_tokens, output_tokens=output_tokens)

def agreement(sets, n):
    
    counter = Counter()
    
    for set in sets:
        
        counter.update(set)
    
    return {k: float(v / n) for k, v in counter.items()}

        
def majority(scores, threshold=0.5):
    
    return {k for k,v in scores.items() if v >= threshold}

def calibration_table(pairs):
    groups = defaultdict(list)
    for pair in pairs:

        groups[pair[0]].append(pair[1]) 
    
    result = {}

    for k, v in groups.items():

        if not isinstance(v, list):
            correct = 1 if v else 0
            result[k] = (1, correct)
        else:
            
            correct = sum(1 if x == True else 0 for x in v) 
            result[k] = (len(v), float(correct / len(v))) if len(v) > 0 else None
    
    return result