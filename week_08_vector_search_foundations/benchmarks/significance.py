import numpy as np
from pathlib import Path
import pandas as pd

RESULT = Path(__file__).resolve().parent.parent / "results"
import json

def significance_test(res1, res2):
    
    keys_1 = set(res1.keys())
    keys_2 = set(res2.keys())
    
    if not same_sample_check(keys_1, keys_2):
        raise ValueError("ERROR: Keys must contain the same id's")
    
    b, c, t = 0,0,0
    
    for k in keys_1:
        
        v1 = res1.get(k)
        v2 = res2.get(k)
        
        d = v1 - v2
        
        if d > 0:
            b += 1
        elif d < 0:
            c +=1
        else:
            t += 1
    
    return b, c, t

def bootstrapping(res1, res2, seed=0, samples=1000):
    
    keys_1 = set(res1.keys())
    keys_2 = set(res2.keys())
    
    if not same_sample_check(keys_1, keys_2):
        raise ValueError("ERROR: Keys must contain the same id's")

    values_1 = []
    values_2 = []
    
    for k in sorted(keys_1):
        
        values_1.append(res1.get(k))
        values_2.append(res2.get(k))
    
    values_1 = np.array(values_1)
    values_2 = np.array(values_2)
    
    differences = []
    counts = len(values_1)
    rng = np.random.default_rng(seed=seed)
    
    for i in range(samples):
        
        index = rng.integers(0, counts, size= counts)
        
        v1 = values_1[index]
        v2 = values_2[index]
        
        diff = np.mean(v1 - v2)
        
        differences.append(diff)
    
    lo, high = np.percentile(differences, [2.5, 97.5])
    
    return lo, high, np.mean(values_1 - values_2)
    

def same_sample_check(keys_1: set, keys_2: set) -> bool:
    
    return keys_1 == keys_2


def run_significance():
    
    path = RESULT / "compare_models_per_query.json"
    return_path = RESULT / "compare_model_head_to_head.json"
    
    pairs = [('e5', 'e5_noprefix'), ('bge', 'bge_noprefix'), ('minilm', 'mpnet'), ('bge', 'e5')]
    
    with open(path, 'r', encoding='utf-8') as f:
        
        per_query = json.load(f)
    
    result = {}
    
    for pair in pairs:
        key = f'{pair[0]}_vs_{pair[1]}'
        result[key] = {}
        
        res1 = per_query[pair[0]]
        res2 = per_query[pair[1]]
        
        
        b, c, t = significance_test(res1, res2)
        
        result[key]['direction'] = f'{pair[0]} - {pair[1]}'
        result[key]['wins'] = b
        result[key]['losses'] = c
        result[key]['ties'] = t
        
        
        lo, high, mean = bootstrapping(res1, res2, seed=0, samples=1000)
        
        result[key]['mean_difference'] = mean
        result[key]['bootstrapping_low_2_5th'] = lo
        result[key]['bootstrapping_high_97_5th'] = high
        
        include_zero = False
        if lo <= 0 and high >= 0:
            include_zero = True
        result[key]['bootstrapping_includes_zero'] = include_zero

        vals_1 = np.mean(list(res1.values()))
        vals_2 = np.mean(list(res2.values()))
        
        result[key][f'{pair[0]}_mean'] = vals_1
        result[key][f'{pair[1]}_mean'] = vals_2
        
    (return_path).write_text(json.dumps(result, indent=2), encoding = 'utf-8')
        

if __name__ == "__main__":
    run_significance()