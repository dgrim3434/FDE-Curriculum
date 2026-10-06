from pathlib import Path
from extraction.evaluation import *
from extraction.grounding import grounded_rate
from collections import defaultdict
import json

RESULTS = Path(__file__).resolve().parent.parent / "results"

llama_data = json.loads((RESULTS / "run_qwen2-5_3b.json").read_text(encoding='utf-8'))
claude_data = json.loads((RESULTS / "run_claude-haiku-4-5-20251001.json").read_text(encoding='utf-8'))

DATASETS = [(llama_data, 'qwen2.5:3b'), (claude_data, 'claude-haiku-4-5-20251001')]

SCHEMAS = ['EntityTypes', 'Extraction', 'Event', 'Graph']

def calculate_results():
    results = {}
    
    for data in DATASETS:
        results[data[1]] = {}
        groups = defaultdict(list)
        
        for idx in range(len(data[0])):
            
            record = data[0][idx]
            groups[record['schema']].append(record)
        
        
        
        for schema in SCHEMAS:
            
            rs = groups[schema]
            
            result = {'n': len(rs)}
            
            outcome_mix = outcomes_counter(rs)
            for k, v in outcome_mix[schema].items():
                result[k] = float(v / len(rs))
            
            if schema == "EntityTypes":
                accuracy = flat_accuracy(rs)
                result['accuracy'] = accuracy
            if schema == 'Extraction':
                
                gold =  [gold_pairs(r) for r in rs]
                pred = [to_pairs(r['value']) for r in rs]
                
                p_r_f = prf(gold, pred)
                result['precision_all'] = p_r_f[0]
                result['recall_all'] = p_r_f[1]
                result['f1_all'] = p_r_f[2]
                
                gold_v = [gold_pairs(r) for r in rs if r['outcome'] != 'failed']
                pred_v = [to_pairs(r['value']) for r in rs if r['outcome'] != 'failed']
                
                p_r_f_v= prf(gold_v, pred_v)
                result['precision_valid'] = p_r_f_v[0]
                result['recall_valid'] = p_r_f_v[1]
                result['f1_valid'] = p_r_f_v[2]

                result['grounded_rate'] = grounded_rate(rs)
                
            if schema == 'Graph':
                
                gold =  [gold_pairs(r) for r in rs]
                pred = [to_pairs(r['value']) for r in rs]
                
                prf_e = prf(gold, pred)
                result['entity_precision_all'] = prf_e[0]
                result['entity_recall_all'] = prf_e[1]
                result['entity_f1_all'] = prf_e[2]
                
                
                gold_t = [gold_triples(r) for r in rs]
                pred_t = [to_triples(r['value']) for r in rs]
                
                prf_t = prf(gold_t, pred_t)
                result['relation_precision_all'] = prf_t[0]
                result['relation_recall_all'] = prf_t[1]
                result['relation_f1_all'] = prf_t[2]
                
                gold = [gold_pairs(r) for r in rs if r['outcome'] != 'failed']
                pred = [to_pairs(r['value']) for r in rs if r['outcome'] != 'failed']
                
                prf_ev = prf(gold, pred)
                result['entity_precision_valid'] = prf_ev[0]
                result['entity_recall_valid'] = prf_ev[1]
                result['entity_f1_valid'] = prf_ev[2]
                gold_t = [gold_triples(r) for r in rs if r['outcome'] != 'failed']
                pred_t = [to_triples(r['value']) for r in rs if r['outcome'] != 'failed']
                
                prf_tv = prf(gold_t, pred_t)
                result['relation_precision_valid'] = prf_tv[0]
                result['relation_recall_valid'] = prf_tv[1]
                result['relation_f1_valid'] = prf_tv[2]
                
                result['grounded_rate'] = grounded_rate(rs)
            
            results[data[1]][schema] = result
        
        input_tokens = sum(x['input_tokens'] for x in data[0])
        output_tokens = sum(x['output_tokens'] for x in data[0])
        results[data[1]]['input_tokens'] = input_tokens
        results[data[1]]['output_tokens'] = output_tokens
        
        hits = rule_hits(data[0])
        
        for k, v in hits.items():
            
            results[data[1]][k] = v

    
    (RESULTS / "extraction_summary.json").write_text(json.dumps(results, indent=2), encoding='utf-8')

if __name__ == "__main__":
    calculate_results()
        