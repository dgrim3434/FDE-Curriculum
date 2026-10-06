from extraction.schemas import EvidenceFirst, LabelFirst
from data.loaders import load_ag_news
from extraction.prompts import build_prompt
from llm import OllamaClient, AnthropicClient
from extraction.extractor import extract
from extraction.evaluation import followed_order, topic_correct, quote_grounded, paired_disagreements, noise_margin
from pydantic import ValidationError
from pathlib import Path
import json

qwen = OllamaClient()
claude = AnthropicClient()

RESULTS = Path(__file__).resolve().parent.parent / "results"

ARMS = [
    {"name": "qwen_evidence_first", "client": qwen, "schema": EvidenceFirst, "format": None},
    {"name": "qwen_label_first", "client": qwen, "schema": LabelFirst, "format": None},
    {"name": "claude_evidence_first", "client": claude, "schema": EvidenceFirst, "format": None},
    {"name": "claude_label_first", "client": claude, "schema": LabelFirst, "format": None},
    {"name": "qwen_evidence_first_formatted", "client": qwen, "schema": EvidenceFirst, "format": EvidenceFirst.model_json_schema()},
    {"name": "qwen_label_first_formatted", "client": qwen, "schema": LabelFirst, "format": LabelFirst.model_json_schema()},
]

def run_experiment():
    
    data = load_ag_news(n=100)
    results = {}
    for arm in ARMS:
        results[arm['name']] = {'summary': {}, 'sentences': []}
        
        valid = 0
        order_followed = 0
        accurate = 0
        accurate_compliant = 0
        ground = 0
        order_denom = 0
        for idx in range(len(data)):
            
            row = data[idx]
            
            text = row['text']
            
            r = extract(text=text, schema=arm['schema'], llm=arm['client'], item_id=row['id'], format=arm['format'])   

            order = followed_order(r.raw_outputs[-1], list(arm['schema'].model_fields))
 
            if order:
                order_followed += 1
            
            if order is not None:
                order_denom += 1
            try:
                res = r.value.model_dump()
                valid += 1
            except AttributeError:
                res = None
                
            
            is_correct = topic_correct(res, row['topic'])
            grounded = quote_grounded(res, row['text'])

            if is_correct:
                accurate += 1
                if order:
                    accurate_compliant += 1
            
            if grounded:
                ground += 1
                
            results[arm['name']]['sentences'].append({'id': row['id'], 'gold': row['topic'], 'outcome': r.outcome, 'value': res, 'followed': order, 'correct': is_correct, 'grounded': grounded})
    
        accuracy = float(accurate / len(data))
        results[arm['name']]['summary'] = {'n': len(data), 'schema_valid': valid, 'order_followed': float(order_followed / order_denom), 'accuracy': accuracy,
                                           'accuracy_and_ordered': float(accurate_compliant / order_followed), 'grounded': float(ground / valid), 'noise_margin': noise_margin(accuracy, len(data))}

    pairs = [[0,1], [2,3], [4,5]]
    results['order_comparison'] = {}
    for pair in pairs:
        d1_name = ARMS[pair[0]]['name']
        d2_name = ARMS[pair[1]]['name']
        
        d1 =  results[d1_name]['sentences']
        d2  = results[d2_name]['sentences']
        
        a, b = run_comparison(d1, d2)
        
        results['order_comparison'][f'{d1_name}_vs_{d2_name}'] = {'a': a, 'b': b}
    
    (RESULTS / "field_order_comparison.json").write_text(json.dumps(results, indent=2), encoding='utf-8')
    
    
def correct_dict(data):
    
    corr_dict = {}
    
    for idx in range(len(data)):
        
        corr_dict[data[idx]['id']] = data[idx]['correct']
    
    return corr_dict

def run_comparison(d1, d2):
    
    d1_corr = correct_dict(d1)
    d2_corr = correct_dict(d2)
    
    return paired_disagreements(d1_corr, d2_corr)

if __name__ == "__main__":
    run_experiment()
