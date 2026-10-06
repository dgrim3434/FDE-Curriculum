from data.loaders import load_conll04
from extraction.confidence import sample, agreement, majority, calibration_table
from extraction.schemas import Graph
from llm import OllamaClient, AnthropicClient
from pathlib import Path
from extraction.evaluation import to_triples, to_pairs, prf
import json



llama_llm = OllamaClient()
claude_llm = AnthropicClient()
SCHEMA = Graph
MODELS = [(llama_llm, 'qwen2.5:3b') , (claude_llm, 'claude-haiku-4-5-20251001')]

RESULTS = Path(__file__).parent.parent / "results"

N_SAMPLES = 5

def run_confidence():
    run_results = []
    
    data = load_conll04(n=100)[50:]
    
    for llm in MODELS:
        model_results = {'model': llm[1], 'n_samples': N_SAMPLES}
        results = []
        entity_confidence = []
        relation_confidence = []
        entity_preds = []
        relation_preds = []
        entity_gold = []
        relation_gold = []
        for idx in range(len(data)):
            
            row = data[idx]
            
            id = row['id']
            text = row['sentence']
            
            
            resp = sample(text, SCHEMA, llm[0], item_id=id, n= N_SAMPLES)
            
            entities = []
            relations= []
            
            
            gold_entities = set(row['entities'])
            entity_gold.append(gold_entities)
            gold_relations = set(row['relations'])
            relation_gold.append(gold_relations)
            
            for pred in resp.results:
                
                res = pred.model_dump()
                
                entities.append(to_pairs(res))
                relations.append(to_triples(res))
            
            entity_score = agreement(entities, n = N_SAMPLES)
            relations_score = agreement(relations, n = N_SAMPLES)
            
            entity_confidence_pairs = calculate_confidence_pairs(entity_score, gold_entities)
            relation_confidence_pairs = calculate_confidence_pairs(relations_score, gold_relations)
            
            entity_confidence.extend(entity_confidence_pairs)
            relation_confidence.extend(relation_confidence_pairs)
            
            ent_pred = majority(entity_score)
            relations_pred = majority(relations_score)
            
            entity_preds.append(ent_pred)
            relation_preds.append(relations_pred)
            
            results.append({'model': llm[1], 'id': id, 'valid_resonses': resp.valid, 'entity_pred': [list(t) for t in ent_pred], 'relations_pred': [list(t) for t in relations_pred], 'gold_entities':[list(t) for t in gold_entities], 'gold_relations': [list(t) for t in gold_relations],
                            'entity_confidence_pairs': [list(t) for t in entity_confidence_pairs], 'relation_confidence_pairs': relation_confidence_pairs, 'input_tokens': resp.input_tokens, 'output_tokens': resp.output_tokens
                            })
        
        
        entity_confidence = calibration_table(entity_confidence)
        relation_confidence = calibration_table(relation_confidence)
        model_results['entity_confidence'] = {}
        
        for k, pair in entity_confidence.items():
            model_results['entity_confidence'][k] = {'samples': pair[0], 'accuracy': pair[1]}
        
        model_results['relation_confidence'] = {}
        for k, pair in relation_confidence.items():
            model_results['relation_confidence'][k] = {'samples': pair[0], 'accuracy': pair[1]}
        
        prf_g = prf(entity_gold, entity_preds)
        prf_r = prf(relation_gold, relation_preds)
        
        model_results['f1'] = {'entity': prf_g[2], 'relation': prf_r[2]}

        model_results['sentences'] = results
        
        run_results.append(model_results)
        
       
    out = RESULTS / f"confidence.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(run_results, indent=2), encoding='utf-8')
        
                
def calculate_confidence_pairs(scores, gold):
    results = []
    for k, v in scores.items():
        
        valid = k in gold
        results.append((v, valid))
    
    return results
                      

if __name__ == "__main__":
    run_confidence()