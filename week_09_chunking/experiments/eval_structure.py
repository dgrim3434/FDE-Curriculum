from metadata.structure import detect_headings
from retrieval.qasper import render_paper
from data.constants import QASPER_PATH
from collections import Counter
import json
from pathlib import Path
import pandas as pd
DATA_PATH = QASPER_PATH / "qasper-dev-v0.3.json"

data = json.loads(DATA_PATH.read_text(encoding='utf-8'))
RESULTS = Path(__file__).resolve().parent.parent / "results/metadata"

def run_evaluation():
    
    fp_examples = []
    results = []
    
    for k, v in data.items():
        
        text, gold = render_paper(v)
        
        pred = detect_headings(text)
        
        pred_set = Counter([(h.level, h.title) for h in pred])
        gold_set = Counter(gold)
        
        perfect_match = False
        if pred_set == gold_set:
            perfect_match = True
        
        tp = sum((gold_set & pred_set).values())
        false_pos = pred_set - gold_set
        missed = gold_set - pred_set
        
        if len(fp_examples) < 20:
            correct = ", ".join([f"label: {k[0]}, title: {k[1]}" for k in gold])
            if len(false_pos) > 0:
            
                extra = ", ".join([f'label: {k[0]}, title: {k[1]}' for k in list(false_pos.keys())])
                
                fp_examples.append({'example_id': k, 'failure_mode': 'false_positve', 'errors': extra, 'gold': correct})
            
            if len(missed) > 0:
                
                missed = ", ".join([f'label: {k[0]}, title: {k[1]}' for k in list(missed.keys())])
                fp_examples.append({'example_id': k, 'failure_mode': 'missed', 'errors': missed, 'gold': correct})
                
        precision = tp / len(pred)
        recall = tp / len(gold)
        
        
        results.append({'perfect_match': perfect_match, 'precision': precision, 'recall': recall})
    
    df = pd.DataFrame(results)
    
    perfect = len(df.loc[df['perfect_match']])
    precision = df['precision'].mean()
    recall = df['recall'].mean()
    
    summary = {'perfect_matches': perfect, 'precision': precision, 'recall': recall}
    
    (RESULTS / "qasper_headings_summary.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (RESULTS / "qasper_headings_raw.json").write_text(json.dumps(results), encoding='utf-8')
    (RESULTS / "qasper_headings_errors.json").write_text(json.dumps(fp_examples, indent=2), encoding='utf-8')


if __name__ == "__main__":
    
    run_evaluation()
    