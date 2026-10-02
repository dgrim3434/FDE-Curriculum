from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
from prompting.parsing import ParseResults, numbers_equal
import pandas as pd


"""
class ParseResults:
    ok: bool
    value: object | None
    error: str | None
"""

def extract_parsed_results(result: ParseResults):
    
    if result.ok:
        return result.value
    
    return None

def compare_pred_ans(id, question, model_resp, gold, pred):
    
    parsed = False
    correct = False
    truncated = model_resp.stop_reason == "max_tokens"
    if pred is not None:
        parsed = True
        
        correct = numbers_equal(pred, gold)
        
    result = {'id': id, 'question': question, 'answer': gold, 'parsed': parsed,'pred': pred, 'correct': correct,
              'truncated': truncated, 'seconds': model_resp.latency_s}
    
    return result

def aggregate_run_stats(run_stats: pd.DataFrame, model, seed, dataset) -> dict:
    
    n = len(run_stats)
    
    total_correct = len(run_stats.loc[run_stats['correct']])
    valid_resp = len(run_stats.loc[run_stats['parsed']])
    valid_resp_rate = float(valid_resp / n)
    accuracy = float(total_correct / n)
    accuracy_on_valid_response = float(total_correct / valid_resp) if valid_resp > 0 else None
    
    total_truncated = len(run_stats.loc[run_stats['truncated']])
    truncation_rate = float(total_truncated / n)
    total_latency = float(run_stats['seconds'].sum())

    
    results = {'model': model, 'seed': seed, 'dataset': dataset, 'n': n, 'valid_response_rate': valid_resp_rate, 'accuracy': accuracy, 'accuracy_on_valid_response': accuracy_on_valid_response,
               'truncation_rate': truncation_rate, 'total_latency': total_latency}
    
    return results