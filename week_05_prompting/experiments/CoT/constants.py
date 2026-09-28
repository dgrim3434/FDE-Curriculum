from pathlib import Path
from prompting.chain_of_thought import CoTResult
import pandas as pd
TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"

QUESTION_COLUMN = "question"
ANSWER_COLUMN = "answer"
GOLDEN_COLUMN = "gold"
MAX_TOKENS = 512
SEED = 42

def compute_result(resp: CoTResult, idx, question, gold) -> dict:
    result = {'idx': idx, 'question': question, 'gold': gold, 'valid': resp.valid, 'reason': resp.reason, 'method': resp.method, 'format_ok': resp.format_ok, 'attempts': resp.attempts,
            'input_tokens': resp.raw.input_tokens, 'output_tokens': resp.raw.output_tokens, 'cost': resp.total_cost, 'error_log': "| ".join(resp.error_log)}
            
    if resp.valid:
        pred = resp.answer
        result['pred'] = pred
        correct = abs(pred - gold) < 1e-6
        result['correct'] = correct
                
        if resp.thought is not None:
            thought = resp.thought.replace("\n", " ")
                
        result['thought'] = thought
    else:
        result['pred'] = None
        result['correct'] = False
        result['thought'] = None
    
    return result

def compute_summary(df, examples=0) -> dict:
    
    correct_preds = len(df.loc[df['correct']])
    accuracy = float(correct_preds / len(df))
    valid_rate = float(len(df.loc[df['valid']]) / len(df))
    accuracy_when_valid = float(len(df.loc[(df['correct']) & (df['valid'])]) / len(df.loc[df['valid']]))
    correct_format = float(len(df.loc[df['format_ok']]) / len(df))
    total_cost = float(df['cost'].sum())
    cost_per_correct = float(total_cost / correct_preds)
    methods = {str(k): int(v) for k, v in df['method'].fillna('none').value_counts().items()}
    
    summary = {'n': len(df), 'seed': SEED, 'max_tokens': MAX_TOKENS, 'examples': examples, 'accuracy': accuracy, 'valid_rate': valid_rate, 'accuracy_when_valid': accuracy_when_valid,
               'format_ok_rate': correct_format, 'average_attempts': float(df['attempts'].mean()), 'total_cost': total_cost, 'cost_per_correct': cost_per_correct, 'methods': methods}
    
    return summary