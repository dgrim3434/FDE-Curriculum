from prompting.llm import AnthropicClient
from prompting.classify import classify_intent
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from experiments.load_data import load_banking77
import pandas as pd
import json
from experiments.plots import plot_confusion_matrix, top_confusions
from experiments.constants import TEMPLATE_DIR, RESULTS_DIR, ARTIFACT_DIR

llm = AnthropicClient()

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=0)

system_template = env.get_template("system/classify_system.j2")
user_template = env.get_template("user/classify_one_shot.j2")

RESULTS_DIR.mkdir(exist_ok=True)


SEED = 42

def run_zero_shot_baseline():
    
    train, test = load_banking77(training_size=1000, testing_size=50, seed=SEED)
    
    labels = train['label_name'].unique().tolist()
    
    system_prompt = system_template.render(categories=labels)
    
    results = []
    
    for idx in range(len(test)):
        
        row = test.iloc[idx]
        label = row['label_name']
        query = row['text']
        
        user_prompt = user_template.render(query=query)
        
        out = classify_intent(llm, system_prompt=system_prompt, user_prompt=user_prompt, valid_labels= set(labels))
        
        # return ClassificationResult(label=None, valid=False, raw=resp)
        # return LLMResponse(text=text, stop_reason=r.stop_reason, input_tokens= r.usage.input_tokens, output_tokens=r.usage.output_tokens)
        pred, valid, response = out.label, out.valid, out.raw
        is_correct = label == pred if valid else False
        
        results.append({'row_id': idx, 'text': query, 'true_label': label, 'pred_label': pred, 'valid': valid, 'correct': is_correct, 'stop_reason': response.stop_reason, 'input_tokens': response.input_tokens, 'output_tokens': response.output_tokens, 'cost': response.cost})
    
    df = pd.DataFrame(results)
    
    df.to_csv(RESULTS_DIR / "zero_shot_baseline.csv", index=False, encoding='utf-8')

    accuracy = float(len(df.loc[df['correct']]) / len(df))
    total_cost = float(df['cost'].sum())
    invalid_count = int(len(df.loc[df['valid'] == False]))
    cost_per_correct = total_cost / len(df.loc[df['correct']]) if len(df.loc[df['correct']]) > 0 else None
    
    summary = {'n': int(len(df)), 'seed': SEED, 'model': 'base', 'accuracy': accuracy, 'invalid_count': invalid_count, 'total_cost_usd': total_cost, 'cost_per_correct': cost_per_correct}
    (RESULTS_DIR / "zero_shot_baseline.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')
    
    plot_confusion_matrix(df, ARTIFACT_DIR / "baseline_confusion_matrix.png", 'Zero-shot baseline (n=50)')
    print(top_confusions(df))

if __name__ == "__main__":
    run_zero_shot_baseline()