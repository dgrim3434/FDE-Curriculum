from prompting.templates import render
from prompting.llm import OllamaClient
from demos.constants import RESULTS_DIR, extract_parsed_results, compare_pred_ans, aggregate_run_stats
from demos.load_data import load_gsm8k
from prompting.parsing import make_number_parser
import pandas as pd
import json

SEED = 42
QUESTION_COL = 'question'
GOLD_COL = 'gold'
def run_baseline_model_comparison():
    
    model_1b = OllamaClient(model="llama3.2:1b")
    model_3b = OllamaClient(model="llama3.2:3b")
    
    system_prompt = render("system/baseline_math_solver.j2")
    
    parser = make_number_parser(tag="answer")
    
    _, _, test = load_gsm8k(test=50)
    
    results_1b = []
    results_3b = []
    
    for idx in range(len(test)):
        
        row = test.iloc[idx]
        
        question = row[QUESTION_COL]
        
        gold = row[GOLD_COL]
        
        user_prompt = [{'role': 'user', 'content': render("user/baseline_math_solver.j2", question=question)}]
        
        resp_1b = model_1b.complete(system_prompt, user_prompt, temperature=0, seed=42, max_tokens=1024)
        resp_3b = model_3b.complete(system_prompt, user_prompt, temperature=0, seed=42, max_tokens=1024)
        
        parse_1b = parser(resp_1b.text)
        parse_3b = parser(resp_3b.text)
        
        ans_1b = extract_parsed_results(parse_1b)
        ans_3b = extract_parsed_results(parse_3b)
        
        result_1b = compare_pred_ans(idx, question, resp_1b, gold, ans_1b)
        result_3b = compare_pred_ans(idx, question, resp_3b, gold, ans_3b)
        
        results_1b.append(result_1b)
        results_3b.append(result_3b)
    
    df_1b = pd.DataFrame(results_1b)
    df_3b = pd.DataFrame(results_3b)
    
    df_1b.to_csv(RESULTS_DIR / "LLAMA_1B_baseline.csv", index=False, encoding='utf-8')
    df_3b.to_csv(RESULTS_DIR / "LLAMA_3B_baseline.csv", index=False, encoding='utf-8')
    
    agg_1b = aggregate_run_stats(df_1b, model_1b.model, SEED, "GSMK8")
    agg_3b = aggregate_run_stats(df_3b, model_3b.model, SEED, "GSMK8")
    
    (RESULTS_DIR / "LLAMA_1B_baseline.json").write_text(json.dumps(agg_1b, indent=2), encoding='utf-8')
    (RESULTS_DIR / "LLAMA_3B_baseline.json").write_text(json.dumps(agg_3b, indent=2), encoding='utf-8')

if __name__ == "__main__":
    
    run_baseline_model_comparison()