from prompting.llm import AnthropicClient
from prompting.chain_of_thought import chain_of_thought
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from experiments.load_data import load_gsm8k
import pandas as pd
import json
from experiments.CoT.constants import TEMPLATE_DIR, RESULTS_DIR, QUESTION_COLUMN, GOLDEN_COLUMN, ANSWER_COLUMN, compute_result, MAX_TOKENS, SEED, compute_summary
from prompting.budget import BudgetExceeded



llm = AnthropicClient()

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=0)

llm = AnthropicClient()

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=0)

system_template = env.get_template("system/math_solver.j2")
user_template = env.get_template("user/math_solver_CoT.j2")

SEED = 42

def run_math_solver():
    
    train, test = load_gsm8k(training_size= 3, testing_size=50, seed=SEED)
    
    examples = generate_examples(train)
    
    
    system_prompt = system_template.render()
    
    results = []
    budget = BudgetExceeded(budget=1.0)
    
    for idx in range(len(test)):
        
        row = test.iloc[idx]
        
        question = row[QUESTION_COLUMN]
        gold = row[GOLDEN_COLUMN]
        
        user_prompt = user_template.render(query=question)
        resp = chain_of_thought(llm, system_prompt, user_prompt, budget, max_tokens=MAX_TOKENS, examples=examples)
        
        result = compute_result(resp, idx, question, gold)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    df.to_csv(RESULTS_DIR / "math_chain_of_though.csv", index=False, encoding='utf-8')
    
    summary = compute_summary(df, examples=0)
    
    (RESULTS_DIR / "math_chain_of_thought.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')

def generate_examples(df):
    results = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        result = {}
        result['question'] = row[QUESTION_COLUMN]
        thinking = row[ANSWER_COLUMN]
        thinking = thinking.replace("<<", " ")
        thinking = thinking.replace(">>", " ")
        result['thinking'] = thinking
        result['answer'] = row[GOLDEN_COLUMN]
        
        results.append(result)
    
    return results
        

    
if __name__ == "__main__":
    run_math_solver()       

