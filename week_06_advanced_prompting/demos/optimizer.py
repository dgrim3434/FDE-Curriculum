from demos.load_data import load_gsm8k
from prompting.parsing import make_number_parser, numbers_equal, normalize_answer, make_text_parser
from prompting.optimizer import build_ngram
from types import SimpleNamespace
from prompting.templates import render
from demos.constants import RESULTS_DIR
from pathlib import Path
from prompting.llm import OllamaClient
from prompting.optimizer import optimizer
from prompting.optimizer import evaluate
import json


N_TRAIN = 20
N_DEV = 80
N_TEST = 200
ROUNDS = 4
K = 3
PATIENCE = 2
ALPHA = 0.1
MAX_WORDS = 150
REQUIRED = ["<answer>"]
SMOKE = False

if SMOKE:
    N_TRAIN = 5
    N_DEV = 10
    N_TEST = 10
    ROUNDS = 1
    K = 2

VERSION = 1       
log_path = RESULTS_DIR / "optimizer.jsonl"



QUESTION_COL = 'question'
GOLD_COL = 'gold'
def estimated_runtime(expected_latency):
    
    calls = (N_TRAIN + N_DEV) + ROUNDS * K * (N_TRAIN + N_DEV) + 3 * N_DEV + N_TEST 
    return calls * expected_latency

score_parser = make_number_parser(tag='answer')

def scorer(text, example):
    
    parsed = score_parser(text)
    gold = example[GOLD_COL]
    
    if not parsed.ok:
        return SimpleNamespace(correct=False, parsed=False)
    
    if numbers_equal(parsed.value, gold):
        return SimpleNamespace(correct=True, parsed=True)
    
    return SimpleNamespace(correct=False, parsed=True)
   
def buid_banned(dev, test):
    dev_text = " ".join(dev[QUESTION_COL])
    test_text = " ".join(test[QUESTION_COL])
    
    norm_dev = normalize_answer(dev_text)
    norm_test = normalize_answer(test_text)
    
    dev_set = build_ngram(norm_dev)
    test_set = build_ngram(norm_test)
    
    return dev_set | test_set

def build_metaprompt(best_prompt, train_result, train):
    
    by_id = {ex['id']: ex for ex in train}
        
    failed_examples = []
    
    for id, result in train_result.per_example.items():
        
        if not result['correct']:
            failed_examples.append({'question': by_id[id][QUESTION_COL], 'output': result['response'], 'gold': by_id[id][GOLD_COL]})
    
    exam = min(len(failed_examples), 5)
    
    prompt_examples = failed_examples[:exam]
    
    return render('system/optimizer.j2', max_words=MAX_WORDS) , render("user/optimizer.j2", prompt=best_prompt, examples=prompt_examples, max_words=MAX_WORDS)


task_llm = OllamaClient(model="llama3.2:3b")
proposer_llm = OllamaClient(model="qwen2.5:7b")

def add_ids(data, val="train"):
    
    data = data.to_dict("records")
    
    for i, row in enumerate(data):
        
        row['id'] = f'{val}-{i}'
    
    return data


def run_optimization():
    
    Path(log_path).unlink(missing_ok=True)
    
    train, dev, test = load_gsm8k(train=N_TRAIN, dev=N_DEV, test=N_TEST, seed=42)
    banned_ngrams = buid_banned(dev, test)
    
    train = add_ids(train, val='train')
    dev = add_ids(dev, val='dev')
    test = add_ids(test, val = 'test')
    
    BASELINE_PROMPT = """You are a math solving agent. You will recieve a math problem and must solve it. When you have your answer respond with the format <answer>YOUR ANSWER</answer>"""
    prompt_parser = make_text_parser(tag="prompt")
    diagnosis_parser = make_text_parser(tag = "diagnosis")
    best, reason = optimizer(task_llm=task_llm, proposer_llm=proposer_llm, baseline_prompt=BASELINE_PROMPT, train=train, dev=dev, scorer=scorer, build_metaprompt=build_metaprompt, prompt_parser=prompt_parser,
              diagnosis_parser=diagnosis_parser, required=REQUIRED, max_words=MAX_WORDS, banned_ngrams=banned_ngrams, rounds=ROUNDS, k=K, alpha=ALPHA, patience=PATIENCE, log_path=log_path)
    
    
    results = evaluate(llm=task_llm, system_prompt=best, examples=test, scorer=scorer, cache={})
    
    print(f"Stop Reason: {reason}\n")
    print(f"Final Prompt:\n {best}\n")
    print(f"Test Accuracy: {results.accuracy}\n")
    print(f"Truncation Rate: {results.truncation_rate}\n")
    print(f"Average Input Tokens: {results.avg_input_tokens}\n")
    print(f"Average Parser Rate: {results.valid_parse_rate}")
    print(f"Average Latency: {results.avg_latency}")

    final = {'stop_reason': reason, 'Final_prompt': best, 'Test_Accuracy': results.accuracy, 'Truncation_Rate': results.truncation_rate, 'Avg_input_tokens': results.avg_input_tokens,
             'Avg_Parsed_Rate': results.valid_parse_rate, 'Avg_latency': results.avg_latency}
    
    (RESULTS_DIR / "optimizer_test.json").write_text(json.dumps(final, indent=2), encoding='utf-8')


if __name__ == "__main__":
    run_optimization()