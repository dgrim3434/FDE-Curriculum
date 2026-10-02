from prompting.llm import OllamaClient
from prompting.parsing import make_number_parser, numbers_equal
from prompting.consistency import self_consistency
import pandas as pd
import json
from demos.constants import RESULTS_DIR
from prompting.templates import render
from demos.load_data import load_gsm8k
from prompting.llm import LLMResponse
from prompting.llm import FakeLLM
import math

MODEL = "llama3.2:3b"

llm = OllamaClient(model=MODEL)
QUESTION_COL = 'question'
GOLD_COL = 'gold'

def run_self_consistency():
    
    test, _, _ = load_gsm8k(train=100)
    
    system = render("system/baseline_math_solver.j2")
    parser = make_number_parser(tag='answer')
    
    results = []
    
    for idx in range(len(test)):
        
        row = test.iloc[idx]
        
        question = row[QUESTION_COL]
        gold = row[GOLD_COL]
        
        user = render("user/baseline_math_solver.j2", question=question)
        user_prompt = [{'role': 'user', 'content': user}]
        con_resp = self_consistency(llm=llm, system=system, messages=user_prompt, parser=parser)
        base_resp = self_consistency(llm=llm, system=system, messages=user_prompt, parser=parser, n=1, temperature=0.0, min_valid=1)
        
        top_ks = extract_top_k(con_resp, system=system, user=user_prompt, parser=parser, top_ks = [1,3])
        
        res_1 = top_ks[1]
        res_3 = top_ks[3]
        
        results.append(caclulate_results(idx, gold, base_resp, con_resp, res_1, res_3))
    
    df = pd.DataFrame(results)
    
    df.to_csv(RESULTS_DIR / "self_consistency.csv", index=False, encoding='utf-8')
    
    results = generate_summary(df)
    
    (RESULTS_DIR / "self_consistency.json").write_text(json.dumps(results, indent=2), encoding='utf-8')

def generate_summary(df):
    
    results = {}
    
    
    # baseline Accuracy
    base_correct = len(df.loc[df['base_correct']])
    base_accuracy = float(base_correct / len(df))
    base_s_per_question = float(df['base_latency'].sum() / len(df))
    base_s_per_correct = float(df['base_latency'].sum() / base_correct) if base_correct > 0 else None
    base_parse_failed_r = float(len(df.loc[df['base_parsed']]) / len(df))
    base_trun_rate = float(len(df.loc[df['base_truncated']]) / len(df))
    
    results['baseline'] = {'accuracy': base_accuracy, 'sec_per_question': base_s_per_question, 'sec_per_correct': base_s_per_correct, 'parse_rate': base_parse_failed_r, 'truncation_rate': base_trun_rate}
    results['by_n'] = per_n_results(df)
    
    calibration, coverage = caclulate_calibration(df)
    results['calibration'] = calibration
    results['coverage'] = coverage
    
    return results

def caclulate_calibration(df):
    
    df['share'] = df['n5_vote_share_all'].round(1)
    
    shares = df.groupby('share')['n5_correct'].agg(['count', 'mean']).reset_index()
    results = []
    
    for idx in range(len(shares)):
        
        row = shares.iloc[idx]
        results.append({'vote_share': row['share'], 'n': int(row['count']), 'accuracy': float(row['mean'])})
    
    coverage = []
    
    cov = 0
    accuracy = 0
    thresholds = [1, .8, .6]
    
    #shares['total_correct'] = shares['count'] * shares['mean']
    for t in thresholds:
        t_cov = shares.loc[shares['share'] >= t]['count'].sum()
        
        mask = df['share'] >= t
        
        t_acc = df.loc[mask, 'n5_correct'].mean()
        
        coverage.append({'threshold': t, 'coverage': int(t_cov), 'accuracy': float(t_acc)})
    
    return results, coverage
        

def per_n_results(df, ks = ['1', '3', '5']):
    results = {}
    
    for k in ks:
        results[k] = {}
        
        correct = len(df.loc[df[f'n{k}_correct']])
        results[k]['accuracy'] = float(correct / len(df))
        tot_seconds = df[f'n{k}_latency'].sum()
        results[k]['sec_per_question'] = float(tot_seconds / len(df))
        results[k]['sec_per_correct'] = float(tot_seconds / correct)
        
        valid_rate = float(len(df.loc[df[f'n{k}_status'] == 'ok']) / len(df))
        insuf_rate = float(len(df.loc[df[f'n{k}_status'] == 'insufficient_samples']) / len(df))
        tie_rate = float(len(df.loc[df[f'n{k}_status'] == 'tie']) / len(df))
        
        results[k]['valid_rate'] = valid_rate
        results[k]['insufficient_rate'] = insuf_rate
        results[k]['tie_rate'] = tie_rate
        
    return results
     
def caclulate_results(id, gold, base_resp, con_resp, res_1, res_3) -> dict:
    
    base_correct = numbers_equal(gold, base_resp.answer)
    base_parsed = True
    if base_resp.status == 'insufficient_samples':
        base_parsed = False
    base_truncated = False
    if base_resp.stop_reasons[0] == 'max_tokens':
        base_truncated = True 
        
        
    return {'id': id, 'gold': gold, 'base_answer': base_resp.answer, 'base_correct': base_correct, 'base_parsed': base_parsed, 'base_truncated': base_truncated,
            'n5_answer': con_resp.answer, 'n5_correct': numbers_equal(gold, con_resp.answer), 'n5_status': con_resp.status, 'n3_answer': res_3.answer, 'n3_correct': numbers_equal(gold, res_3.answer),
            'n3_status': res_3.status, 'n1_answer': res_1.answer, 'n1_correct': numbers_equal(gold, res_1.answer), 'n1_status': res_1.status, 'n5_vote_share_all': con_resp.vote_share_all,
            'n5_vote_share_valid': con_resp.vote_share_valid, 'n5_margin': con_resp.margin, 'n5_entropy': con_resp.entropy, 'n5_n_valid': con_resp.n_valid, 'n5_n_distinct': len(con_resp.distribution),
            'base_latency': base_resp.latency_s, 'n5_latency': con_resp.latency_s, 'n3_latency': res_3.latency_s, 'n1_latency': res_1.latency_s}


def extract_top_k(resps, system, user, parser, top_ks: list = [1,3]):
    
    results = {}
    for k in top_ks:
        
        first_k = resps.llm_resp[:k]
        
        res_k = self_consistency(FakeLLM(first_k), system=system, messages=user, parser=parser, n=k, min_valid= k -1 if k -1 > 0 else 0)
        
        results[k] = res_k
    
    return results
    
def significance_check(path):
    
    df = pd.read_csv(RESULTS_DIR / path, encoding='utf-8')
    b, c = 0,0
    for idx in range(len(df)):
        
        row = df.iloc[idx]
        base_correct = row['base_correct']
        cons_correct = row['n5_correct']
        
        if base_correct and not cons_correct:
            b +=1
        if cons_correct and not base_correct:
            c += 1
    
    denom = b + c
    
    num = sum([math.comb(denom, i) for i in range(c, denom + 1)])
    den = 2 ** denom
    
    return num / den, b, c

if __name__ == "__main__":
    
    
    res, b, c = significance_check("self_consistency.csv")
    print(print(f"B: {b}, C: {c}, Significance: {res * 2}"))