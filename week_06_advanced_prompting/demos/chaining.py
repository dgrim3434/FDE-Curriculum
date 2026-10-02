from prompting.chain import run_chain, Step
from prompting.llm import OllamaClient
from demos.load_data import load_hot_qa
from prompting.parsing import make_text_parser, make_label_parser, make_pair_parser, exact_match, f1
from prompting.templates import render
from demos.constants import RESULTS_DIR
import pandas as pd
import json

MODEL = "llama3.2:3b"

llm = OllamaClient(model=MODEL)


def convert_to_paragraphs(state):
    
    titles, sentences = state['context']['title'], state['context']['sentences']
    parts = []
    
    for t, s in zip(titles, sentences):
                
        para ="".join(s)
        
        parts.append(t +": " + para)
    
    return "\n\n".join(parts)

def router_prompt(state):
    
    question = state['question']
    
    return render('user/classification.j2', question=question)

def entity_extraction(state):
    
    question = state['question']
    
    return render('user/extraction.j2', question=question)

def extract_fact_a(state):
    question, entity, text = state['question'], state['entities']['a'], state['paragraphs']
    
    return render('user/fact_finder.j2', question=question, entity=entity, text=text)

def extract_fact_b(state):
    question, entity, text = state['question'], state['entities']['b'], state['paragraphs']
    
    return render('user/fact_finder.j2', question=question, entity=entity, text=text)

def compare(state):
    fact_a = state['fact_a']['fact']
    fact_b = state['fact_b']['fact']
    question = state['question']
    
    return render('user/qa.j2', fact_a=fact_a, fact_b = fact_b, question=question)

def find_bridge(state):
    question = state['question']
    text = state['paragraphs']
    
    return render('user/bridge.j2', question=question, text=text)

def hop2(state):
    question = state['question']
    fact = state['bridge']['bridge']
    text = state['paragraphs']
    
    return render('user/hop2.j2', question=question, fact=fact, text=text)

def route(state):
    
    return 'entities' if state['qtype'] == 'comparison' else 'find_bridge'

def fact_check_a(state):
    source = squash(state['fact_a']['source'])
    
    return source in squash(state['paragraphs'])

def fact_check_b(state):
    source = squash(state['fact_b']['source'])
    
    return source in squash(state['paragraphs'])

def fact_check_bridge(state):
    source = squash(state['bridge']['source'])
    
    return source in squash(state['paragraphs'])
    
def squash(s):
    return " ".join(s.lower().split())
router_examples = [
    {'label': 'comparison', 'description': 'The question compares tow named things (same nationality?, which is older?)'},
    {'label': 'bridge', 'description': 'you must find one peice of information first, then use that information to answer the second question.'}
]       

STEPS = {
    'format_context': Step('fromat_context', None, None, 'paragraphs', 'router', None, None, None, None, convert_to_paragraphs),
    'router': Step('router', router_prompt, make_label_parser(tag='type', allowed=set(['comparison', 'bridge'])), 'qtype', route, 0.0, 512, 1, render('system/classification.j2', examples=router_examples), None),
    'entities': Step('entities', entity_extraction, make_pair_parser(tag1='a', tag2='b'), 'entities', 'extract_a', 0.0, 512, 1, render('system/extraction.j2'),None),
    'extract_a': Step('extract_a', extract_fact_a, make_pair_parser(tag1='fact', tag2='source'), 'fact_a', 'check_a', 0.0, 512, 1, render('system/fact_finder.j2'), None),
    'check_a': Step('check_a', None, None, 'grounded_a', 'extract_b', None, None, None, None, fact_check_a),
    'extract_b': Step('extract_b', extract_fact_b, make_pair_parser(tag1='fact', tag2='source'), 'fact_b', 'check_b', 0.0, 512, 1, render('system/fact_finder.j2'), None),
    'check_b': Step('check_b', None, None, 'grounded_b','compare', None, None, None, None, fact_check_b),
    'compare': Step('compare', compare, make_text_parser(tag='answer'), 'answer', None, 0.0, 512, 1, render('system/qa.j2'), None),
    'find_bridge': Step('find_bridge', find_bridge, make_pair_parser(tag1='bridge', tag2='source'), 'bridge', 'check_bridge', 0.0, 512, 1, render('system/bridge.j2'), None),
    'check_bridge': Step('check_bridge', None, None, 'grounded_bridge', 'hop2', None, None, None, None, fact_check_bridge),
    'hop2': Step('hop2', hop2, make_text_parser(tag='answer'), 'answer', None, 0.0, 1024, 1, render('system/hop2.j2'), None)
}   

def base_prompt(schema):
    
    question = schema['question']
    text = schema['paragraphs']
    
    return render('user/zero_shot.j2', question=question, text=text)
    

BASE = {
    'format_context': Step('fromat_context', None, None, 'paragraphs', 'answerer', None, None, None, None, convert_to_paragraphs),
    'answerer': Step('answerer', base_prompt, make_text_parser(tag='answer'), 'answer', None, 0.0, 1024, 1, render('system/zero_shot.j2'), None)
}
def run_chaining():
    
    val = load_hot_qa()
    start = 'format_context'
    results = []
    
    for id in range(len(val)):
        
        row = val[id]
        
        question = row['question']
        context = row['context']
        
        init_state = {'question': question, 'context': context}
        
        resp_chain = run_chain(steps=STEPS, start=start, initial_state=init_state, llm=llm)
        resp_base = run_chain(steps = BASE, start=start, initial_state=init_state, llm=llm)
        
        result = calculate_results(id, resp_chain=resp_chain, resp_base=resp_base, row=row)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    df.to_csv(RESULTS_DIR / "chaining.csv", index=False, encoding='utf-8')
    
    results = aggregate_results(llm.model, df)
    
    (RESULTS_DIR / "chaining.json").write_text(json.dumps(results, indent=2), encoding='utf-8')
  
  

def aggregate_results(model, df):
    
    chain_em = float(df['chain_em'].mean())
    base_em = float(df['base_em'].mean())
    chain_f1 = float(df['chain_f1'].mean())
    base_f1 = float(df['base_f1'].mean())
    
    router_correct = len(df.loc[df['type'] == df['router_pred']])
    router_accuracy = float(router_correct / len(df))
    
    a_grounded = float(len(df.loc[df['grounded_a'] == True]) / len(df.loc[df['router_pred'] == 'comparison'])) if len(df.loc[df['router_pred'] == 'comparison']) > 0 else 0
    b_grounded = float(len(df.loc[df['grounded_b'] == True]) / len(df.loc[df['router_pred'] == 'comparison'])) if len(df.loc[df['router_pred'] == 'comparison']) > 0 else 0
    bridge_grounded = float(len(df.loc[df['grounded_bridge'] == True]) / len(df.loc[df['router_pred'] == 'bridge'])) if len(df.loc[df['router_pred'] == 'bridge']) > 0 else 0
    
    avg_input_tok = float(df['total_input_tokens'].mean())
    avg_out_tok = float(df['total_output_tokens'].mean())
    avg_latency_chain = float(df['total_latency'].mean())
    avg_latency_base = float(df['base_latency'].mean())
    
    return {'Model': model, 'Chain_em': chain_em, 'Base_em': base_em, 'Chain_f1': chain_f1, 'Base_f1': base_f1, 'router_accuracy': router_accuracy, 'grounded_a': a_grounded,
            'grounded_b': b_grounded, 'grounded_bridge': bridge_grounded, 'avg_input_tokens': avg_input_tok, 'avg_output_tokens': avg_out_tok, 'avg_latency_chain': avg_latency_chain,
            'avg_latency_base': avg_latency_base}
    
def calculate_results(id, resp_chain, resp_base, row):
    
    chain_em = None
    base_em = None
    chain_f1 = None
    base_f1 = None
    
    if resp_chain.status == 'ok':
        chain_em = exact_match(resp_chain.output, row['answer'])
        chain_f1 = f1(resp_chain.output, row['answer'])
    if resp_base.status == 'ok':
        base_em = exact_match(resp_base.output, row['answer'])
        base_f1 = f1(resp_base.output, row['answer'])
    
    failed_step = resp_chain.trace[-1].name if resp_chain.status == 'failed' else None
    grounded_a = resp_chain.state['grounded_a'] if 'grounded_a' in resp_chain.state else None
    grounded_b = resp_chain.state['grounded_b'] if 'grounded_b' in resp_chain.state else None
    grounded_bridge = resp_chain.state['grounded_bridge'] if 'grounded_bridge' in resp_chain.state else None
    
    repairs = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0
    
    for step in resp_chain.trace:
        repairs += step.attempts -1
        total_input_tokens += step.input_tokens
        total_output_tokens += step.output_tokens
        total_latency += step.latency_s
        
    
    result = {'id': id, 'question': row['question'], 'gold': row['answer'], 'type': row['type'], 'chain_answer': resp_chain.output, 'baseline_answer': resp_base.output,
              'chain_em': chain_em, 'base_em': base_em, 'chain_f1': chain_f1, 'base_f1': base_f1, 'chain_status': resp_chain.status, 'failed_step': failed_step, 'router_pred': resp_chain.state['qtype'],
              'grounded_a': grounded_a, 'grounded_b': grounded_b, 'grounded_bridge': grounded_bridge, 'chain_calls': len(resp_chain.trace), 'total_repairs': repairs,
              'total_input_tokens': total_input_tokens, 'total_output_tokens': total_output_tokens, 'total_latency': total_latency, 'base_latency': resp_base.trace[-1].latency_s
              }

    return result


if __name__ == "__main__":
    run_chaining()
