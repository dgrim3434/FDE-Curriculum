from llm import AnthropicClient
from function_calling.tools import TOOL_MAPPING
from function_calling.loop import run_tool_loop,extract_with_tool
import json
from data.loaders import load_conll2003
from extraction.prompts import build_prompt
from extraction.schemas import Extraction
from pathlib import Path
from extraction.evaluation import to_pairs, prf
from templates.template import render

llm = AnthropicClient()
RESULTS = Path(__file__).resolve().parent.parent / "results"


QUESTIONS = [
    "Find a sentence that mentions Germany and tell me what it says.",
    "How many of the first sentences about cricket mention a person? Name them.",
    "Find the sentence at id -100"
]
def force_tools():
    
    results = []
    for question in QUESTIONS:
    
        res = run_tool_loop(client=llm, user_message=question, registry=TOOL_MAPPING)
        result = {'question': question, 'steps': res.steps, 'hit_max_steps': res.hit_max_steps, 'tool_calls': res.tool_calls, 'final_text': res.final_text}
        print(result)
        results.append(result)
    
    (RESULTS / "sample_tool_questions.json").write_text(json.dumps(results, indent=2), encoding='utf-8')


class CountingClient:
    
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
    
    def complete_with_tools(self, **kwargs):
        resp = self.inner.complete_with_tools(**kwargs)
        self.calls += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        return resp

def evaluate():
    
    data = load_conll2003(n=50)
    model_log = {}
    results = []
    first_pass = 0
    retry_needed = 0
    failed = 0
    labels = []
    gold = []
    for idx in range(len(data)):
        
        text = data[idx]['sentence']
        
        sys, _ = build_prompt(text, Extraction)
        
        prompt = render("user/native_tool.j2", text=sys, sentence=text)
        
        counter = CountingClient(llm)
        res = extract_with_tool(client=counter, text=prompt)
        
        if counter.calls == 1:
            first_pass += 1
        elif counter.calls > 1 and res is not None:
            retry_needed += 1
        else:
            failed += 1
        
        if res is not None:
            labels.append(to_pairs(res.model_dump()))
            gold.append(set(data[idx]['entities']))
        
        result = {'id': data[idx]['id'], 'calls': counter.calls, 'input_tokens': counter.input_tokens, 'output_tokens': counter.output_tokens}
        
                      
        if res is None:
            result['valid'] = False
            result['pairs'] = None
        else:
            result['valid'] = True
            result['pairs'] = [list(t) for t in to_pairs(res.model_dump())]
        
        results.append(result)
    
    _prf = prf(gold, labels)
    
    model_log['model'] = llm.model
    model_log['f1'] = _prf[2]
    model_log['precision'] = _prf[0]
    model_log['recall'] = _prf[1]
    model_log['first_pass'] = first_pass
    model_log['pass_after_retry'] = retry_needed
    model_log['failed'] = failed
    
    tot_input = sum(x['input_tokens'] for x in results)
    tot_output = sum(x['output_tokens'] for x in results)
    
    model_log['output_tokens'] = tot_output
    model_log['input_tokens'] = tot_input
    model_log['sentences'] = results
    
    (RESULTS / "native_tool_use.json").write_text(json.dumps(model_log, indent=2), encoding='utf-8')

if __name__ == "__main__":
    #force_tools()
    evaluate()
