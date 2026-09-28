from dataclasses import dataclass
from prompting.llm import LLMResponse
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from prompting.extraction import BudgetExceeded
import re



TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=0)

cot_template = env.get_template("system/cot.j2")
error_template = env.get_template("user/invalid_output_error_log.j2")


@dataclass(frozen=True)
class CoTResult:
    method: str
    format_ok: bool
    answer: int | str | None
    valid: bool
    reason: str
    thought: str | None
    attempts: int
    total_cost: float
    error_log: list
    raw: LLMResponse
    

def generate_cot_system(system_prompt, examples = None) -> str:
    
    if examples is None or len(examples) == 0:
        return cot_template.render(system_prompt=system_prompt)
    
    if not isinstance(examples, list):
        raise ValueError("ERROR: examples must be a list")
    if not isinstance(examples[0], dict):
        raise ValueError("ERROR: examples must be a list of dictionaries")
    
    return cot_template.render(system_prompt=system_prompt, examples=examples)

def chain_of_thought(llm, system_prompt: str, user_prompt: str,budget: BudgetExceeded, examples: list[dict] = None, max_retries= 4, max_tokens = 512, COT=True) -> CoTResult:
    
    total_spend = 0
    history = []
    curr_max = max_tokens
    
    if COT:
        system_prompt = generate_cot_system(system_prompt=system_prompt, examples=examples)
    
    history.append({'role': 'user', 'content': user_prompt})
    attempt = 0
    error_log = []
    
    while attempt < max_retries:
        
        feedback = []
        
        try:
            budget.check()
        except BudgetExceeded:
            return CoTResult(method=None, format_ok=False, answer=None, valid=False, reason='budget', thought=None, attempts=attempt, total_cost=total_spend, error_log=error_log, raw=None)
        
        resp = llm.complete(system = system_prompt, messages = history, temperature=0.0, max_tokens = curr_max)
        attempt += 1
        budget.add(resp.cost)
        total_spend += resp.cost
        
        if resp.stop_reason == 'refusal':
            return CoTResult(method=None, format_ok=False, answer=None, valid=False, reason='refusal', thought=None, attempts=attempt, total_cost=total_spend, error_log=error_log, raw=resp)
        
        if resp.stop_reason == 'max_tokens':
            while resp.stop_reason == 'max_tokens' and attempt < max_retries:
                
                try:
                    budget.check()
                except BudgetExceeded:
                    return CoTResult(method=None, format_ok=False, answer=None, valid=False, reason='truncated', thought=None, attempts=attempt, total_cost=total_spend, error_log=error_log, raw=None)
                                
                curr_max = int(curr_max * 1.5)
                
                resp = llm.complete(system=system_prompt, messages=history, temperature=0.0, max_tokens=curr_max)
                budget.add(resp.cost)
                total_spend += resp.cost
                attempt += 1
        
        if resp.stop_reason == "end_turn":

            ans, think, method, errors = parse_output(resp.text)
            error_log += errors
            feedback += errors
            
            
            if ans is not None:
                formating = len(errors) == 0
                return CoTResult(method=method, format_ok = formating, answer=ans, valid=True, reason='ok',thought=think, attempts=attempt, total_cost=total_spend, error_log=error_log, raw=resp)
        
        history.append({'role': 'assistant', 'content': resp.text})
        error_prompt = error_template.render(error_logs=feedback)
        history.append({'role': 'user', 'content': error_prompt})
    
    if resp.stop_reason == "max_tokens":
        return CoTResult(method=None, format_ok = False, answer=None, valid=False, reason='truncated',thought=None, attempts=attempt, total_cost=total_spend, error_log=error_log, raw=resp)
    else:
        return CoTResult(method=None, format_ok = False, answer=None, valid=False, reason='exhausted',thought=None, attempts=attempt, total_cost=total_spend, error_log=error_log, raw=resp)      


def parse_output(response):
    errors = []
    flags = re.DOTALL | re.IGNORECASE
    method = None
    match = None
    RE_TAG = r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>"
    RE_OPEN = r"<\s*answer\s*>\s*([^<\n]*)"
    RE_MARKET = r"(?:####|final answer\s*(?:is)?\s*[:=]?|the answer is|answer\s*:)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)"
    
    answers = re.findall(RE_TAG, response, flags=flags)
    
    answer = None
    if len(answers) == 0:
        
        errors.append("Formating error no <answer></answer> brackets where found encasing an answer")
        
        answers = re.findall(RE_OPEN, response, flags=flags)
        
        if len(answers) == 0:
            errors.append("No opening answer bracket detected within the response")
            
            answers = re.findall(RE_MARKET, response, flags=flags)
            if len(answers) == 0:
                errors.append("Could not find any trace of an answer within the response")
                return None, None, None, errors
            
            method = "marker"
            answer = extract_answer_from_list(answers, errors)
            matches = list(re.finditer(RE_MARKET, response, flags))
            match = matches[-1]
            
        else:
            
            method = "open_tag"
            
            answer = extract_answer_from_list(answers, errors)
            matches = list(re.finditer(RE_OPEN, response, flags))
            match = matches[-1]
    else:
        method = "tag"
        answer = extract_answer_from_list(answers, errors)
        matches = list(re.finditer(RE_TAG, response, flags))
        match = matches[-1]
    
    try:
        answer = clean_answer(answer)
    
    except ValueError:
        errors.append(f"Answer Must be of type float. Invalid output: {answer}")
        return None, None, None, errors
    
    
    thinking = re.findall(r"<\s*thinking\s*>(.*?)<\s*/\s*thinking\s*>", response, flags=flags)
    
    if len(thinking) == 0:
        errors.append("No Thinking blocks where found in the response")
        if match is None:
            return answer, None, False, errors
        
        thought = response[:match.start()].strip()
    else:
        thought = extract_answer_from_list(thinking, errors, answer=False)
    
    
    return answer, thought, method, errors
    

def extract_answer_from_list(answers, errors, answer=True):
    
    if len(answers) > 1 and answer:
        errors.append("Multiple answer brackets where found within response.")
    return answers[-1]


def clean_answer(ans):
    
    return float(re.sub(r"[,$\s]", "", ans)) 
    
    