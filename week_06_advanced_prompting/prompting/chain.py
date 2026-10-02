from dataclasses import dataclass
from collections.abc import Callable
from copy import deepcopy

@dataclass
class Step:
    name: str
    build_prompt: Callable[[str], str] | None
    parser: Callable[[str], str] | None
    output_key: str
    next: str | Callable[[str], str] | None
    temperature: float | None
    max_tokens: int | None
    max_repairs: int | None
    system: str | None
    fn: Callable[[str], str] | None

@dataclass
class StepRecord:
    name: str
    prompt: str | None
    raw: str | None
    value: float | str 
    attempts: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    error: str | None
    next: str | None

@dataclass
class ChainResult:
    status: str
    output: str | float | None
    state: dict
    trace: list[StepRecord] | list
MAX_TOKENS = 2048

def run_chain(steps: dict, start, initial_state, llm, max_steps=20):
    
    current = start
    state = deepcopy(initial_state)
    
    names = set()
    next_steps = set()
    for name, step in steps.items():
        names.add(name)
        
        if isinstance(step.next, str):
            next_steps.add(step.next)
    
    if len(next_steps - names) > 0:
        raise ValueError(f"ERROR: All string next values must correspond to a step within the steps dictionary. The following steps where not found as keys: {next_steps - names}")
    
    if current not in steps:
        raise ValueError('ERROR Initial step must be in the provided state')
    
    last_state = None
    complete_steps = []
    step_count = 0
    while current is not None:
        step_count += 1
        last_state = current
        
        if current not in steps:
            return ChainResult(status='failed', output=None, state=state, trace=complete_steps)
        
        step = steps[current]
        succeeded = False
        if step.fn is not None:
            
            result = step.fn(state)
            state[step.output_key] = result
            next = step.next(state) if callable(step.next) else step.next
            complete_steps.append(StepRecord(name=current, prompt=None, raw=None, value=result, attempts=1, input_tokens=0, output_tokens=0, latency_s=0, error=None, next=next))
            current = next
            succeeded = True
            
        else:
            prompt = step.build_prompt(state)
            #breakpoint()
            messages = [{'role': 'user', 'content': prompt}]
            curr_max_tokens = step.max_tokens
            tot_input_toks = 0
            tot_output_toks = 0
            tot_latency = 0.0
            for attempt in range(1, step.max_repairs + 2):
                
                update = None
                resp = llm.complete(system=step.system, messages = messages, temperature = step.temperature, max_tokens = curr_max_tokens)
                
                tot_input_toks += resp.input_tokens
                tot_output_toks += resp.output_tokens
                tot_latency += resp.latency_s
                
                if resp.stop_reason == 'max_tokens':
                    
                    if attempt == step.max_repairs:
                        next = step.next(state) if callable(step.next) else step.next
                        complete_steps.append(StepRecord(name=current, prompt=prompt, raw=resp.text, value = None, attempts=attempt, input_tokens=tot_input_toks,
                                                         output_tokens=tot_output_toks, latency_s=tot_latency, error="Truncation", next=next))
                        
                    curr_max_tokens = int(min(curr_max_tokens * 1.5, MAX_TOKENS))
                    continue
                
                elif resp.stop_reason == 'end_turn':
                    
                    parsed = step.parser(resp.text)

                    if parsed.ok == False:
                        update = parsed.error
                    else:
                        
                        state[step.output_key] = parsed.value
                        next = step.next(state) if callable(step.next) else step.next
                        complete_steps.append(StepRecord(name=current, prompt=prompt, raw=resp.text, value = parsed.value, attempts=attempt, input_tokens=tot_input_toks,
                                                         output_tokens=tot_output_toks, latency_s=tot_latency, error=None, next=next))
                        current = step.next(state) if callable(step.next) else step.next
                        succeeded = True
                        break
                
                else:
                    
                    raise ValueError("ERROR: LLM Interface error chain could not be continued")
                
                if attempt == step.max_repairs + 1:
                    next = step.next(state) if callable(step.next) else step.next
                    complete_steps.append(StepRecord(name=current, prompt=prompt, raw=resp.text, value = None, attempts=attempt, input_tokens=tot_input_toks,
                                                    output_tokens=tot_output_toks, latency_s=tot_latency, error="Parser Error", next=next))
                    
                    return ChainResult(status='failed', output=None, state=state, trace=complete_steps)
                
                #breakpoint()
                messages.append({'role': 'assistant', 'content': resp.text})
                messages.append({'role': 'user', 'content': update})  

            if not succeeded:
                return ChainResult(status='failed', output=None, state=state, trace=complete_steps)
        
        if step_count >= max_steps:
            return ChainResult(status='failed', output=None, state=state, trace=complete_steps) 
           
    return ChainResult(status='ok', output=state[steps[last_state].output_key], state=state, trace=complete_steps)