from dataclasses import dataclass
from copy import deepcopy
from prompting.parsing import Action, Finish, ReActParserError, normalize_label
from prompting.templates import render



@dataclass
class ReActStep:
    thought: str | None
    tool: str | None
    arg: str | None
    observation: str | None
    raw: str
    error: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    attempts: int

@dataclass
class ReActResults:
    answer: str | None
    status: str
    used_fallback: bool
    steps: list

MAX_TOKENS = 2048




def run_react(llm, system, messages, parser, tools, max_steps, allowed_repeats: set, max_format_errors=3, max_tokens=1024, num_ctx = 8192, ctx_margin =256) -> ReActResults:
    
    step = 0
    history = [{'role': 'user', 'content': messages}]
    steps = []
    seen = {}
    tool_labels = set(tools.keys())
    
    while step < max_steps:
        
        step += 1
        input_tokens = 0
        output_tokens = 0
        latency = 0
        curr_history = deepcopy(history)
        
        for attempt in range(1, max_format_errors + 1):
            
            resp = llm.complete(system=system, messages=curr_history, max_tokens=max_tokens, stop=['<observation>'])
            input_tokens += resp.input_tokens
            output_tokens += resp.output_tokens
            latency += resp.latency_s
            
            if resp.input_tokens + resp.output_tokens >= num_ctx - ctx_margin:
                
                steps.append(ReActStep(thought=None, tool=None, arg=None, observation=None, raw=resp.text, error=None, input_tokens=input_tokens, output_tokens=output_tokens, latency_s=latency, attempts=attempt))
                
                ans, step = fallback_trigger(llm, history=history, max_tokens=max_tokens, parser=parser)
                
                steps.append(step)
                
                return ReActResults(answer=ans, status='context_full', used_fallback=True, steps=steps)
                
                
            if resp.stop_reason == "end_turn" or resp.stop_reason == "max_tokens":
                
                parsed = parser(resp.text)
                
                if isinstance(parsed, Finish):
                    steps.append(ReActStep(thought=parsed.thought, tool=None, arg=None, observation=None, raw = resp.text, error=None, input_tokens=input_tokens, output_tokens=output_tokens,
                                           latency_s=latency, attempts=attempt))
                    
                    return ReActResults(parsed.answer, status='ok', used_fallback=False, steps=steps)
                
                if isinstance(parsed, Action):
                    
                    tool = normalize_label(s=parsed.tool, labels=tool_labels)
                    
                    if tool is None:
                        curr_history.append({'role': 'assistant', 'content': parsed.kept})
                        curr_history.append({'role': 'user', 'content': f'The provided tool did not match any of the provided options. You selected: {parsed.tool} but must select from the following: {tool_labels}.'})
                        continue
                    
                    arg = parsed.arg.lower()
                    
                    if (tool, arg) in seen and tool not in allowed_repeats:
                        
                        if seen[(tool, arg)] == 1:
                            observation = f'Duplicate tool call detected. You have already called tool: {parsed.tool} with parameter: {arg}. Dupicates are not allowed another duplicate will terminate the run.'
                            steps.append(ReActStep(thought=parsed.thought, tool=tool, arg=parsed.arg, observation=observation, raw=resp.text, error= 'repeat tool call', input_tokens=input_tokens, output_tokens=output_tokens, latency_s=latency, attempts=attempt))
                            curr_history.append({'role': 'assistant', 'content': parsed.kept})
                            curr_history.append({'role': 'user', 'content': observation})
                            seen[(tool, arg)] += 1
                            continue
                        else:
                            
                            ans, step = fallback_trigger(llm, history=history, max_tokens=max_tokens, parser=parser)
                            steps.append(step)
                            
                            return ReActResults(answer=ans, status='loop', used_fallback=True, steps=steps)
                                                 
                    else:
                        
                        try:
                            observation = tools[tool](parsed.arg)
                        except Exception as e:
                            print(e)
                            curr_history.append({'role': 'assistant', 'content': parsed.kept})
                            curr_history.append({'role': 'user', 'content': f'Tool error: {e}'})
                            continue
                            
                        history.append({'role': 'assistant', 'content': parsed.kept})
                        history.append({'role': 'user', 'content': f'<observation>{observation}</observation>'})
                        
                        steps.append(ReActStep(thought=parsed.thought, tool=tool, arg=parsed.arg, observation=observation, raw=resp.text, error=None, input_tokens=input_tokens,
                                               output_tokens=output_tokens, latency_s=latency, attempts=attempt))
                    
                        seen[(tool, arg)] = 0
                    
                    seen[(tool, arg)] += 1
                    break
                
                else:
                        
                    if attempt == max_format_errors:
                        steps.append(ReActStep(thought=None, tool=None, arg=None, observation=None, raw=resp.text, error=parsed.reason, input_tokens=input_tokens, output_tokens=output_tokens,
                                               latency_s=latency, attempts=attempt))    

                        ans, step = fallback_trigger(llm, history, max_tokens, parser)
                        steps.append(step)
                        return ReActResults(answer=ans, status='format_failure', used_fallback=True, steps=steps)
                        
                    curr_history.append({'role': 'assistant', 'content': parsed.kept})
                    curr_history.append({'role': 'user', 'content': parsed.reason})  
            else:
                raise ValueError("ERROR: LLM API connection issues")
            
                
    ans, step = fallback_trigger(llm, history, max_tokens, parser)
    steps.append(step)
    return ReActResults(answer=ans, status='max_steps', used_fallback=True, steps=steps)
                        
                        
def fallback_trigger(llm, history, max_tokens, parser):
        

    fallback_system = render('system/react_fallback.j2')
    fallback_user = [{'role': 'user', 'content': render('user/react_fallback.j2', history=history)}]
    
    resp = llm.complete(system=fallback_system, messages=fallback_user, max_tokens=max_tokens, stop=['<observation>'])
    
    if resp.stop_reason == 'end_turn' or resp.stop_reason == 'max_tokens':
        
        parsed = parser(resp.text)
        #breakpoint()
        if not isinstance(parsed, Finish):
            return None, ReActStep(thought=None, tool=None, arg=None, observation=None, raw=resp.text, error=None, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                                       latency_s=resp.latency_s, attempts=1)
        
        
        return parsed.answer, ReActStep(thought=parsed.thought, tool=None, arg=None, observation=None, raw=resp.text, error=None, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                                   latency_s=resp.latency_s, attempts=1)
            
    


"""
@dataclass
class Action:
    thought: str | None
    tool: str
    arg: str
    kept: str = ""

@dataclass  
class Finish:
    thought: str | None
    answer: str
    kept: str = ""

@dataclass
class ReActParserError:
    reason: str
    raw: str
    kept: str = ""
"""
