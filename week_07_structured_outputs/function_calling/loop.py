from function_calling.formatter import tools_for_claude, parse_tool_calls, run_tool_calls, run_tool_call
from dataclasses import dataclass
from extraction.schemas import Extraction
from function_calling.tools import record_entities


@dataclass
class ToolLoopResult:
    final_text: str | None
    steps: int
    hit_max_steps: bool
    tool_calls: list
    messages: list
    input_tokens: int
    output_tokens: int

def run_tool_loop(client, user_message, registry, max_steps=5, tool_choice=None, system=None):
    
    tools = tools_for_claude(registry=registry)
    input_tokens = 0
    output_tokens = 0
    messages = [{'role': 'user', 'content': user_message}]
    tool_calls = []
    
    step = 0
    while step < max_steps + 1:
        
        if step == 0:
            resp = client.complete_with_tools(messages=messages, tools=tools, tool_choice=tool_choice, system=system)
        else:
            resp = client.complete_with_tools(messages=messages, tools=tools, tool_choice=None, system=system)
            
        input_tokens += resp.usage.input_tokens
        output_tokens += resp.usage.output_tokens
        
        if resp.stop_reason != "tool_use":
            answer = "".join(block.text for block in resp.content if block.type == "text")
            
            return ToolLoopResult(final_text=answer, steps=step, hit_max_steps=False, tool_calls=tool_calls, messages=messages, input_tokens=input_tokens, output_tokens=output_tokens)
        
        if step == max_steps:
            return ToolLoopResult(final_text=None, steps=step, hit_max_steps=True, tool_calls=tool_calls, messages=messages, input_tokens=input_tokens, output_tokens=output_tokens)
        
        calls = parse_tool_calls(resp)
        tool_mapping = {b.id: (b.name, b.input) for b in calls}
        
        results = run_tool_calls(calls, registry)
        
        for r in results:
            if 'is_error' in r:
                is_error = True
            else:
                is_error = False
            if r['tool_use_id'] in tool_mapping:
                tup = tool_mapping[r['tool_use_id']]
                tool_calls.append((tup + (is_error,)))
                
        messages.append({'role': 'assistant', 'content': resp.content})
        messages.append({'role': 'user', 'content': results})
        step += 1
        
extraction_registry = {
        'record_entities': {'description': 'Takes a list of entity mappings which maps the entity to its type. Should be used when you are ready to answer the question.',
                            'input_model': Extraction, 'function': record_entities}
}

def extract_with_tool(client, text, max_retries = 2):
    
    messages = [{'role': 'user', 'content': text}]
    
    attempt = 0
    tool_choice = {"type": "tool", "name": "record_entities"}
    tools = tools_for_claude(extraction_registry)
    
    while attempt < max_retries + 1:
        
        resp = client.complete_with_tools(messages=messages, tools=tools, tool_choice=tool_choice)
        attempt += 1
        calls = parse_tool_calls(resp)
        
        if len(calls) == 0:
            return None
        
        call = calls[0]
        
        res = run_tool_call(call, extraction_registry)
        
        if 'is_error' in res:
            messages.append({'role': 'assistant', 'content': resp.content})
            messages.append({'role': 'user', 'content': [res]})
            continue
        
        return Extraction.model_validate(call.input)
    
    return None
        
        