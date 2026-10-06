from extraction.repair import describe_loc
from pydantic import ValidationError
import json

def to_claude_tool(name, description, input_model):
    
    return {'name': name, 'description': description, 'input_schema': input_model.model_json_schema()}

def tools_for_claude(registry):
    
    results = []
    
    for name, funcs in registry.items():
        
        results.append(to_claude_tool(name=name, description=funcs['description'], input_model=funcs['input_model']))
    
    return results

def parse_tool_calls(response):
    
    return [block for block in response.content if block.type == 'tool_use']

def run_tool_call(block, registry):
    
    name = block.name
    
    if name not in registry:
        
        msg = f"Tool '{name}' did not match any of the provided tool names. Tool options are: {" ,".join(r for r,_ in registry.items())}"
        
        return {'type': 'tool_result', 'tool_use_id': block.id, 'content': msg, 'is_error': True}
    
    try:
        model = registry[name]['input_model']
        
        validated = model.model_validate(block.input)
    except ValidationError as e:
        lines = ["Your JSON did not match the required format. Fix these problems:"]
        errs = e.errors(include_url=False, include_context=False)
        for err in errs:
            where = describe_loc(err["loc"])
            wrote = repr(err['input'])[:80]
            lines.append(f"- {where}: {err['msg']} (you wrote {wrote})")
        
        lines.append("Return the corrected JSON only. Do not add any explanation")
        
        return {'type': 'tool_result', 'tool_use_id': block.id, 'content': "\n".join(lines), 'is_error': True}
    
    try:
        
        result = registry[name]['function'](validated)
    
    except ValueError as e:
        
        return {'type': 'tool_result', 'tool_use_id': block.id, 'content': str(e), 'is_error': True}
    
    return {'type': 'tool_result', 'tool_use_id': block.id, 'content': json.dumps(result)}

def run_tool_calls(blocks, registry):
    
    return [run_tool_call(block, registry) for block in blocks]

