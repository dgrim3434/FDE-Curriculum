from dataclasses import dataclass
from prompting.llm import LLMResponse
import json
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from prompting.budget import BudgetExceeded
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=0)

error_template = env.get_template("user/json_repair_user.j2")

class ExtractionResults:
    ok : bool
    data: dict | None
    reason: str
    attempts: int
    error_log: list
    total_cost: float
    raw: LLMResponse | None

def extraction(llm, system_prompt: str, user_prompt: str, schema: dict, budget: BudgetExceeded, schema_template: list[dict], max_retries = 4, max_tokens=50) -> ExtractionResults:
    
    total_spend = 0
    schema_mapping = define_schema_field_mapping(schema)
    history = []
    error_log = []
    history.append({'role': 'user', 'content': user_prompt})

    invalid_json = False
    attempt = 0
    while attempt < max_retries:
        
        feedback = []
        
        try:
            budget.check()
        except BudgetExceeded:
            return ExtractionResults(ok=False, data=None, reason="budget", attempts=attempt, error_log=error_log, total_cost=total_spend, raw=None)
        
        resp = llm.complete(system = system_prompt, messages = history, temperature=0.0, max_tokens = max_tokens)
        attempt += 1
        budget.add(resp.cost)
        total_spend += resp.cost
        if resp.stop_reason == 'refusal':
            return ExtractionResults(ok=False, data=None, reason='refusal', attempts=attempt, error_log=error_log, total_cost=total_spend, raw=resp)
        
        if resp.stop_reason == 'max_tokens':
            
            curr_max = max_tokens
            while resp.stop_reason == 'max_tokens' and attempt < max_retries:
                
                invalid_json = False
                try:
                    budget.check()
                except BudgetExceeded:
                    return ExtractionResults(ok=False, data=None, reason='truncated', attempts=attempt, error_log =error_log, total_cost=total_spend, raw=resp)
                
                curr_max = int(curr_max * 1.5)

                resp = llm.complete(system=system_prompt, messages=history, temperature=0.0, max_tokens=curr_max)
                budget.add(resp.cost)
                total_spend += resp.cost
                attempt += 1
        
        # Parsing the JSON
        if resp.stop_reason == "end_turn":
            parsed, errors = parse_json(resp.text)
            if len(errors) > 0:
                feedback.append(errors[-1])
                error_log += errors
            
            if parsed is not None:
                valid_schema, found_errors, errors = validate_json_fields(parsed, schema, schema_mapping) 
                
                if not found_errors:
            
                    return ExtractionResults(ok=True, data=valid_schema, reason="ok", attempts=attempt, error_log=error_log, total_cost=total_spend, raw=resp)
            
                error_log += errors
                feedback += errors
                
            invalid_json = True
        history.append({'role': 'assistant', 'content': resp.text})
        upd_prompt = error_template.render(fields = schema_template, errors = feedback)
        history.append({'role': 'user', 'content': upd_prompt})
    
    if invalid_json:
        return ExtractionResults(ok=False, data=None, reason='exhausted', attempts=attempt, error_log=error_log, total_cost=total_spend, raw=resp)
    
    return ExtractionResults(ok=False, data=None, reason='truncated', attempts=attempt, error_log=error_log, total_cost=total_spend, raw=resp)


def norm(s): 
    return s.strip().lower().replace(" ", "_")     


def validate_json_fields(parsed, schema, schema_mapping):
    
    validated_schema = {}
    norm_fields = set()
    errors_found = False
    errors = []
    for field, value in parsed.items():
        
        if norm(field) not in schema_mapping:
            errors.append(f"ERROR When Parsing the provided json : Invalid Field Found. {field} was not within the provided schema")
            errors_found = True
            continue
        
        normalized_field = schema_mapping[norm(field)]
        
        norm_fields.add(normalized_field)
        if schema[normalized_field]['nullable'] and parsed[field] is None:
            validated_schema[normalized_field] = None
            continue
        
        allowed_dtype = schema[normalized_field]['type']
        
        if isinstance(value, allowed_dtype):
            
            if int in allowed_dtype and bool not in allowed_dtype and isinstance(value, bool):
                errors.append(f"Invalid Datatype for field: {normalized_field}, allowed datatypes are: {allowed_dtype} but dtype was: {type(value)}")
                errors_found = True
                continue
            
            if 'allowed' in schema[normalized_field]:
                
                allowed_mapping = {norm(v): v for v in schema[normalized_field]['allowed']}
                lookup = allowed_mapping.get(norm(value))
                
                if lookup is not None:

                    validated_schema[normalized_field] = lookup
                else:
                    errors.append(f"{field} requires values to be in the set: {schema[normalized_field]['allowed']} but it's value was: {value}")
                    errors_found = True
                continue
        
            validated_schema[normalized_field] = value
        else:
            errors.append(f"{field} must be of type: {allowed_dtype} but was: {type(value)}")
            errors_found = True
            
    if set(schema) != set(norm_fields):
        missing = set(schema) - set(norm_fields)
            
        errors.append(f"The output was missing required fields: {missing}")
        errors_found = True
            
    return validated_schema, errors_found, errors
            
            
        
"""
Parsing the JSON Output
"""
def parse_json(text):
    error_logger = []
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, error_logger
        
    except json.JSONDecodeError as e:
        error = f"JSON load failed on raw text at position: {e.pos}. The error message was: {e.msg}"
        error_logger.append(error)
    
    # Try stripping away the markdowns in the text
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    
    if m:
        text = m.group(1)
        
        try:
            result = json.loads(text)
                    
            if isinstance(result, dict):
                return result, error_logger
            
        except json.JSONDecodeError as e:
                error = f"Markdown stripping failed JSON load failed at position: {e.pos}"
                error_logger.append(error)

    start = text.find("{")
    if start == -1:
        error = "No '{' was found in the text."
        error_logger.append(error)
        
        return None, error_logger
    
    try:
        result, end = json.JSONDecoder().raw_decode(text, start)
        if isinstance(result, dict):
            return result, error_logger
        
        error = f"Invalid JSON json.JSONDecoder returned: {result}"
        
        error_logger.append(error)
        
        return None, error_logger
    
    except json.JSONDecodeError as e:
        error = f"ERROR When Parsing the provided json: Final Attempt failed error position: {e.pos}. Error Message: {e.msg}"
        error_logger.append(error)
        
        return None, error_logger

"""
Returns a mapping of all the possible field variants which can be expected without the output and maps them back to the schema
"""
def define_schema_field_mapping(schema: dict) -> dict:
    
    mapping = {}
    
    for field_name in schema.keys():
        
        mapping[norm(field_name)] = field_name
    
    return mapping
