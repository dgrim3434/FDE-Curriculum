import json
import re
from dataclasses import dataclass

from prompting.budget import BudgetExceeded, BudgetTracker
from prompting.llm import LLMResponse
from prompting.templates import render

TYPE_NAMES = {int: "integer", float: "number", str: "string", bool: "boolean"}
MAX_TOKENS_CAP = 4096 

@dataclass(frozen=True)
class ExtractionResults:
    ok : bool
    data: dict | None
    reason: str
    attempts: int
    error_log: list
    total_cost: float
    raw: LLMResponse | None

def extraction(llm, system_prompt: str, user_prompt: str, schema: dict, budget: BudgetTracker, schema_template: list[dict] | None = None, max_retries = 4, max_tokens=256) -> ExtractionResults:
    
    schema_template = schema_template or get_schema_template(schema)
    schema_mapping = define_schema_field_mapping(schema)
    
    total_spend = 0.0
    history = [{'role': 'user', 'content': user_prompt}]
    error_log = []

    cur_max = max_tokens
    attempt = 0
    resp = None
    
    def result(ok, reason, data=None):
        
        return ExtractionResults(ok=ok, data=data, reason=reason, attempts=attempt,
                                 error_log=error_log, total_cost=total_spend, raw=resp)
    
    while attempt < max_retries:
        
        try:
            budget.check()
        except BudgetExceeded:
            return result(False, "budget")
        
        resp = llm.complete(system = system_prompt, messages = history, temperature=0.0, max_tokens = cur_max)
        attempt += 1
        budget.add(resp.cost)
        total_spend += resp.cost
        
        if resp.stop_reason == 'refusal':
            return result(False, "refusal")
        
        if resp.stop_reason == 'max_tokens':
            
            error_log.append(f"truncated at max_tokens={cur_max}")
            cur_max = min(int(cur_max * 1.5), MAX_TOKENS_CAP)
            continue
        
        parsed, parse_errors = parse_json(resp.text)
        error_log += parse_errors
        if parsed is None:
            feedback = parse_errors[-1:]
        else:
            validated, field_errors = validate_json_fields(parsed, schema, schema_mapping)
            
            if not field_errors:
                return result(True, 'ok', data=validated)
            
            error_log += field_errors
            feedback = field_errors
        
        history.append({'role': 'assistant', 'content': resp.text})
        history.append({'role': 'user', 'content': render("user/json_repair_user.j2", fields=schema_template, errors=feedback)})
    
    last_truncated = resp is not None and resp.stop_reason == "max_tokens"
    return result(False, "truncated" if last_truncated else "exhausted")


def norm(s: str) -> str: 
    return s.strip().lower().replace(" ", "_")     


def validate_json_fields(parsed, schema, schema_mapping):
    
    validated = {}
    seen = set()
    errors = []
    
    
    for field, value in parsed.items():
        
        key = schema_mapping.get(norm(field))
        
        if key is None:
            errors.append(f'Unknown field "{field}": it is not in the schema')
            continue
        
        if key in seen:
            errors.append(f'Field "{key}" appears more than once')
            continue
        seen.add(key)
        
        spec = schema[key]
        
        if value is None:
            if not spec.get("nullable", False):
                errors.append(f'"{key}" cannot be null')
            else:
                validated[key] = None
            
            continue
        
        types = _as_tuple(spec["type"])
        wrong_type = not isinstance(value, types)
        bool_as_number = isinstance(value, bool) and bool not in types
        
        if wrong_type or bool_as_number:
            expected = " or ".join(TYPE_NAMES.get(t, t.__name__) for t in types)
            errors.append(f'"{key}" must be {expected}, got {type(value).__name__} ({value!r})')
            continue
        
        allowed = spec.get("allowed")
        if allowed:
            lookup = {norm(a) if isinstance(a, str) else a: a for a in allowed}
            canonical = lookup.get(norm(value) if isinstance(value, str) else value)
            if canonical is None:
                errors.append(f'"{key}" must be one of the allowed values, got {value!r}')
                continue
            value = canonical
        
        validated[key] = value
        
    missing = set(schema) - seen
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    return validated, errors
            
        
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
        result, _ = json.JSONDecoder().raw_decode(text, start)
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
    
    return {norm(field): field for field in schema}

def _as_tuple(t) -> tuple:
    return t if isinstance(t, tuple) else (t,)

def get_schema_template(schema: dict) -> list[dict]:
    """Schema (Python types) -> list of plain dicts the Jinja templates can print.
    Lives in the library, next to the schema format it reads, so the two can't drift apart."""
    fields = []
    for name, spec in schema.items():
        types = _as_tuple(spec["type"])
        fields.append({
            "name": name,
            "type": "number" if float in types else " or ".join(TYPE_NAMES.get(t, t.__name__) for t in types),
            "nullable": spec.get("nullable", False),
            "allowed": spec.get("allowed"),
            "description": spec.get("description"),
        })
    return fields