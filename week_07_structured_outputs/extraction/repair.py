import re
import ast
from dataclasses import dataclass, field
from extraction.schemas import validate_with_repairs
import json

FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
TRAILING_COMMA = re.compile(r",\s*([}\]])")


def strip_fences(text):
    
    m = FENCE.match(text)
    
    if m is not None:
        
        return m.group(1), True
    
    return text, False

def extract_json_span(text):
    
    starts = [i for i in(text.find('{'), text.find('[')) if i != -1]
    
    if not starts:
        return text, False
    
    start = min(starts)
    
    end = max(text.rfind('}'), text.rfind(']'))
    
    if end <= start:
        return text, False
    
    span = text[start: end + 1]
    
    return span, span != text

def trailing_comma_removal(text):
    
    new = TRAILING_COMMA.sub(r"\1", text)
    
    return new, new != text

def parse_python_literal(text):
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, (dict, list)) else None


@dataclass
class ParseResults:
    ok: bool
    data: dict | None
    rules_fired: list
    failed_stage: str | None
    error: str | None


def parse_with_repairs(raw_text, done_reason=None):
    

    if done_reason in set(["length", "max_tokens"]):
        return ParseResults(ok=False, data=None, rules_fired=[], failed_stage="truncated", error=None)
    
    try:
        result = json.loads(raw_text)
        
        return ParseResults(ok=True, data=result, rules_fired=[], failed_stage=None, error=None)
    except json.JSONDecodeError:
        pass
    
    rules_fired = []
    
    text, fired = strip_fences(raw_text)
    
    if fired:
        rules_fired.append('strip_fences')
    
    text, fired = extract_json_span(text)
    
    if fired:
        rules_fired.append('extract_json_span')
        
    text, fired = trailing_comma_removal(text)
    
    if fired:
        rules_fired.append('trailing_comma_removal')
        
    error = None
    
    try:
        result = json.loads(text)
        
        return ParseResults(ok=True, data=result, rules_fired=rules_fired, failed_stage=None, error=None)
    except json.JSONDecodeError as e:
        
        error = f"JSON decoder error: {e}. Return the corrected JSON only"
    
    result = parse_python_literal(text)
    
    if result is None:
        return ParseResults(ok=False, data=None, rules_fired=rules_fired, failed_stage='parse', error=error)
    
    rules_fired.append('python_literal')
    
    return ParseResults(ok=True, data=result, rules_fired=rules_fired, failed_stage=None, error=None)


@dataclass
class PipelineResult:
    ok: bool
    value: object | None
    failed_stage: str | None
    rules_fired: list[str] = field(default_factory=list)
    errors: list = field(default_factory=list)
    model_message: str | None = None
    

def run_pipeline(raw_text, schema, done_reason=None):
    
    parsed = parse_with_repairs(raw_text, done_reason)
    
    if not parsed.ok:
        
        return PipelineResult(ok=False, value=None, failed_stage=parsed.failed_stage, rules_fired=parsed.rules_fired, errors=[parsed.error], model_message=parsed.error)
    
    checked = validate_with_repairs(parsed.data, schema)
    
    rules_fired = parsed.rules_fired + checked.rules_fired
    
    if checked.ok:
        
        return PipelineResult(ok=True, value=checked.value, failed_stage=None, rules_fired=rules_fired, errors=[], model_message=None)
    
    lines = ["Your JSON did not match the required format. Fix these problems:"]
    
    for err in checked.errors:
        where = describe_loc(err["loc"])
        
        wrote = repr(err['input'])[:80]
        lines.append(f"- {where}: {err['msg']} (you wrote {wrote})")
    
    lines.append("Return the corrected JSON only. Do not add any explanation")
    
    model_message = "\n".join(lines)
    
    return PipelineResult(ok=False, value=None, failed_stage = "validation", rules_fired=rules_fired, errors=checked.errors, model_message=model_message)


def describe_loc(loc):
    
    if not loc:
        return "the whole object"
    
    parts = []
    for p in loc:
        if isinstance(p, int):
            parts.append(f"item {p + 1}")
        else:
            parts.append(f'"{p}"')
    
    return " -> ".join(parts)
