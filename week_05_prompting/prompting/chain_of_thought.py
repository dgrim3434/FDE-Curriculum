"""Chain-of-thought wrapper: adds CoT instructions to any system prompt and extracts the final answer."""
import re
from dataclasses import dataclass

from prompting.budget import BudgetExceeded, BudgetTracker
from prompting.llm import LLMResponse
from prompting.templates import render

MAX_TOKENS_CAP = 4096
FLAGS = re.DOTALL | re.IGNORECASE
RE_TAG = r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>"
RE_OPEN = r"<\s*answer\s*>\s*([^<\n]*)"
RE_MARKER = r"(?:####|final answer\s*(?:is)?\s*[:=]?|the answer is|answer\s*:)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)"
RE_THINKING = r"<\s*thinking\s*>(.*?)<\s*/\s*thinking\s*>"


@dataclass(frozen=True)
class CoTResult:
    method: str | None
    format_ok: bool
    answer: float | None
    valid: bool            
    reason: str               
    thought: str | None
    attempts: int
    total_cost: float
    input_tokens: int         
    output_tokens: int
    error_log: list
    raw: LLMResponse | None


def generate_cot_system(system_prompt: str, examples: list[dict] | None = None) -> str:
    if examples:
        if not isinstance(examples, list) or not all(isinstance(e, dict) for e in examples):
            raise ValueError("examples must be a list of dicts with question/thinking/answer")
    return render("system/cot.j2", system_prompt=system_prompt, examples=examples or None)


def chain_of_thought(llm, system_prompt: str, user_prompt: str, budget: BudgetTracker,
                     examples: list[dict] | None = None, max_retries: int = 4,
                     max_tokens: int = 512, use_cot: bool = True) -> CoTResult:
    if use_cot:
        system_prompt = generate_cot_system(system_prompt, examples)

    history = [{"role": "user", "content": user_prompt}]
    error_log: list[str] = []
    total_spend, in_tok, out_tok = 0.0, 0, 0
    curr_max = max_tokens
    attempt = 0
    resp = None

    def result(reason, answer=None, thought=None, method=None, format_ok=False):
        return CoTResult(method=method, format_ok=format_ok, answer=answer, valid=answer is not None,
                         reason=reason, thought=thought, attempts=attempt, total_cost=total_spend,
                         input_tokens=in_tok, output_tokens=out_tok, error_log=error_log, raw=resp)

    while attempt < max_retries:
        try:
            budget.check()
        except BudgetExceeded:
            return result("budget")

        resp = llm.complete(system=system_prompt, messages=history, temperature=0.0, max_tokens=curr_max)
        attempt += 1
        budget.add(resp.cost)
        total_spend += resp.cost
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens

        if resp.stop_reason == "refusal":
            return result("refusal")

        if resp.stop_reason == "max_tokens":
            error_log.append(f"truncated at max_tokens={curr_max}")
            curr_max = min(int(curr_max * 1.5), MAX_TOKENS_CAP)
            continue

        answer, thought, method, errors = parse_output(resp.text, require_thinking=use_cot)
        error_log += errors
        if answer is not None:
            return result("ok", answer=answer, thought=thought, method=method, format_ok=not errors)

        history.append({"role": "assistant", "content": resp.text})
        history.append({"role": "user", "content": render("user/invalid_output_error_log.j2", error_logs=errors)})

    last_truncated = resp is not None and resp.stop_reason == "max_tokens"
    return result("truncated" if last_truncated else "exhausted")


def parse_output(response: str, require_thinking: bool = True):
    """Returns (answer: float | None, thought: str | None, method: str | None, errors: list[str])."""
    errors: list[str] = []

    for method, pattern in (("tag", RE_TAG), ("open_tag", RE_OPEN), ("marker", RE_MARKER)):
        matches = list(re.finditer(pattern, response, FLAGS))
        if matches:
            break
        errors.append({"tag": "No <answer>...</answer> block found",
                       "open_tag": "No opening <answer> tag found",
                       "marker": "No final answer found anywhere in the response"}[method])
    else:
        return None, None, None, errors

    if len(matches) > 1:
        errors.append("Multiple answers found; only one is allowed")
    match = matches[-1]

    raw_answer = match.group(1)
    try:
        answer = clean_answer(raw_answer)
    except ValueError:
        errors.append(f"The answer must be a plain number, got: {raw_answer.strip()!r}")
        return None, None, None, errors

    thinking = re.findall(RE_THINKING, response, FLAGS)
    if thinking:
        thought = thinking[-1].strip()
    else:
        if require_thinking:
            errors.append("No <thinking>...</thinking> block found")
        thought = response[:match.start()].strip() or None

    return answer, thought, method, errors


def clean_answer(ans: str) -> float:
    return float(re.sub(r"[,$\s]", "", ans))