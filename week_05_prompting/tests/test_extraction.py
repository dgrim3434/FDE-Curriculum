"""
AI Generated Test cases for the extraction functions. 
"""
"""
Tests for the structured-output extraction pipeline.

Run from week_05_prompting/:   python -m pytest tests/test_extraction.py -v

ASSUMED INTERFACE — if yours differs, change the import line / the `run()` helper below,
not the individual tests:

    extraction(llm, system_prompt, user_prompt, schema, tracker, max_retries=..., max_tokens=...)
        -> ExtractionResults(ok, data, reason, attempts, error_log, total_cost, raw)
        reason is one of: "ok", "refusal", "truncated", "exhausted", "budget"
    CostTracker(budget)  with .add(amount) and .check() (raises BudgetExceeded)
    parse_json(text, error_log) -> dict | None
    validate_json_fields(obj, schema, schema_mapping) -> (validated, errors_found, errors)
    define_schema_field_mapping(schema) -> dict
"""
import json

import pytest

from prompting.llm import LLMResponse
from prompting.extraction import (
    define_schema_field_mapping,
    extraction,
    parse_json,
    validate_json_fields,
)
from prompting.budget import BudgetExceeded
from experiments.extraction import get_schema_template

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Small schema with the same shape as EXTRACTION_SCHEMA (3 topics instead of 77 keeps it readable).
SCHEMA = {
    "topic": {
        "type": (str,),
        "nullable": False,
        "allowed": ["card_arrival", "lost_or_stolen_card", "Refund_not_showing_up"],
        "description": "the intent",
    },
    "urgency": {
        "type": (str,),
        "nullable": False,
        "allowed": ["low", "medium", "high"],
        "description": "how urgent",
    },
    "amount_mentioned": {
        "type": (int, float),
        "nullable": True,
        "description": "amount in the message",
    },
}

VALID = {"topic": "card_arrival", "urgency": "low", "amount_mentioned": None}


def js(**overrides) -> str:
    """Valid JSON text with some fields overridden. Pass drop=[...] to remove fields."""
    drop = overrides.pop("drop", [])
    d = {**VALID, **overrides}
    for k in drop:
        d.pop(k)
    return json.dumps(d)


def R(text: str, stop: str = "end_turn", inp: int = 500, out: int = 30) -> LLMResponse:
    return LLMResponse(text=text, stop_reason=stop, input_tokens=inp, output_tokens=out)


class RecordingLLM:
    """Scripted fake LLM.

    Why not the plain FakeLLM: extraction() keeps appending to the SAME `history` list after
    each call. If the fake stores a reference to that list, calls[0]["messages"] silently changes
    later. This fake stores a COPY of the messages at the moment of each call.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system, messages, temperature=0.0, max_tokens=512):
        self.calls.append({
            "system": system,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if not self._responses:
            raise AssertionError(f"extraction made call #{len(self.calls)} but only "
                                 f"{len(self.calls) - 1} responses were scripted")
        return self._responses.pop(0)


def run(responses, budget=None, max_retries=3, max_tokens=200):
    """Call extraction with a fake LLM. Returns (result, llm) so tests can inspect calls."""
    llm = RecordingLLM(responses)
    template = get_schema_template(SCHEMA)
    budget = budget if budget is not None else BudgetExceeded(1.00)
    result = extraction(llm, system_prompt="SYS", user_prompt="USER MESSAGE", schema=SCHEMA, schema_template=template,
                        budget=budget, max_retries=max_retries, max_tokens=max_tokens)
    return result, llm


def last_user_message(call) -> str:
    return call["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# 1. Happy paths — no retry should happen
# ---------------------------------------------------------------------------

def test_valid_json_first_try():
    result, llm = run([R(js())])
    assert result.ok and result.reason == "ok"
    assert result.data == VALID
    assert result.attempts == 1 and len(llm.calls) == 1


def test_first_call_sends_only_the_user_message_and_system_prompt():
    _, llm = run([R(js())])
    call = llm.calls[0]
    assert call["system"] == "SYS"
    assert call["messages"] == [{"role": "user", "content": "USER MESSAGE"}]


def test_fenced_json_is_fixed_locally_without_retry():
    text = f"```json\n{js()}\n```"
    result, llm= run([R(text)])
    assert result.ok and len(llm.calls) == 1


def test_fence_without_json_tag():
    result, llm = run([R(f"```\n{js()}\n```")])
    assert result.ok and len(llm.calls) == 1


def test_prose_around_json_is_fixed_locally_without_retry():
    result, llm = run([R(f"Sure! Here is the extraction: {js()} Let me know if you need more.")])
    assert result.ok and len(llm.calls) == 1


def test_case_and_spacing_normalized_to_canonical_values():
    result, _ = run([R(js(topic="Card Arrival", urgency="HIGH"))])
    assert result.ok
    assert result.data["topic"] == "card_arrival"
    assert result.data["urgency"] == "high"


def test_canonical_capitalized_label_is_returned():
    result, _ = run([R(js(topic="refund not showing up"))])
    assert result.ok
    assert result.data["topic"] == "Refund_not_showing_up"


def test_capitalized_field_name_accepted():
    text = json.dumps({"Topic": "card_arrival", "Urgency": "low", "amount_mentioned": None})
    result, llm = run([R(text)])
    assert result.ok and len(llm.calls) == 1
    assert set(result.data) == set(SCHEMA)          # keys come back in schema spelling


@pytest.mark.parametrize("amount", [None, 40, 40.5, 0])
def test_amount_accepts_null_int_float(amount):
    result, _ = run([R(js(amount_mentioned=amount))])
    assert result.ok
    assert result.data["amount_mentioned"] == amount


# ---------------------------------------------------------------------------
# 2. Content retries — bad output, then good output
# ---------------------------------------------------------------------------

def test_invalid_value_then_fixed():
    bad = js(urgency="extreme")
    result, llm = run([R(bad), R(js())])
    assert result.ok and result.attempts == 2 and len(llm.calls) == 2

    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"] == bad                     # model sees its own bad answer
    feedback = last_user_message(llm.calls[1])
    assert feedback.strip() != ""
    assert "urgency" in feedback.lower()                   # names the broken field
    assert "missing" not in feedback.lower()               # present-but-invalid ≠ missing


def test_invented_topic_then_fixed():
    result, llm = run([R(js(topic="card_delay")), R(js())])
    assert result.ok and len(llm.calls) == 2
    assert "topic" in last_user_message(llm.calls[1]).lower()


def test_not_json_then_fixed():
    result, llm = run([R("I think this is about a card arriving."), R(js())])
    assert result.ok and len(llm.calls) == 2
    feedback = last_user_message(llm.calls[1])
    assert feedback.strip() != ""
    assert "json" in feedback.lower()


def test_single_quoted_pseudo_json_then_fixed():
    bad = "{'topic': 'card_arrival', 'urgency': 'low', 'amount_mentioned': None}"
    result, llm = run([R(bad), R(js())])
    assert result.ok and len(llm.calls) == 2


def test_missing_field_then_fixed():
    result, llm = run([R(js(drop=["urgency"])), R(js())])
    assert result.ok and len(llm.calls) == 2
    assert "urgency" in last_user_message(llm.calls[1]).lower()


def test_extra_field_then_fixed():
    text = json.dumps({**VALID, "sentiment": "angry"})
    result, llm = run([R(text), R(js())])
    assert result.ok and len(llm.calls) == 2
    assert "sentiment" in last_user_message(llm.calls[1]).lower()


def test_bool_amount_rejected_then_fixed():
    result, llm = run([R(js(amount_mentioned=True)), R(js())])
    assert result.ok and len(llm.calls) == 2


def test_null_for_required_field_rejected_then_fixed():
    result, llm = run([R(js(urgency=None)), R(js())])
    assert result.ok and len(llm.calls) == 2


def test_multiple_errors_reported_together():
    text = json.dumps({"topic": "card_delay", "urgency": "extreme", "amount_mentioned": None})
    _, llm = run([R(text), R(js())])
    feedback = last_user_message(llm.calls[1]).lower()
    assert "topic" in feedback and "urgency" in feedback     # both in ONE retry message


def test_error_log_records_retry_errors():
    result, _ = run([R(js(urgency="extreme")), R(js())])
    assert result.ok
    assert len(result.error_log) >= 1


# ---------------------------------------------------------------------------
# 3. Giving up correctly
# ---------------------------------------------------------------------------

def test_exhausted_after_max_retries():
    bad = R(js(urgency="extreme"))
    result, llm = run([bad, bad, bad], max_retries=3)
    assert not result.ok and result.reason == "exhausted"
    assert result.data is None
    assert len(llm.calls) == 3 and result.attempts == 3
    assert result.raw is not None                          # keep last response for debugging


def test_refusal_is_not_retried():
    result, llm = run([R("I can't help with that.", stop="refusal")])
    assert not result.ok and result.reason == "refusal"
    assert len(llm.calls) == 1


def test_max_tokens_then_valid_retries_with_bigger_limit():
    truncated = R('{"topic": "card_arr', stop="max_tokens", out=200)
    result, llm = run([truncated, R(js())], max_tokens=200)
    assert result.ok
    assert llm.calls[1]["max_tokens"] > llm.calls[0]["max_tokens"]


def test_max_tokens_every_time_ends_truncated_without_empty_messages():
    t = R('{"topic": "card_arr', stop="max_tokens", out=200)
    result, llm = run([t, t, t], max_retries=3)
    assert not result.ok and result.reason == "truncated"
    assert len(llm.calls) <= 3
    for call in llm.calls:
        for m in call["messages"]:
            assert m["content"].strip() != "", "an empty message would be rejected by the API"


# ---------------------------------------------------------------------------
# 4. Cost and budget
# ---------------------------------------------------------------------------

def test_budget_already_spent_makes_no_calls():
    tracker = BudgetExceeded(0.0)
    result, llm = run([R(js())], budget=tracker)
    assert not result.ok and result.reason == "budget"
    assert len(llm.calls) == 0


def test_total_cost_is_sum_of_this_items_responses():
    r1, r2 = R(js(urgency="extreme"), inp=500, out=30), R(js(), inp=700, out=30)
    result, _ = run([r1, r2])
    assert result.total_cost == pytest.approx(r1.cost + r2.cost)


def test_shared_tracker_accumulates_across_items_but_item_cost_does_not():
    tracker = BudgetExceeded(1.00)
    a = R(js(), inp=500, out=30)
    b = R(js(), inp=900, out=30)
    res_a, _ = run([a], budget=tracker)
    res_b, _ = run([b], budget=tracker)
    # each result reports ITS OWN cost…
    assert res_a.total_cost == pytest.approx(a.cost)
    assert res_b.total_cost == pytest.approx(b.cost)
    # …while the run-level tracker blocks once the shared budget is gone
    tiny = BudgetExceeded(a.cost)          # budget exactly one call
    run([R(js())], budget=tiny)
    res_c, llm_c = run([R(js())], budget=tiny)
    assert res_c.reason == "budget" and len(llm_c.calls) == 0


# ---------------------------------------------------------------------------
# 5. Unit tests for the pieces (no LLM) — these localize failures from sections 1–3
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    js(),
    f"```json\n{js()}\n```",
    f"```\n{js()}\n```",
    f"Here you go: {js()} Thanks!",
    f"{js()}\n\n{js(urgency='high')}",               # two objects → first one wins
])
def test_parse_json_recovers(text):
    val, out = parse_json(text)
    assert val == VALID


@pytest.mark.parametrize("text", [
    "",
    "not json at all",
    "{'topic': 'card_arrival'}",                     # single quotes
    '{"topic": "card_arrival",}',                    # trailing comma
    '{"topic": "card_arr',                           # truncated
])
def test_parse_json_returns_none_on_garbage(text):
    val, out = parse_json(text)
    assert val is None


def test_parse_json_success_logs_nothing():

    ans, log = parse_json(js())
    assert log == []


MAPPING = define_schema_field_mapping(SCHEMA)


def test_validate_all_valid():
    out, found, errors = validate_json_fields(dict(VALID), SCHEMA, MAPPING)
    assert out == VALID and not found and errors == []


@pytest.mark.parametrize("override, field", [
    ({"urgency": "extreme"}, "urgency"),
    ({"topic": "card_delay"}, "topic"),
    ({"amount_mentioned": True}, "amount_mentioned"),
    ({"amount_mentioned": "40"}, "amount_mentioned"),   # delete this case if you chose to clean "40" locally
    ({"urgency": None}, "urgency"),
])
def test_validate_single_bad_field_gives_exactly_one_error(override, field):
    _, found, errors = validate_json_fields({**VALID, **override}, SCHEMA, MAPPING)
    assert found
    assert len(errors) == 1, f"expected one error for {field}, got: {errors}"
    assert field in errors[0].lower()


def test_validate_missing_field():
    obj = {k: v for k, v in VALID.items() if k != "urgency"}
    _, found, errors = validate_json_fields(obj, SCHEMA, MAPPING)
    assert found and any("urgency" in e.lower() for e in errors)


def test_validate_extra_field():
    _, found, errors = validate_json_fields({**VALID, "sentiment": "angry"}, SCHEMA, MAPPING)
    assert found and any("sentiment" in e.lower() for e in errors)
