"""
AI generated Testcases
"""

import pytest

from prompting.chain_of_thought import (  # ← change module name if yours differs
    chain_of_thought,
    generate_cot_system,
    parse_output,
)
from prompting.budget import BudgetExceeded
from prompting.llm import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOOD = "<thinking>16 - 3 - 4 = 9 eggs. 9 * 2 = 18.</thinking>\n<answer>18</answer>"
UNPARSABLE = "She makes 9 × $2 = $18.00 (from 2 steps)"          # no tag, no marker → must NOT guess 2
TRUNCATED = "<thinking>16 - 3 - 4 = 9 eggs. Then 9 *"


def R(text, stop="end_turn", inp=300, out=150):
    return LLMResponse(text=text, stop_reason=stop, input_tokens=inp, output_tokens=out)


class RecordingLLM:
    """Scripted fake that stores a COPY of messages per call and raises if called too often."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system, messages, temperature=0.0, max_tokens=512):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages],
                           "temperature": temperature, "max_tokens": max_tokens})
        if not self._responses:
            raise AssertionError(f"chain_of_thought made call #{len(self.calls)} but only "
                                 f"{len(self.calls) - 1} responses were scripted "
                                 f"(missing attempt counter / infinite loop?)")
        return self._responses.pop(0)


def run(responses, budget=None, max_retries=3, **kwargs):
    llm = RecordingLLM(responses)
    budget = budget if budget is not None else BudgetExceeded(1.00)
    result = chain_of_thought(llm, system_prompt="SOLVE MATH", user_prompt="QUESTION",
                              budget=budget, max_retries=max_retries, **kwargs)
    return result, llm


# ---------------------------------------------------------------------------
# 1. Wrapper: the system prompt
# ---------------------------------------------------------------------------

def test_wrapper_keeps_original_system_prompt_and_adds_instructions():
    s = generate_cot_system("SOLVE MATH")
    assert "SOLVE MATH" in s
    assert "<answer>" in s.lower()
    assert "<thinking>" in s.lower()


def test_wrapper_with_examples_includes_them():
    # ← adjust the dict keys to whatever your cot.j2 loop reads
    ex = [{"question": "What is 2 + 3?", "thinking": "2 + 3 = 5", "answer": 5}]
    s = generate_cot_system("SOLVE MATH", examples=ex)
    assert "What is 2 + 3?" in s
    assert s != generate_cot_system("SOLVE MATH")


def test_wrapper_empty_example_list_does_not_crash():
    generate_cot_system("SOLVE MATH", examples=[])


def test_call_uses_wrapped_system_prompt():
    _, llm = run([R(GOOD)])
    assert "SOLVE MATH" in llm.calls[0]["system"]
    assert "<answer>" in llm.calls[0]["system"].lower()


def test_first_call_sends_only_the_user_message():
    _, llm = run([R(GOOD)])
    assert llm.calls[0]["messages"] == [{"role": "user", "content": "QUESTION"}]


def test_DECISION_default_max_tokens_leaves_room_for_reasoning():
    """DECISION: reasoning is ~150-300 tokens; a default near 50 truncates almost every call."""
    llm = RecordingLLM([R(GOOD)])
    chain_of_thought(llm, system_prompt="SOLVE MATH", user_prompt="QUESTION",
                     budget=BudgetExceeded(1.00))
    assert llm.calls[0]["max_tokens"] >= 512


# ---------------------------------------------------------------------------
# 2. Happy paths
# ---------------------------------------------------------------------------

def test_clean_response_first_try():
    result, llm = run([R(GOOD)])
    assert result.valid and result.reason == "ok"
    assert result.answer == pytest.approx(18.0)
    assert isinstance(result.answer, float)
    assert result.method == "tag"
    assert result.format_ok is True
    assert "9 * 2 = 18" in result.thought
    assert result.attempts == 1 and len(llm.calls) == 1
    assert result.raw is not None


def test_answer_without_thinking_tags_is_accepted_but_flagged():
    result, llm = run([R("First 16 - 3 - 4 = 9, then 9 * 2 = 18.\n<answer>18</answer>")])
    assert result.valid and len(llm.calls) == 1          # no retry just for missing thinking tags
    assert result.format_ok is False
    assert result.thought.startswith("First 16")         # fallback: text before <answer>


def test_marker_answer_is_accepted_without_retry():
    result, llm = run([R("9 eggs left, 9 * 2 = 18.\nThe final answer is $18.")])
    assert result.valid and result.method == "marker" and len(llm.calls) == 1
    assert result.answer == pytest.approx(18.0)
    assert result.format_ok is False


def test_money_formatting_in_tag():
    result, _ = run([R("<thinking>x</thinking><answer> $1,250.50 </answer>")])
    assert result.answer == pytest.approx(1250.5)


# ---------------------------------------------------------------------------
# 3. Repair / retry
# ---------------------------------------------------------------------------

def test_unparsable_then_fixed_uses_repair_structure():
    result, llm = run([R(UNPARSABLE), R("<answer>18</answer>")])
    assert result.valid and result.answer == pytest.approx(18.0)
    assert result.attempts == 2 and len(llm.calls) == 2

    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"] == UNPARSABLE              # model sees its own reasoning
    feedback = second[2]["content"]
    assert feedback.strip() != ""
    assert "answer" in feedback.lower()


def test_last_number_is_never_guessed():
    """The (from 2 steps) trap must lead to a retry, never to answer=2."""
    result, llm = run([R(UNPARSABLE), R(UNPARSABLE), R(UNPARSABLE)])
    assert result.answer is None
    assert not result.valid


def test_word_answer_in_tag_triggers_retry_with_nonempty_feedback():
    result, llm = run([R("<thinking>x</thinking><answer>eighteen</answer>"), R(GOOD)])
    assert result.valid and len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1]["content"].strip() != ""


def test_exhausted_after_exactly_max_retries_calls():
    """Catches the missing-attempt-increment infinite loop."""
    result, llm = run([R(UNPARSABLE)] * 3, max_retries=3)
    assert not result.valid and result.reason == "exhausted"
    assert len(llm.calls) == 3 and result.attempts == 3
    assert result.raw is not None
    assert len(result.error_log) >= 1


def test_attempts_equals_number_of_calls():
    result, llm = run([R(TRUNCATED, stop="max_tokens"), R(UNPARSABLE), R(GOOD)], max_retries=3)
    assert result.valid
    assert result.attempts == len(llm.calls) == 3


# ---------------------------------------------------------------------------
# 4. Truncation
# ---------------------------------------------------------------------------

def test_truncation_then_valid_retries_with_bigger_limit():
    result, llm = run([R(TRUNCATED, stop="max_tokens"), R(GOOD)], max_tokens=200)
    assert result.valid
    assert llm.calls[1]["max_tokens"] > llm.calls[0]["max_tokens"]


def test_truncation_retry_does_not_append_truncated_text():
    """A cut-off response isn't a format mistake — resend the same conversation, bigger limit."""
    _, llm = run([R(TRUNCATED, stop="max_tokens"), R(GOOD)], max_tokens=200)
    assert llm.calls[1]["messages"] == [{"role": "user", "content": "QUESTION"}]


def test_raised_limit_persists_across_later_attempts():
    _, llm = run([R(TRUNCATED, stop="max_tokens"), R(UNPARSABLE), R(GOOD)], max_tokens=200)
    limits = [c["max_tokens"] for c in llm.calls]
    assert limits[1] > limits[0]
    assert limits[2] >= limits[1], f"limit fell back after truncation: {limits}"


def test_truncated_every_time_ends_truncated():
    t = R(TRUNCATED, stop="max_tokens")
    result, llm = run([t, t, t], max_retries=3, max_tokens=200)
    assert not result.valid and result.reason == "truncated"
    assert len(llm.calls) == 3
    for call in llm.calls:
        for m in call["messages"]:
            assert m["content"].strip() != ""


def test_open_tag_is_not_trusted_when_truncated():
    """'<answer>1' cut off from '<answer>18' must not be accepted as 1."""
    result, llm = run([R("<thinking>9 * 2</thinking><answer>1", stop="max_tokens"), R(GOOD)],
                      max_tokens=200)
    assert result.answer == pytest.approx(18.0)
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# 5. Refusal, budget, cost
# ---------------------------------------------------------------------------

def test_refusal_is_not_retried():
    result, llm = run([R("I can't help with that.", stop="refusal")])
    assert not result.valid and result.reason == "refusal"
    assert len(llm.calls) == 1
    assert result.raw is not None


def test_budget_already_spent_makes_no_calls():
    result, llm = run([R(GOOD)], budget=BudgetExceeded(0.0))
    assert not result.valid and result.reason == "budget"
    assert len(llm.calls) == 0


def test_budget_stops_retries_midway():
    first = R(UNPARSABLE, inp=300, out=150)
    budget = BudgetExceeded(first.cost)                  # exactly one call's worth
    result, llm = run([first, R(GOOD)], budget=budget)
    assert result.reason == "budget"
    assert len(llm.calls) == 1


def test_total_cost_is_this_items_cost_only():
    shared = BudgetExceeded(1.00)
    a, b = R(GOOD, inp=300, out=150), R(GOOD, inp=900, out=150)
    ra, _ = run([a], budget=shared)
    rb, _ = run([b], budget=shared)
    assert ra.total_cost == pytest.approx(a.cost)
    assert rb.total_cost == pytest.approx(b.cost)


def test_total_cost_sums_retries():
    r1, r2 = R(UNPARSABLE, inp=300, out=150), R("<answer>18</answer>", inp=500, out=10)
    result, _ = run([r1, r2])
    assert result.total_cost == pytest.approx(r1.cost + r2.cost)


# ---------------------------------------------------------------------------
# 6. parse_output on its own (no LLM) — localizes failures from sections 2-4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, answer, method", [
    ("<thinking>a</thinking><answer>18</answer>", 18.0, "tag"),
    ("<answer>16</answer> wait, muffins. <answer>18</answer>", 18.0, "tag"),   # last tag wins
    ("<Answer> $1,250.50 </Answer>", 1250.5, "tag"),
    ("< answer >42</ answer >", 42.0, "tag"),
    ("<answer>-3.5</answer>", -3.5, "tag"),
    ("work it out...\n<answer>18", 18.0, "open_tag"),
    ("So she makes 18 per day.\n#### 18", 18.0, "marker"),
    ("Therefore, the final answer is $18.", 18.0, "marker"),
    ("Final Answer: -3.5", -3.5, "marker"),
    ("Answer: 42 apples", 42.0, "marker"),
])
def test_parse_output_finds_answer(text, answer, method):
    got_answer, _, got_method, _ = parse_output(text)
    assert got_answer == pytest.approx(answer)
    assert got_method == method


@pytest.mark.parametrize("text", [
    "no answer here",
    "",
    "She makes 9 × $2 = $18.00 (from 2 steps)",
    "<answer>eighteen</answer>",
    "<answer></answer>",
])
def test_parse_output_returns_none_when_no_usable_answer(text):
    answer, _, _, errors = parse_output(text)
    assert answer is None
    assert len(errors) >= 1


def test_DECISION_strict_tag_rejects_units():
    """DECISION (strict policy): '<answer>18 dollars</answer>' is rejected → repair.
    If you chose to extract the number from inside the tag instead, change to == 18.0."""
    answer, _, _, _ = parse_output("<answer>18 dollars</answer>")
    assert answer is None


def test_parse_output_thinking_multiline():
    _, thought, _, _ = parse_output("<thinking>line one\nline two</thinking><answer>18</answer>")
    assert thought == "line one\nline two"


def test_parse_output_thinking_case_insensitive():
    _, thought, _, _ = parse_output("<Thinking>x</Thinking><answer>18</answer>")
    assert thought == "x"


def test_parse_output_fallback_thought_is_text_before_answer():
    _, thought, _, _ = parse_output("Step 1: 9 eggs.\nStep 2: 18 dollars.\n<answer>18</answer>")
    assert thought == "Step 1: 9 eggs.\nStep 2: 18 dollars."


def test_parse_output_fallback_thought_for_marker():
    _, thought, _, _ = parse_output("Reasoning here.\n#### 18")
    assert thought == "Reasoning here."


def test_parse_output_last_thinking_block_used():
    _, thought, _, _ = parse_output("<thinking>first</thinking><thinking>second</thinking>"
                                    "<answer>1</answer>")
    assert thought == "second"


def test_parse_output_return_shape_is_consistent():
    ok = parse_output(GOOD)
    bad = parse_output("no answer here")
    assert len(ok) == len(bad) == 4
    assert bad[2] is None or isinstance(bad[2], str)       # method slot never False