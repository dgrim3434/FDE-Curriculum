"""
AI Generated Test Cases for Chaining logic
"""

import pytest

from prompting.chain import run_chain, Step, StepRecord, ChainResult
from prompting.llm import FakeLLM, LLMResponse
from prompting.parsing import make_text_parser, make_label_parser


# ─────────────────────────────── helpers ───────────────────────────────

def R(text, stop="end_turn", i=10, o=5):
    """A scripted model reply."""
    return LLMResponse(text=text, stop_reason=stop, input_tokens=i, output_tokens=o, latency_s=0.1)


def make_step(name, output_key, next=None, prompt=None, fn=None, parser=None, **kw):
    settings = dict(system="SYS-" + name, temperature=0.0, max_tokens=256, max_repairs=1)
    settings.update(kw)
    if prompt is None and fn is None:
        prompt = lambda s, n=name: f"prompt for {n}"
    return Step(
        name=name,
        build_prompt=prompt,
        parser=parser or make_text_parser("answer"),
        output_key=output_key,
        next=next,
        fn=fn,
        **settings,
    )


def all_message_texts(call):
    return [m.get("content", "") for m in call["messages"]]


# ─────────────────────────────── linear flow ───────────────────────────

class TestLinear:
    def test_two_steps_run_in_order(self):
        steps = {
            "a": make_step("a", "a_out", next="b"),
            "b": make_step("b", "b_out", next=None),
        }
        fake = FakeLLM([R("thinking... <answer>A</answer>"), R("<answer>B</answer>")])
        res = run_chain(steps, "a", {"question": "q"}, fake)

        assert isinstance(res, ChainResult)
        assert res.status == "ok"
        assert [r.name for r in res.trace] == ["a", "b"]
        assert res.state["a_out"] == "A" and res.state["b_out"] == "B"
        assert res.output == "B"
        assert len(fake.calls) == 2
        assert all(isinstance(r, StepRecord) for r in res.trace)

    def test_parsed_value_flows_forward_not_raw_text(self):
        steps = {
            "a": make_step("a", "a_out", next="b"),
            "b": make_step("b", "b_out", prompt=lambda s: f"previous answer was {s['a_out']}"),
        }
        fake = FakeLLM([R("let me think it through carefully <answer>A</answer>"), R("<answer>B</answer>")])
        run_chain(steps, "a", {}, fake)

        second_prompt = " ".join(all_message_texts(fake.calls[1]))
        assert "previous answer was A" in second_prompt
        assert "think it through" not in second_prompt

    def test_initial_state_visible_to_first_prompt(self):
        steps = {"a": make_step("a", "a_out", prompt=lambda s: f"Q: {s['question']}")}
        fake = FakeLLM([R("<answer>x</answer>")])
        run_chain(steps, "a", {"question": "Were they the same nationality?"}, fake)
        assert "Were they the same nationality?" in " ".join(all_message_texts(fake.calls[0]))

    def test_messages_are_well_formed(self):
        steps = {"a": make_step("a", "a_out", next="b"), "b": make_step("b", "b_out")}
        fake = FakeLLM([R("<answer>A</answer>"), R("<answer>B</answer>")])
        run_chain(steps, "a", {}, fake)
        for call in fake.calls:
            assert call["messages"], "messages list is empty"
            assert call["messages"][0]["role"] == "user"
            for m in call["messages"]:
                assert set(m) >= {"role", "content"}, f"bad message dict: {m}"

    def test_step_settings_are_passed_to_llm(self):
        steps = {"a": make_step("a", "a_out", system="You classify.", temperature=0.3, max_tokens=99)}
        fake = FakeLLM([R("<answer>A</answer>")])
        run_chain(steps, "a", {}, fake)
        call = fake.calls[0]
        assert call["system"] == "You classify."
        assert call["temperature"] == 0.3
        assert call["max_tokens"] == 99

    def test_trace_records_prompt_raw_value_and_next(self):
        steps = {"a": make_step("a", "a_out", next="b"), "b": make_step("b", "b_out")}
        fake = FakeLLM([R("raw A <answer>A</answer>"), R("<answer>B</answer>")])
        res = run_chain(steps, "a", {}, fake)
        rec = res.trace[0]
        assert rec.prompt == "prompt for a"
        assert rec.raw == "raw A <answer>A</answer>"
        assert rec.value == "A"
        assert rec.attempts == 1
        assert rec.error is None
        assert rec.next == "b"
        assert res.trace[1].next is None


# ─────────────────────────────── branching ─────────────────────────────

def _router_chain():
    route = lambda s: "compare" if s["qtype"] == "comparison" else "bridge"
    return {
        "router": make_step("router", "qtype", next=route,
                            parser=make_label_parser("type", {"comparison", "bridge"})),
        "compare": make_step("compare", "answer"),
        "bridge": make_step("bridge", "answer"),
    }


class TestBranching:
    def test_comparison_branch(self):
        fake = FakeLLM([R("<type>Comparison</type>"), R("<answer>yes</answer>")])
        res = run_chain(_router_chain(), "router", {}, fake)
        assert res.status == "ok"
        assert [r.name for r in res.trace] == ["router", "compare"]

    def test_bridge_branch(self):
        fake = FakeLLM([R("<type>bridge</type>"), R("<answer>Paris</answer>")])
        res = run_chain(_router_chain(), "router", {}, fake)
        assert [r.name for r in res.trace] == ["router", "bridge"]
        assert res.output == "Paris"

    def test_trace_next_is_resolved_name_not_function(self):
        fake = FakeLLM([R("<type>bridge</type>"), R("<answer>Paris</answer>")])
        res = run_chain(_router_chain(), "router", {}, fake)
        assert res.trace[0].next == "bridge"

    def test_callable_next_returning_unknown_name_fails_cleanly(self):
        steps = {"a": make_step("a", "a_out", next=lambda s: "does_not_exist")}
        fake = FakeLLM([R("<answer>A</answer>")])
        res = run_chain(steps, "a", {}, fake)
        assert res.status == "failed"


# ─────────────────────────────── code steps ────────────────────────────

class TestCodeSteps:
    def test_code_step_makes_no_llm_call(self):
        steps = {"fmt": make_step("fmt", "upper", fn=lambda s: s["question"].upper())}
        fake = FakeLLM([])
        res = run_chain(steps, "fmt", {"question": "abc"}, fake)
        assert res.status == "ok"
        assert res.state["upper"] == "ABC"
        assert fake.calls == []
        assert res.trace[0].input_tokens == 0 and res.trace[0].output_tokens == 0

    def test_code_step_then_llm_step(self):
        steps = {
            "fmt": make_step("fmt", "upper", fn=lambda s: s["question"].upper(), next="ask"),
            "ask": make_step("ask", "answer", prompt=lambda s: f"text: {s['upper']}"),
        }
        fake = FakeLLM([R("<answer>ok</answer>")])
        res = run_chain(steps, "fmt", {"question": "abc"}, fake)
        assert "text: ABC" in " ".join(all_message_texts(fake.calls[0]))
        assert res.output == "ok"


# ─────────────────────────────── repair ────────────────────────────────

class TestRepair:
    def test_bad_reply_is_repaired(self):
        steps = {"a": make_step("a", "a_out", max_repairs=1)}
        fake = FakeLLM([R("I forgot the tag"), R("<answer>A</answer>")])
        res = run_chain(steps, "a", {}, fake)
        assert res.status == "ok"
        assert res.state["a_out"] == "A"
        assert len(fake.calls) == 2
        assert res.trace[0].attempts == 2

    def test_repair_messages_contain_bad_reply_and_parser_error(self):
        parser = make_text_parser("answer")
        expected_error = parser("I forgot the tag").error
        steps = {"a": make_step("a", "a_out", parser=parser, max_repairs=1)}
        fake = FakeLLM([R("I forgot the tag"), R("<answer>A</answer>")])
        run_chain(steps, "a", {}, fake)

        msgs = fake.calls[1]["messages"]
        roles = [m["role"] for m in msgs]
        texts = [m["content"] for m in msgs]
        assert roles[0] == "user" and "assistant" in roles
        assert "I forgot the tag" in texts
        assert any(expected_error in t for t in texts)
        assert msgs[-1]["role"] == "user"

    def test_repairs_are_bounded(self):
        steps = {
            "a": make_step("a", "a_out", next="b", max_repairs=2),
            "b": make_step("b", "b_out"),
        }
        fake = FakeLLM([R("garbage")] * 10)
        res = run_chain(steps, "a", {}, fake)
        assert len(fake.calls) == 3                     # 1 try + 2 repairs
        assert res.status == "failed"
        assert res.output is None
        assert "a_out" not in res.state                 # nothing written for a failed step
        assert [r.name for r in res.trace] == ["a"]     # b never ran
        assert res.trace[-1].error
        assert res.trace[-1].value is None

    def test_zero_repairs_means_one_attempt(self):
        steps = {"a": make_step("a", "a_out", max_repairs=0)}
        fake = FakeLLM([R("garbage")] * 5)
        res = run_chain(steps, "a", {}, fake)
        assert len(fake.calls) == 1
        assert res.status == "failed"


# ─────────────────────────────── truncation ────────────────────────────

class TestTruncation:
    def test_truncated_reply_is_never_used(self):
        steps = {"a": make_step("a", "a_out", max_repairs=1)}
        fake = FakeLLM([R("<answer>WRONG</answer>", stop="max_tokens"), R("<answer>A</answer>")])
        res = run_chain(steps, "a", {}, fake)
        assert res.status == "ok"
        assert res.state["a_out"] == "A"
        assert len(fake.calls) == 2

    def test_truncation_uses_up_attempts(self):
        steps = {"a": make_step("a", "a_out", max_repairs=1)}
        fake = FakeLLM([R("<answer>x</answer>", stop="max_tokens")] * 5)
        res = run_chain(steps, "a", {}, fake)
        assert len(fake.calls) == 2
        assert res.status == "failed"
        assert "a_out" not in res.state


# ─────────────────────────────── validation & limits ───────────────────

class TestValidationAndLimits:
    def test_unknown_string_next_raises_before_any_call(self):
        steps = {"a": make_step("a", "a_out", next="typo_step")}
        fake = FakeLLM([R("<answer>A</answer>")])
        with pytest.raises(ValueError):
            run_chain(steps, "a", {}, fake)
        assert fake.calls == []

    def test_unknown_start_raises_before_any_call(self):
        steps = {"a": make_step("a", "a_out")}
        fake = FakeLLM([R("<answer>A</answer>")])
        with pytest.raises(ValueError):
            run_chain(steps, "nope", {}, fake)
        assert fake.calls == []

    def test_max_steps_stops_a_loop(self):
        steps = {"a": make_step("a", "a_out", next="a")}           # loops forever
        fake = FakeLLM([R("<answer>A</answer>")] * 50)
        res = run_chain(steps, "a", {}, fake, max_steps=5)
        assert res.status == "failed"
        assert len(res.trace) == 5
        assert len(fake.calls) == 5


# ─────────────────────────────── state & accounting ────────────────────

class TestStateAndAccounting:
    def test_initial_state_not_modified(self):
        initial = {"question": "q", "context": {"title": ["t"]}}
        steps = {"a": make_step("a", "a_out")}
        run_chain(steps, "a", initial, FakeLLM([R("<answer>A</answer>")]))
        assert initial == {"question": "q", "context": {"title": ["t"]}}

    def test_two_runs_do_not_share_state(self):
        steps = {"a": make_step("a", "a_out")}
        first = run_chain(steps, "a", {"question": "q"}, FakeLLM([R("<answer>one</answer>")]))
        second = run_chain(steps, "a", {"question": "q"}, FakeLLM([R("<answer>two</answer>")]))
        assert first.state["a_out"] == "one"
        assert second.state["a_out"] == "two"
        assert len(second.trace) == 1

    def test_tokens_recorded_per_step(self):
        steps = {"a": make_step("a", "a_out", next="b"), "b": make_step("b", "b_out")}
        fake = FakeLLM([R("<answer>A</answer>", i=11, o=5), R("<answer>B</answer>", i=12, o=7)])
        res = run_chain(steps, "a", {}, fake)
        assert (res.trace[0].input_tokens, res.trace[0].output_tokens) == (11, 5)
        assert (res.trace[1].input_tokens, res.trace[1].output_tokens) == (12, 7)
        assert sum(r.input_tokens for r in res.trace) == 23

    def test_tokens_summed_across_repair_attempts(self):
        steps = {"a": make_step("a", "a_out", max_repairs=1)}
        fake = FakeLLM([R("garbage", i=10, o=3), R("<answer>A</answer>", i=20, o=4)])
        res = run_chain(steps, "a", {}, fake)
        assert res.trace[0].input_tokens == 30
        assert res.trace[0].output_tokens == 7
        assert res.trace[0].latency_s == pytest.approx(0.2)