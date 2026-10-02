"""
AI generated Test cases
"""
import pytest

import prompting.react as react_module
from prompting.react import run_react, ReActResults, ReActStep
from prompting.parsing import parse_react
from prompting.llm import FakeLLM, LLMResponse


# ─────────────────────────────── helpers ───────────────────────────────

@pytest.fixture(autouse=True)
def stub_fallback_templates(monkeypatch):
    monkeypatch.setattr(react_module, "render", lambda name, **kw: f"FALLBACK PROMPT ({name})")


def R(text, stop="end_turn", i=10, o=5, t=0.1):
    return LLMResponse(text=text, stop_reason=stop, input_tokens=i, output_tokens=o, latency_s=t)


def ACT(tool, arg, thought="thinking", **kw):
    return R(f"<thought>{thought}</thought><action>{tool}</action><input>{arg}</input>", **kw)


def FIN(answer, thought="done", **kw):
    return R(f"<thought>{thought}</thought><finish>{answer}</finish>", **kw)


QUESTION = "Question: Were Scott Derrickson and Ed Wood of the same nationality?"


def make_tools(log=None):
    def search(arg):
        if log is not None:
            log.append(("search", arg))
        return f"PARAGRAPH ABOUT {arg.upper()}"

    def lookup(arg):
        if log is not None:
            log.append(("lookup", arg))
        return f"SENTENCE WITH {arg}"

    return {"search": search, "lookup": lookup}


def run(replies, tools=None, max_steps=8, allowed_repeats=None, **kw):
    fake = FakeLLM(replies)
    res = run_react(
        fake, "SYS", QUESTION, parse_react, tools or make_tools(),
        max_steps, allowed_repeats if allowed_repeats is not None else {"lookup"}, **kw,
    )
    return res, fake


def text_of(call):
    return "\n".join(m["content"] for m in call["messages"])


# ─────────────────────────────── finishing ─────────────────────────────

class TestFinish:
    def test_immediate_finish(self):
        res, fake = run([FIN("yes")])
        assert isinstance(res, ReActResults)
        assert res.status == "ok"
        assert res.answer == "yes"
        assert res.used_fallback is False
        assert len(fake.calls) == 1
        assert len(res.steps) == 1
        assert isinstance(res.steps[0], ReActStep)

    def test_search_then_finish(self):
        log = []
        res, fake = run([ACT("search", "Ed Wood"), FIN("yes")], tools=make_tools(log))
        assert res.status == "ok"
        assert res.answer == "yes"
        assert log == [("search", "Ed Wood")]
        assert len(res.steps) == 2
        assert res.steps[0].tool == "search"
        assert res.steps[0].observation == "PARAGRAPH ABOUT ED WOOD"

    def test_first_message_is_the_question(self):
        _, fake = run([FIN("yes")])
        first = fake.calls[0]["messages"][0]
        assert first["role"] == "user"
        assert first["content"] == QUESTION
        assert fake.calls[0]["system"] == "SYS"


# ─────────────────────────────── observations & history ────────────────

class TestHistory:
    def test_observation_reaches_next_call_in_tags(self):
        _, fake = run([ACT("search", "Ed Wood"), FIN("yes")])
        assert "<observation>PARAGRAPH ABOUT ED WOOD</observation>" in text_of(fake.calls[1])

    def test_assistant_turn_is_kept_text_not_raw(self):
        reply = R("<thought>t</thought><action>search</action><input>Ed Wood</input>"
                  "\n<observation>Ed Wood was a Canadian filmmaker.</observation>")
        _, fake = run([reply, FIN("yes")])
        second = text_of(fake.calls[1])
        assert "Canadian" not in second
        assert "PARAGRAPH ABOUT ED WOOD" in second
        roles = [m["role"] for m in fake.calls[1]["messages"]]
        assert roles == ["user", "assistant", "user"]

    def test_tool_receives_original_arg_not_lowercased(self):
        log = []
        run([ACT("search", "Ed Wood"), FIN("yes")], tools=make_tools(log))
        assert log == [("search", "Ed Wood")]

    def test_history_grows_across_steps(self):
        _, fake = run([ACT("search", "Ed Wood"), ACT("search", "Scott Derrickson"), FIN("yes")])
        third = text_of(fake.calls[2])
        assert "PARAGRAPH ABOUT ED WOOD" in third
        assert "PARAGRAPH ABOUT SCOTT DERRICKSON" in third


# ─────────────────────────────── bad tools ─────────────────────────────

class TestToolProblems:
    def test_unknown_tool_lists_valid_tools(self):
        res, fake = run([ACT("google", "Ed Wood"), FIN("yes")])
        assert res.status == "ok"
        second = text_of(fake.calls[1])
        assert "search" in second and "lookup" in second

    def test_unknown_tool_is_not_called(self):
        log = []
        run([ACT("google", "Ed Wood"), FIN("yes")], tools=make_tools(log))
        assert log == []

    def test_tool_crash_does_not_kill_the_run(self):
        def broken(arg):
            raise RuntimeError("index out of range in search")

        tools = {"search": broken, "lookup": make_tools()["lookup"]}
        res, fake = run([ACT("search", "Ed Wood"), FIN("no")], tools=tools)
        assert res.status == "ok"
        assert res.answer == "no"
        second = text_of(fake.calls[1])
        assert "Tool error" in second
        assert "index out of range in search" in second


# ─────────────────────────────── repeats ───────────────────────────────

class TestRepeats:
    def test_first_repeat_sends_warning_and_does_not_rerun_tool(self):
        log = []
        res, fake = run([ACT("search", "Ed Wood"), ACT("search", "Ed Wood"), FIN("yes")],
                        tools=make_tools(log))
        assert res.status == "ok"
        assert log == [("search", "Ed Wood")]           # tool ran once
        warning_step = res.steps[1]
        assert warning_step.observation
        assert warning_step.observation in text_of(fake.calls[2])   # warning reached the model

    def test_repeat_detection_ignores_case(self):
        log = []
        run([ACT("search", "Ed Wood"), ACT("search", "ed wood"), FIN("yes")], tools=make_tools(log))
        assert len(log) == 1

    def test_second_repeat_stops_with_loop(self):
        res, fake = run([ACT("search", "Ed Wood")] * 3 + [FIN("yes")])
        assert res.status == "loop"
        assert res.used_fallback is True
        assert len(fake.calls) == 4                     # 3 steps + fallback
        assert res.answer == "yes"

    def test_non_adjacent_repeat_is_still_a_repeat(self):
        log = []
        run([ACT("search", "A"), ACT("search", "B"), ACT("search", "A"), FIN("x")],
            tools=make_tools(log))
        assert log == [("search", "A"), ("search", "B")]

    def test_allowed_repeats_are_never_blocked(self):
        log = []
        res, _ = run([ACT("lookup", "born")] * 3 + [FIN("1966")],
                     tools=make_tools(log), allowed_repeats={"lookup"})
        assert res.status == "ok"
        assert log == [("lookup", "born")] * 3


# ─────────────────────────────── format failures ───────────────────────

class TestFormatFailures:
    def test_repair_then_success(self):
        res, fake = run([R("I think the answer is yes"), FIN("yes")])
        assert res.status == "ok"
        assert len(fake.calls) == 2
        assert res.steps[-1].attempts == 2

    def test_repair_message_contains_parser_reason(self):
        bad = "I think the answer is yes"
        reason = parse_react(bad).reason
        _, fake = run([R(bad), FIN("yes")])
        assert reason.strip()[:40] in text_of(fake.calls[1])

    def test_format_failure_after_max_errors(self):
        res, fake = run([R("garbage")] * 3 + [FIN("no")], max_format_errors=3)
        assert res.status == "format_failure"
        assert res.used_fallback is True
        assert len(fake.calls) == 4                     # 3 attempts + fallback
        assert res.answer == "no"

    def test_failed_repairs_do_not_leak_into_later_steps(self):
        _, fake = run([R("garbage one"), ACT("search", "Ed Wood"), FIN("yes")])
        assert "garbage one" not in text_of(fake.calls[2])

    def test_truncated_reply_with_complete_action_is_used(self):
        log = []
        res, _ = run([ACT("search", "Ed Wood", stop="max_tokens"), FIN("yes")], tools=make_tools(log))
        assert log == [("search", "Ed Wood")]
        assert res.status == "ok"


# ─────────────────────────────── limits ────────────────────────────────

class TestLimits:
    def test_max_steps(self):
        replies = [ACT("search", f"entity {i}") for i in range(3)] + [FIN("maybe")]
        res, fake = run(replies, max_steps=3)
        assert res.status == "max_steps"
        assert res.used_fallback is True
        assert len(fake.calls) == 4
        assert res.answer == "maybe"

    def test_context_full(self):
        res, fake = run([ACT("search", "Ed Wood", i=8000, o=10), FIN("yes")],
                        num_ctx=8192, ctx_margin=256)
        assert res.status == "context_full"
        assert res.used_fallback is True
        assert len(fake.calls) == 2

    def test_context_just_below_limit_is_fine(self):
        res, _ = run([ACT("search", "Ed Wood", i=7900, o=10), FIN("yes")],
                     num_ctx=8192, ctx_margin=256)
        assert res.status == "ok"


# ─────────────────────────────── fallback ──────────────────────────────

class TestFallback:
    def test_fallback_without_finish_gives_none(self):
        res, _ = run([ACT("search", f"e{i}") for i in range(2)] + [R("I really can't say")], max_steps=2)
        assert res.status == "max_steps"
        assert res.used_fallback is True
        assert res.answer is None

    def test_fallback_is_recorded_as_last_step(self):
        res, fake = run([ACT("search", f"e{i}") for i in range(2)] + [FIN("maybe")], max_steps=2)
        assert len(res.steps) == len(fake.calls)
        assert res.steps[-1].tool is None

    def test_fallback_step_recorded_for_format_failure(self):
        res, fake = run([R("garbage")] * 3 + [FIN("no")])
        assert res.steps[-1].tool is None
        assert sum(s.input_tokens for s in res.steps) == sum(10 for _ in fake.calls)


# ─────────────────────────────── accounting ────────────────────────────

class TestAccounting:
    def test_tokens_cover_every_call(self):
        replies = [R("garbage", i=5, o=1), ACT("search", "Ed Wood", i=20, o=2), FIN("yes", i=30, o=3)]
        res, _ = run(replies)
        assert sum(s.input_tokens for s in res.steps) == 55
        assert sum(s.output_tokens for s in res.steps) == 6
        assert sum(s.latency_s for s in res.steps) == pytest.approx(0.3)

    def test_repair_tokens_land_on_the_same_step(self):
        res, _ = run([R("garbage", i=5, o=1), FIN("yes", i=30, o=3)])
        assert len(res.steps) == 1
        assert res.steps[0].input_tokens == 35