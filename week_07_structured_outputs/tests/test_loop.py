"""Tests for Deliverable 4, piece 4: run_tool_loop and extract_with_tool.

Run from week07/:   python -m pytest tests/test_loop.py -v

No tokens are spent. FakeToolClient stands in for AnthropicClient: it has the same
complete_with_tools(messages, tools, tool_choice, ...) method, returns scripted
responses in order, and records exactly what was sent on every call.
"""
import copy
from types import SimpleNamespace

import pytest

from extraction.schemas import Extraction
from function_calling import tools
from function_calling.loop import ToolLoopResult, extract_with_tool, run_tool_loop
from function_calling.tools import TOOL_MAPPING

KEYWORD = max(tools.data[0]["sentence"].split(), key=len)
SENTENCE = "EU rejects German call to boycott British lamb ."
FORCED = {"type": "tool", "name": "record_entities"}
GOOD_ENTITIES = {"entities": [{"text": "EU", "type": "ORG"}, {"text": "German", "type": "MISC"}]}
BAD_ENTITIES = {"entities": [{"text": "EU", "type": "organization"}]}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeToolClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete_with_tools(self, messages, tools, tool_choice=None, max_tokens=1024, temperature=0.0, system=None):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "tool_choice": tool_choice})
        if not self._responses:
            raise AssertionError(f"FakeToolClient ran out of scripted responses on call {len(self.calls)}")
        return self._responses.pop(0)


def usage(tin=10, tout=5):
    return SimpleNamespace(input_tokens=tin, output_tokens=tout)


def text_resp(*texts, tin=10, tout=5):
    blocks = [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(content=blocks, stop_reason="end_turn", usage=usage(tin, tout))


def tool_block(name, input, id="toolu_1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def tool_resp(*blocks, tin=10, tout=5, text=None):
    content = ([SimpleNamespace(type="text", text=text)] if text else []) + list(blocks)
    return SimpleNamespace(content=content, stop_reason="tool_use", usage=usage(tin, tout))


def search(id="toolu_1"):
    return tool_block("search_sentences", {"keyword": KEYWORD}, id=id)


def tool_result_messages(call):
    """The user messages in one recorded call that carry tool_results."""
    return [m for m in call["messages"]
            if m["role"] == "user" and isinstance(m["content"], list)
            and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])]


# ===========================================================================
# 1. run_tool_loop: ending normally
# ===========================================================================
class TestLoopEnding:
    def test_returns_result_object(self):
        r = run_tool_loop(FakeToolClient([text_resp("Hi.")]), "Hello", TOOL_MAPPING)
        assert isinstance(r, ToolLoopResult)

    def test_immediate_answer(self):
        client = FakeToolClient([text_resp("No tools needed.")])
        r = run_tool_loop(client, "Hello", TOOL_MAPPING)
        assert r.final_text == "No tools needed."
        assert r.steps == 0
        assert r.hit_max_steps is False
        assert r.tool_calls == []
        assert len(client.calls) == 1

    def test_one_tool_round_then_answer(self):
        client = FakeToolClient([tool_resp(search()), text_resp("Found it.")])
        r = run_tool_loop(client, "Find something", TOOL_MAPPING)
        assert r.final_text == "Found it."
        assert r.steps == 1
        assert r.hit_max_steps is False
        assert len(client.calls) == 2

    def test_final_text_joins_all_text_blocks(self):
        r = run_tool_loop(FakeToolClient([text_resp("Part one. ", "Part two.")]), "Hi", TOOL_MAPPING)
        assert r.final_text == "Part one. Part two."

    def test_first_message_is_the_users_question(self):
        client = FakeToolClient([text_resp("ok")])
        run_tool_loop(client, "What mentions Germany?", TOOL_MAPPING)
        assert client.calls[0]["messages"] == [{"role": "user", "content": "What mentions Germany?"}]

    def test_sends_every_registry_tool(self):
        client = FakeToolClient([text_resp("ok")])
        run_tool_loop(client, "Hi", TOOL_MAPPING)
        assert {t["name"] for t in client.calls[0]["tools"]} == set(TOOL_MAPPING)


# ===========================================================================
# 2. run_tool_loop: max_steps (the Week 6 off-by-one)
# ===========================================================================
class TestMaxSteps:
    def test_tools_run_exactly_max_steps_times(self):
        client = FakeToolClient([tool_resp(search(f"t{i}")) for i in range(4)])
        r = run_tool_loop(client, "Loop forever", TOOL_MAPPING, max_steps=3)
        assert r.steps == 3
        assert r.hit_max_steps is True
        assert r.final_text is None
        assert len(r.tool_calls) == 3                       # 3 tool rounds ran
        assert len(client.calls) == 4                       # the 4th answer asked for a tool and was refused

    def test_last_call_saw_all_three_rounds(self):
        client = FakeToolClient([tool_resp(search(f"t{i}")) for i in range(4)])
        run_tool_loop(client, "Loop forever", TOOL_MAPPING, max_steps=3)
        assert len(tool_result_messages(client.calls[-1])) == 3

    def test_max_steps_zero_runs_no_tools(self):
        client = FakeToolClient([tool_resp(search())])
        r = run_tool_loop(client, "Hi", TOOL_MAPPING, max_steps=0)
        assert r.steps == 0
        assert r.hit_max_steps is True
        assert r.tool_calls == []
        assert len(client.calls) == 1

    def test_answer_on_the_last_allowed_call_is_not_a_limit_hit(self):
        client = FakeToolClient([tool_resp(search("a")), tool_resp(search("b")), text_resp("Done.")])
        r = run_tool_loop(client, "Hi", TOOL_MAPPING, max_steps=2)
        assert r.steps == 2
        assert r.hit_max_steps is False
        assert r.final_text == "Done."


# ===========================================================================
# 3. run_tool_loop: what gets sent back to Claude
# ===========================================================================
class TestLoopMessages:
    def test_assistant_then_tool_results(self):
        first = tool_resp(search("toolu_A"))
        client = FakeToolClient([first, text_resp("ok")])
        run_tool_loop(client, "Find", TOOL_MAPPING)
        msgs = client.calls[1]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[1]["content"] == first.content          # Claude's blocks, as-is
        results = msgs[2]["content"]
        assert isinstance(results, list)
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "toolu_A"

    def test_parallel_calls_answered_in_one_message(self):
        resp = tool_resp(search("toolu_A"), tool_block("search_sentences", {"keyword": "the"}, id="toolu_B"))
        client = FakeToolClient([resp, text_resp("ok")])
        r = run_tool_loop(client, "Find", TOOL_MAPPING)
        results = client.calls[1]["messages"][2]["content"]
        assert [x["tool_use_id"] for x in results] == ["toolu_A", "toolu_B"]
        assert len(r.tool_calls) == 2
        assert r.steps == 1                                 # two calls, one round

    def test_unknown_tool_does_not_crash_the_loop(self):
        client = FakeToolClient([tool_resp(tool_block("delete_everything", {})), text_resp("Sorry.")])
        r = run_tool_loop(client, "Hi", TOOL_MAPPING)
        assert r.final_text == "Sorry."
        assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True

    def test_tool_choice_only_on_first_call(self):
        forced = {"type": "any"}
        client = FakeToolClient([tool_resp(search()), text_resp("ok")])
        run_tool_loop(client, "Find", TOOL_MAPPING, tool_choice=forced)
        assert client.calls[0]["tool_choice"] == forced
        assert client.calls[1]["tool_choice"] is None

    def test_returned_messages_hold_the_conversation(self):
        client = FakeToolClient([tool_resp(search()), text_resp("ok")])
        r = run_tool_loop(client, "Find", TOOL_MAPPING)
        assert [m["role"] for m in r.messages][:3] == ["user", "assistant", "user"]


# ===========================================================================
# 4. run_tool_loop: the log and the tokens
# ===========================================================================
class TestLoopBookkeeping:
    def test_log_has_name_input_and_error_flag(self):
        client = FakeToolClient([tool_resp(search()), text_resp("ok")])
        r = run_tool_loop(client, "Find", TOOL_MAPPING)
        name, inp, is_error = r.tool_calls[0]
        assert name == "search_sentences"
        assert inp == {"keyword": KEYWORD}
        assert is_error is False

    def test_log_marks_failed_calls(self):
        bad = tool_block("search_sentences", {"keyword": "EU", "limit": "five"})
        client = FakeToolClient([tool_resp(bad), text_resp("ok")])
        r = run_tool_loop(client, "Find", TOOL_MAPPING)
        assert r.tool_calls[0][2] is True

    def test_log_spans_all_rounds(self):
        client = FakeToolClient([tool_resp(search("a")), tool_resp(search("b")), text_resp("ok")])
        r = run_tool_loop(client, "Find", TOOL_MAPPING)
        assert len(r.tool_calls) == 2

    def test_tokens_add_up(self):
        client = FakeToolClient([tool_resp(search(), tin=100, tout=20), text_resp("ok", tin=150, tout=30)])
        r = run_tool_loop(client, "Find", TOOL_MAPPING)
        assert (r.input_tokens, r.output_tokens) == (250, 50)


# ===========================================================================
# 5. extract_with_tool
# ===========================================================================
class TestExtractWithTool:
    def test_valid_first_try(self):
        client = FakeToolClient([tool_resp(tool_block("record_entities", GOOD_ENTITIES))])
        r = extract_with_tool(client, SENTENCE)
        assert isinstance(r, Extraction)
        assert [(e.text, e.type) for e in r.entities] == [("EU", "ORG"), ("German", "MISC")]
        assert len(client.calls) == 1

    def test_result_is_claudes_input_not_the_receipt(self):
        client = FakeToolClient([tool_resp(tool_block("record_entities", GOOD_ENTITIES))])
        r = extract_with_tool(client, SENTENCE)
        assert r.model_dump() == GOOD_ENTITIES

    def test_forces_record_entities_every_attempt(self):
        client = FakeToolClient([tool_resp(tool_block("record_entities", BAD_ENTITIES, id="t1")),
                                 tool_resp(tool_block("record_entities", GOOD_ENTITIES, id="t2"))])
        extract_with_tool(client, SENTENCE)
        assert all(c["tool_choice"] == FORCED for c in client.calls)

    def test_only_record_entities_is_offered(self):
        client = FakeToolClient([tool_resp(tool_block("record_entities", GOOD_ENTITIES))])
        extract_with_tool(client, SENTENCE)
        assert [t["name"] for t in client.calls[0]["tools"]] == ["record_entities"]

    def test_callers_prompt_is_sent_unchanged(self):
        # The caller builds the prompt; extract_with_tool sends it as the first user message.
        prompt = "Record every named entity in this text: " + SENTENCE
        client = FakeToolClient([tool_resp(tool_block("record_entities", GOOD_ENTITIES))])
        extract_with_tool(client, prompt)
        assert client.calls[0]["messages"] == [{"role": "user", "content": prompt}]

    def test_invalid_then_valid_is_retried(self):
        client = FakeToolClient([tool_resp(tool_block("record_entities", BAD_ENTITIES, id="t1")),
                                 tool_resp(tool_block("record_entities", GOOD_ENTITIES, id="t2"))])
        r = extract_with_tool(client, SENTENCE)
        assert isinstance(r, Extraction)
        assert len(client.calls) == 2

    def test_retry_sends_error_as_tool_result(self):
        first = tool_resp(tool_block("record_entities", BAD_ENTITIES, id="t1"))
        client = FakeToolClient([first, tool_resp(tool_block("record_entities", GOOD_ENTITIES, id="t2"))])
        extract_with_tool(client, SENTENCE)
        msgs = client.calls[1]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[1]["content"] == first.content
        results = msgs[2]["content"]
        assert isinstance(results, list)                    # the API needs a list of blocks
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "t1"
        assert results[0]["is_error"] is True

    def test_gives_up_after_max_retries(self):
        client = FakeToolClient([tool_resp(tool_block("record_entities", BAD_ENTITIES, id=f"t{i}"))
                                 for i in range(3)])
        assert extract_with_tool(client, SENTENCE, max_retries=2) is None
        assert len(client.calls) == 3

    def test_zero_retries_means_one_call(self):
        client = FakeToolClient([tool_resp(tool_block("record_entities", BAD_ENTITIES))])
        assert extract_with_tool(client, SENTENCE, max_retries=0) is None
        assert len(client.calls) == 1

    def test_no_tool_call_in_response_does_not_crash(self):
        client = FakeToolClient([text_resp("I can't do that.")] * 3)
        assert extract_with_tool(client, SENTENCE, max_retries=2) is None
        assert len(client.calls) <= 3