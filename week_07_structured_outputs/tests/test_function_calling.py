"""Tests for Deliverable 4, pieces 1 and 2: tools.py and formatter.py.

Run from week07/:   python -m pytest tests/test_function_calling.py -v

No real Claude call is made. Claude's tool_use blocks and responses are faked with
SimpleNamespace objects that have the same attributes the real SDK objects have:
    block.type, block.id, block.name, block.input        response.content
"""
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from extraction.schemas import Extraction, GetSentenceInput, SearchInput
from function_calling import tools
from function_calling.formatter import (
    parse_tool_calls,
    run_tool_call,
    run_tool_calls,
    to_claude_tool,
    tools_for_claude,
)
from function_calling.tools import TOOL_MAPPING, get_sentence, record_entities, search_sentences

TOOL_NAMES = {"search_sentences", "get_sentence", "record_entities"}

# Real data from the corpus your tools load, so the tests don't depend on which sentences they are.
FIRST = tools.data[0]
REAL_ID = FIRST["id"]
REAL_SENTENCE = FIRST["sentence"]
KEYWORD = max(REAL_SENTENCE.split(), key=len)          # the longest word in a real sentence
ID_FIELD = next(iter(GetSentenceInput.model_fields))   # whatever you named the id field


def tool_use(name, input, id="toolu_1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def text_block(text="Let me check."):
    return SimpleNamespace(type="text", text=text)


def get_input(sentence_id):
    return GetSentenceInput.model_validate({ID_FIELD: sentence_id})


# ===========================================================================
# 1. Input models
# ===========================================================================
class TestInputModels:
    def test_search_default_limit_is_5(self):
        assert SearchInput.model_validate({"keyword": "EU"}).limit == 5

    @pytest.mark.parametrize("bad", [
        {"keyword": "EU", "limit": "five"},
        {"keyword": "EU", "limit": 50},
        {"keyword": "EU", "limit": 0},
        {"limit": 3},
    ])
    def test_search_rejects_bad_input(self, bad):
        with pytest.raises(ValidationError):
            SearchInput.model_validate(bad)

    def test_get_sentence_requires_an_id(self):
        with pytest.raises(ValidationError):
            GetSentenceInput.model_validate({})

    @pytest.mark.parametrize("model", [SearchInput, GetSentenceInput])
    def test_every_field_has_a_description(self, model):
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name} has no description"


# ===========================================================================
# 2. The tool functions
# ===========================================================================
class TestSearchSentences:
    def test_returns_id_and_sentence(self):
        results = search_sentences(SearchInput(keyword=KEYWORD))
        assert len(results) >= 1
        assert all(set(r) == {"id", "sentence"} for r in results)

    def test_every_result_contains_the_keyword(self):
        for r in search_sentences(SearchInput(keyword=KEYWORD)):
            assert KEYWORD.lower() in r["sentence"].lower()

    def test_ignores_capitalization(self):
        lower = search_sentences(SearchInput(keyword=KEYWORD.lower()))
        upper = search_sentences(SearchInput(keyword=KEYWORD.upper()))
        assert lower == upper

    def test_respects_limit(self):
        assert len(search_sentences(SearchInput(keyword="the", limit=2))) == 2

    def test_default_limit(self):
        assert len(search_sentences(SearchInput(keyword="the"))) == 5

    def test_no_match(self):
        assert search_sentences(SearchInput(keyword="zzqqxx-not-a-word")) == []

    def test_result_is_json_ready(self):
        json.dumps(search_sentences(SearchInput(keyword=KEYWORD)))


class TestGetSentence:
    def test_finds_real_id(self):
        result = get_sentence(get_input(REAL_ID))
        assert result["id"] == REAL_ID
        assert result["sentence"] == REAL_SENTENCE

    def test_unknown_id_raises_value_error(self):
        with pytest.raises(ValueError):
            get_sentence(get_input("no_such_id_123"))

    def test_error_names_the_bad_id(self):
        with pytest.raises(ValueError, match="no_such_id_123"):
            get_sentence(get_input("no_such_id_123"))


class TestRecordEntities:
    def test_reports_the_count(self):
        ex = Extraction.model_validate({"entities": [{"text": "EU", "type": "ORG"},
                                                     {"text": "German", "type": "MISC"}]})
        assert "2" in str(record_entities(ex))

    def test_empty(self):
        assert "0" in str(record_entities(Extraction.model_validate({"entities": []})))


# ===========================================================================
# 3. The registry
# ===========================================================================
class TestRegistry:
    def test_has_all_three_tools(self):
        assert set(TOOL_MAPPING) == TOOL_NAMES

    @pytest.mark.parametrize("name", sorted(TOOL_NAMES))
    def test_entry_is_complete(self, name):
        entry = TOOL_MAPPING[name]
        assert isinstance(entry["description"], str) and len(entry["description"]) > 20
        assert issubclass(entry["input_model"], BaseModel)
        assert callable(entry["function"])

    def test_models_and_functions_match(self):
        assert TOOL_MAPPING["search_sentences"]["input_model"] is SearchInput
        assert TOOL_MAPPING["get_sentence"]["input_model"] is GetSentenceInput
        assert TOOL_MAPPING["record_entities"]["input_model"] is Extraction
        assert TOOL_MAPPING["record_entities"]["function"] is record_entities


# ===========================================================================
# 4. Formatting tools for Claude
# ===========================================================================
class TestToClaudeTool:
    def test_shape(self):
        t = to_claude_tool("search_sentences", "Find sentences.", SearchInput)
        assert set(t) == {"name", "description", "input_schema"}
        assert t["name"] == "search_sentences"
        assert t["description"] == "Find sentences."

    def test_schema_comes_from_the_model(self):
        t = to_claude_tool("search_sentences", "Find sentences.", SearchInput)
        assert t["input_schema"] == SearchInput.model_json_schema()
        assert t["input_schema"]["type"] == "object"


class TestToolsForClaude:
    def test_one_tool_per_registry_entry(self):
        out = tools_for_claude(TOOL_MAPPING)
        assert len(out) == 3
        assert {t["name"] for t in out} == TOOL_NAMES

    def test_descriptions_come_from_the_registry(self):
        for t in tools_for_claude(TOOL_MAPPING):
            assert t["description"] == TOOL_MAPPING[t["name"]]["description"]

    def test_can_be_sent_to_the_api(self):
        json.dumps(tools_for_claude(TOOL_MAPPING))      # must be plain JSON


# ===========================================================================
# 5. Parsing tool calls out of a response
# ===========================================================================
class TestParseToolCalls:
    def test_keeps_only_tool_use_blocks_in_order(self):
        a = tool_use("search_sentences", {"keyword": "EU"}, id="toolu_A")
        b = tool_use("get_sentence", {ID_FIELD: REAL_ID}, id="toolu_B")
        response = SimpleNamespace(content=[text_block(), a, b])
        assert parse_tool_calls(response) == [a, b]

    def test_no_tool_calls(self):
        assert parse_tool_calls(SimpleNamespace(content=[text_block("Final answer.")])) == []


# ===========================================================================
# 6. Running one tool call
# ===========================================================================
class TestRunToolCall:
    def test_success_shape(self):
        r = run_tool_call(tool_use("search_sentences", {"keyword": KEYWORD}, id="toolu_9"), TOOL_MAPPING)
        assert r["type"] == "tool_result"
        assert r["tool_use_id"] == "toolu_9"
        assert isinstance(r["content"], str)
        assert not r.get("is_error", False)

    def test_success_content_is_the_function_output_as_json(self):
        r = run_tool_call(tool_use("search_sentences", {"keyword": KEYWORD}), TOOL_MAPPING)
        assert json.loads(r["content"]) == search_sentences(SearchInput(keyword=KEYWORD))

    def test_get_sentence_through_the_formatter(self):
        r = run_tool_call(tool_use("get_sentence", {ID_FIELD: REAL_ID}), TOOL_MAPPING)
        assert not r.get("is_error", False)
        assert REAL_SENTENCE in json.loads(r["content"])["sentence"]

    def test_record_entities_through_the_formatter(self):
        entities = {"entities": [{"text": "EU", "type": "ORG"}]}
        r = run_tool_call(tool_use("record_entities", entities), TOOL_MAPPING)
        assert not r.get("is_error", False)
        assert isinstance(r["content"], str)

    def test_invalid_input_becomes_error_result(self):
        r = run_tool_call(tool_use("search_sentences", {"keyword": "EU", "limit": "five"}, id="toolu_X"),
                          TOOL_MAPPING)
        assert r["is_error"] is True
        assert r["tool_use_id"] == "toolu_X"
        assert '"limit"' in r["content"]
        assert "'five'" in r["content"]

    def test_one_line_per_validation_error(self):
        r = run_tool_call(tool_use("search_sentences", {"limit": "five"}), TOOL_MAPPING)
        problem_lines = [ln for ln in r["content"].splitlines() if ln.startswith("- ")]
        assert len(problem_lines) == 2           # keyword missing + limit not an integer

    def test_nested_error_location_is_one_indexed(self):
        bad = {"entities": [{"text": "EU", "type": "ORG"}, {"text": "German", "type": "nationality"}]}
        r = run_tool_call(tool_use("record_entities", bad), TOOL_MAPPING)
        assert r["is_error"] is True
        assert "item 2" in r["content"]

    def test_error_message_has_no_urls(self):
        r = run_tool_call(tool_use("search_sentences", {"keyword": "EU", "limit": 50}), TOOL_MAPPING)
        assert "http" not in r["content"]

    def test_function_error_becomes_error_result(self):
        r = run_tool_call(tool_use("get_sentence", {ID_FIELD: "no_such_id_123"}), TOOL_MAPPING)
        assert r["is_error"] is True
        assert "no_such_id_123" in r["content"]

    def test_unknown_tool_becomes_error_result_not_an_exception(self):
        r = run_tool_call(tool_use("delete_everything", {}), TOOL_MAPPING)
        assert r["is_error"] is True
        assert r["type"] == "tool_result"
        for name in TOOL_NAMES:
            assert name in r["content"]          # tells Claude which tools DO exist

    def test_input_that_is_not_a_dict_does_not_crash(self):
        r = run_tool_call(tool_use("search_sentences", "EU"), TOOL_MAPPING)
        assert r["is_error"] is True


# ===========================================================================
# 7. Running several tool calls
# ===========================================================================
class TestRunToolCalls:
    def test_one_result_per_call_in_order(self):
        blocks = [tool_use("search_sentences", {"keyword": KEYWORD}, id="toolu_A"),
                  tool_use("get_sentence", {ID_FIELD: REAL_ID}, id="toolu_B")]
        results = run_tool_calls(blocks, TOOL_MAPPING)
        assert [r["tool_use_id"] for r in results] == ["toolu_A", "toolu_B"]

    def test_one_failure_does_not_stop_the_others(self):
        blocks = [tool_use("get_sentence", {ID_FIELD: "no_such_id_123"}, id="toolu_A"),
                  tool_use("search_sentences", {"keyword": KEYWORD}, id="toolu_B")]
        results = run_tool_calls(blocks, TOOL_MAPPING)
        assert results[0]["is_error"] is True
        assert not results[1].get("is_error", False)

    def test_every_call_gets_an_answer(self):
        blocks = [tool_use("delete_everything", {}, id="toolu_A"),
                  tool_use("search_sentences", {"limit": "five"}, id="toolu_B"),
                  tool_use("get_sentence", {ID_FIELD: REAL_ID}, id="toolu_C")]
        assert len(run_tool_calls(blocks, TOOL_MAPPING)) == 3

    def test_no_calls(self):
        assert run_tool_calls([], TOOL_MAPPING) == []