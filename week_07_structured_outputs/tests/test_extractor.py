"""Tests for Deliverable 1: docs generator, prompt builder, extraction + retry loop.

Run from week07/:   python -m pytest tests/test_extractor.py -v

Uses YOUR FakeLLM and LLMResponse from llm.py, so no real model is ever called.
FakeLLM records every call in `llm.calls` (system, messages, temperature, max_tokens),
so the tests can check exactly what was sent on each attempt.
"""
import logging
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from extraction.docs_gen import generate_docs, model_table, nested_models, type_name
from extraction.extractor import ExtractionResult, extract
from extraction.prompts import EXAMPLES, build_prompt
from extraction.schemas import Entity, EntityTypes, Event, Extraction, Location
from llm import FakeLLM, LLMResponse

SENTENCE = "EU rejects German call to boycott British lamb ."
VALID = '{"entities": [{"text": "EU", "type": "ORG"}]}'
FENCED_VALID = "```json\n" + VALID + "\n```"
INVALID = '{"entities": [{"text": "EU", "type": "organization"}]}'   # no repair can fix this
UNPARSEABLE = '{"entities": "EU"'
TRUNCATED = '{"entities": [{"text": "EU", "ty'


def resp(text, stop="end_turn", tin=10, tout=5):
    return LLMResponse(text=text, stop_reason=stop, input_tokens=tin, output_tokens=tout)


def prompt_text(call):
    """Everything the model saw on one call: system prompt + all message contents."""
    return (call["system"] or "") + "\n" + "\n".join(m["content"] for m in call["messages"])


# Test-only schema: uses Entity twice, to check it is documented once.
class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    main: Entity
    others: list[Entity] = []


# ===========================================================================
# 1. type_name
# ===========================================================================
class TestTypeName:
    @pytest.mark.parametrize("annotation, words", [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "true or false"),
        (Literal["PER", "ORG", "LOC", "MISC"], "one of: PER, ORG, LOC, MISC"),
        (list[Entity], "list of Entity"),
        (Location, "Location"),
        (str | None, "string or null"),
        (Location | None, "Location or null"),
    ])
    def test_words(self, annotation, words):
        assert type_name(annotation) == words

    def test_real_fields(self):
        names = {n: type_name(f.annotation) for n, f in Event.model_fields.items()}
        assert names == {"actor": "string", "action": "string",
                         "date": "string or null", "location": "Location or null"}


# ===========================================================================
# 2. model_table
# ===========================================================================
class TestModelTable:
    def test_title_and_header(self):
        lines = model_table(Entity).splitlines()
        assert lines[0] == "## Entity"
        assert lines[1] == "| Field | Type | Required | Description |"
        assert lines[2] == "|---|---|---|---|"

    def test_one_row_per_field(self):
        rows = [ln for ln in model_table(Event).splitlines()[3:] if ln.startswith("|")]
        assert len(rows) == len(Event.model_fields)

    def test_required_field_row(self):
        assert "| text | string | yes |" in model_table(Entity)

    def test_default_shown_for_optional_field(self):
        assert "no (default: [])" in model_table(Extraction)
        assert "no (default: None)" in model_table(Event)

    def test_description_included(self):
        assert Entity.model_fields["text"].description in model_table(Entity)

    def test_missing_description_does_not_print_none(self):
        class NoDesc(BaseModel):
            x: str
        assert "None" not in model_table(NoDesc)


# ===========================================================================
# 3. nested_models
# ===========================================================================
class TestNestedModels:
    def test_list_of_models(self):
        assert nested_models(Extraction) == [Entity]

    def test_optional_single_model(self):
        assert nested_models(Event) == [Location]

    def test_no_nesting(self):
        assert nested_models(Entity) == []
        assert nested_models(EntityTypes) == []


# ===========================================================================
# 4. generate_docs
# ===========================================================================
class TestGenerateDocs:
    def test_top_schema_comes_first(self):
        assert generate_docs(Extraction).startswith("## Extraction")

    def test_nested_model_documented(self):
        docs = generate_docs(Extraction)
        assert "## Entity" in docs

    def test_nested_optional_model_documented(self):
        assert "## Location" in generate_docs(Event)

    def test_model_used_twice_documented_once(self):
        assert generate_docs(Pair).count("## Entity") == 1

    @pytest.mark.parametrize("schema", [EntityTypes, Extraction, Event])
    def test_every_field_appears(self, schema):
        docs = generate_docs(schema)
        for model in [schema] + nested_models(schema):
            for name in model.model_fields:
                assert f"| {name} |" in docs

    def test_flat_schema_is_a_single_table(self):
        assert generate_docs(EntityTypes).count("## ") == 1


# ===========================================================================
# 5. Prompts
# ===========================================================================
class TestExamples:
    @pytest.mark.parametrize("schema", [EntityTypes, Extraction, Event])
    def test_every_schema_has_an_example(self, schema):
        assert schema in EXAMPLES

    @pytest.mark.parametrize("schema", list(EXAMPLES))
    def test_every_example_is_valid(self, schema):
        schema.model_validate(EXAMPLES[schema]["output"])

    def test_example_sentence_is_not_the_test_sentence(self):
        for ex in EXAMPLES.values():
            assert ex["text"] != SENTENCE


class TestBuildPrompt:
    def _all(self, text, schema):
        parts = build_prompt(text, schema)
        if isinstance(parts, str):
            return parts
        return "\n".join(p for p in parts if p)

    def test_contains_the_input_text(self):
        assert SENTENCE in self._all(SENTENCE, Extraction)

    def test_contains_the_field_docs(self):
        full = self._all(SENTENCE, Extraction)
        assert "## Extraction" in full and "## Entity" in full

    def test_contains_the_example(self):
        full = self._all(SENTENCE, Extraction)
        assert EXAMPLES[Extraction]["text"] in full
        assert '"Reuters"' in full

    def test_example_is_json_not_python(self):
        # The model copies what it sees. It must see null / false, not None / False.
        full = self._all(SENTENCE, Event)
        assert '"date": null' in full
        assert "'date'" not in full
        full = self._all(SENTENCE, EntityTypes)
        assert '"has_person": false' in full

    def test_example_keeps_schema_field_order(self):
        # Rule 6 / Deliverable 5: field order in the example must match the schema.
        full = self._all(SENTENCE, Event)
        assert full.index('"actor"') < full.index('"action"') < full.index('"date"')

    def test_asks_for_json_only(self):
        assert "JSON" in self._all(SENTENCE, Extraction)

    def test_unknown_schema_raises_clear_error(self):
        with pytest.raises(ValueError):
            build_prompt(SENTENCE, Pair)


# ===========================================================================
# 6. extract: outcomes
# ===========================================================================
class TestOutcomes:
    def test_first_pass(self):
        llm = FakeLLM([resp(VALID)])
        r = extract(SENTENCE, Extraction, llm)
        assert isinstance(r, ExtractionResult)
        assert r.outcome == "first_pass"
        assert isinstance(r.value, Extraction)
        assert len(r.attempts) == 1 and len(llm.calls) == 1

    def test_repaired(self):
        r = extract(SENTENCE, Extraction, FakeLLM([resp(FENCED_VALID)]))
        assert r.outcome == "repaired"
        assert r.value.entities[0].text == "EU"

    def test_retry(self):
        llm = FakeLLM([resp(INVALID), resp(VALID)])
        r = extract(SENTENCE, Extraction, llm)
        assert r.outcome == "retry"
        assert isinstance(r.value, Extraction)
        assert len(r.attempts) == 2 and len(llm.calls) == 2

    def test_retry_beats_repaired(self):
        # Success on attempt 2 is "retry" even if repair also fired on that attempt.
        r = extract(SENTENCE, Extraction, FakeLLM([resp(INVALID), resp(FENCED_VALID)]))
        assert r.outcome == "retry"

    def test_retry_after_parse_failure(self):
        r = extract(SENTENCE, Extraction, FakeLLM([resp(UNPARSEABLE), resp(VALID)]))
        assert r.outcome == "retry"

    def test_failed_uses_every_attempt(self):
        llm = FakeLLM([resp(INVALID)] * 3)
        r = extract(SENTENCE, Extraction, llm, max_retries=2)
        assert r.outcome == "failed"
        assert r.value is None
        assert len(llm.calls) == 3                 # first try + 2 retries
        assert len(r.attempts) == 3

    def test_zero_retries_means_one_call(self):
        llm = FakeLLM([resp(INVALID)])
        r = extract(SENTENCE, Extraction, llm, max_retries=0)
        assert r.outcome == "failed"
        assert len(llm.calls) == 1

    def test_failure_never_raises(self):
        extract(SENTENCE, Extraction, FakeLLM([resp("not json at all")] * 3), max_retries=2)

    def test_works_for_other_schemas(self):
        ev = '{"actor": "EU", "action": "rejects a boycott call", "date": null, "location": null}'
        r = extract(SENTENCE, Event, FakeLLM([resp(ev)]))
        assert r.outcome == "first_pass" and isinstance(r.value, Event)


# ===========================================================================
# 7. extract: what gets sent on each call
# ===========================================================================
class TestMessages:
    def test_first_call_is_one_user_message_with_the_text(self):
        llm = FakeLLM([resp(VALID)])
        extract(SENTENCE, Extraction, llm)
        msgs = llm.calls[0]["messages"]
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        assert SENTENCE in prompt_text(llm.calls[0])

    def test_temperature_is_zero(self):
        llm = FakeLLM([resp(VALID)])
        extract(SENTENCE, Extraction, llm)
        assert llm.calls[0]["temperature"] == 0

    def test_retry_sends_answer_then_feedback(self):
        llm = FakeLLM([resp(INVALID), resp(VALID)])
        r = extract(SENTENCE, Extraction, llm)
        msgs = llm.calls[1]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[1]["content"] == INVALID
        assert msgs[2]["content"] == r.attempts[0].model_message

    def test_retry_keeps_the_original_prompt(self):
        llm = FakeLLM([resp(INVALID), resp(VALID)])
        extract(SENTENCE, Extraction, llm)
        assert llm.calls[1]["messages"][0] == llm.calls[0]["messages"][0]
        assert llm.calls[1]["system"] == llm.calls[0]["system"]

    def test_conversation_grows_by_two_each_retry(self):
        llm = FakeLLM([resp(INVALID)] * 3)
        extract(SENTENCE, Extraction, llm, max_retries=2)
        assert [len(c["messages"]) for c in llm.calls] == [1, 3, 5]

    def test_parse_feedback_mentions_position(self):
        llm = FakeLLM([resp(UNPARSEABLE), resp(VALID)])
        extract(SENTENCE, Extraction, llm)
        assert "line" in llm.calls[1]["messages"][-1]["content"]

    def test_empty_answer_gets_placeholder(self):
        # Claude rejects an assistant message with empty content.
        llm = FakeLLM([resp(""), resp(VALID)])
        extract(SENTENCE, Extraction, llm)
        assistant = llm.calls[1]["messages"][1]
        assert assistant["role"] == "assistant" and assistant["content"].strip() != ""


# ===========================================================================
# 8. extract: truncation
# ===========================================================================
class TestTruncation:
    def test_first_call_uses_given_max_tokens(self):
        llm = FakeLLM([resp(VALID)])
        extract(SENTENCE, Extraction, llm, max_tokens=1024)
        assert llm.calls[0]["max_tokens"] == 1024

    def test_truncation_raises_max_tokens(self):
        llm = FakeLLM([resp(TRUNCATED, stop="max_tokens"), resp(VALID)])
        extract(SENTENCE, Extraction, llm, max_tokens=1024)
        assert llm.calls[1]["max_tokens"] > llm.calls[0]["max_tokens"]

    def test_max_tokens_stays_an_integer(self):
        # APIs reject a float token limit.
        llm = FakeLLM([resp(TRUNCATED, stop="max_tokens"), resp(VALID)])
        extract(SENTENCE, Extraction, llm, max_tokens=1000)
        assert isinstance(llm.calls[1]["max_tokens"], int)

    def test_truncation_does_not_add_feedback(self):
        llm = FakeLLM([resp(TRUNCATED, stop="max_tokens"), resp(VALID)])
        extract(SENTENCE, Extraction, llm)
        assert len(llm.calls[1]["messages"]) == 1

    def test_success_after_truncation_is_retry(self):
        r = extract(SENTENCE, Extraction, FakeLLM([resp(TRUNCATED, stop="max_tokens"), resp(VALID)]))
        assert r.outcome == "retry"

    def test_repeated_truncation_fails_without_crashing(self):
        llm = FakeLLM([resp(TRUNCATED, stop="max_tokens")] * 3)
        r = extract(SENTENCE, Extraction, llm, max_retries=2)
        assert r.outcome == "failed"
        assert 1 <= len(llm.calls) <= 3


# ===========================================================================
# 9. extract: bookkeeping and logging
# ===========================================================================
class TestBookkeeping:
    def test_tokens_added_across_calls(self):
        llm = FakeLLM([resp(INVALID, tin=100, tout=20), resp(VALID, tin=150, tout=25)])
        r = extract(SENTENCE, Extraction, llm)
        assert (r.input_tokens, r.output_tokens) == (250, 45)

    def test_raw_outputs_kept_unrepaired(self):
        llm = FakeLLM([resp(INVALID), resp(FENCED_VALID)])
        r = extract(SENTENCE, Extraction, llm)
        assert r.raw_outputs == [INVALID, FENCED_VALID]

    def test_one_attempt_record_per_call(self):
        llm = FakeLLM([resp(INVALID), resp(UNPARSEABLE), resp(VALID)])
        r = extract(SENTENCE, Extraction, llm, max_retries=2)
        assert [a.failed_stage for a in r.attempts] == ["validation", "parse", None]

    def test_failure_is_logged_with_item_id(self, caplog):
        with caplog.at_level(logging.WARNING):
            extract(SENTENCE, Extraction, FakeLLM([resp(INVALID), resp(VALID)]), item_id="conll_0042")
        assert "conll_0042" in caplog.text
        assert "validation" in caplog.text

    def test_truncation_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            extract(SENTENCE, Extraction,
                    FakeLLM([resp(TRUNCATED, stop="max_tokens"), resp(VALID)]), item_id="conll_0043")
        assert "conll_0043" in caplog.text

    def test_first_pass_logs_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            extract(SENTENCE, Extraction, FakeLLM([resp(VALID)]), item_id="conll_0044")
        assert "conll_0044" not in caplog.text