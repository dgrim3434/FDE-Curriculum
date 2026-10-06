"""Tests for Deliverable 3: validation and repair pipeline.

Run from week07/:   python -m pytest tests/test_repair.py -v

Layout these tests assume (yours):
  extraction/schemas.py  -> Entity, Extraction, wrap_list_field, normalize_enum, nullify,
                            walk, validate_with_repairs
  extraction/repair.py   -> strip_fences, extract_json_span, trailing_comma_removal,
                            parse_python_literal, parse_with_repairs, run_pipeline, describe_loc

Rule names: every rule records the SAME name as its function, except the parse
fallback, which records "python_literal". If you pick different names, change
them in one place: the constants below.
"""
import json
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from extraction.repair import (
    describe_loc,
    extract_json_span,
    parse_python_literal,
    parse_with_repairs,
    run_pipeline,
    strip_fences,
    trailing_comma_removal,
)
from extraction.schemas import (
    Entity,
    Extraction,
    normalize_enum,
    nullify,
    validate_with_repairs,
    walk,
    wrap_list_field,
)

STRIP = "strip_fences"
SPAN = "extract_json_span"
COMMA = "trailing_comma_removal"
PYLIT = "python_literal"
WRAP = "wrap_list_field"
ENUM = "normalize_enum"
NULL = "nullify"

LABELS = ("PER", "ORG", "LOC", "MISC")


# ---------------------------------------------------------------------------
# Test-only schemas: they exercise nullable fields, a nested optional model,
# a Literal | None field, and a schema with two list fields.
# ---------------------------------------------------------------------------
class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str | None = None
    country: str | None = None


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor: str
    action: str
    date: str | None = None
    location: Location | None = None
    category: Literal["politics", "trade", "other"] | None = None


class TwoLists(BaseModel):
    model_config = ConfigDict(extra="forbid")
    people: list[Entity] = []
    places: list[Entity] = []


# ===========================================================================
# 0. Your schemas
# ===========================================================================
class TestSchemas:
    def test_entity_accepts_valid(self):
        e = Entity.model_validate({"text": "EU", "type": "ORG"})
        assert e.text == "EU" and e.type == "ORG"

    def test_entity_rejects_unknown_label(self):
        with pytest.raises(Exception):
            Entity.model_validate({"text": "EU", "type": "organization"})

    def test_entity_forbids_extra_fields(self):
        with pytest.raises(Exception):
            Entity.model_validate({"text": "EU", "type": "ORG", "confidence": 0.9})

    def test_entity_text_is_required(self):
        with pytest.raises(Exception):
            Entity.model_validate({"type": "ORG"})

    def test_extraction_defaults_to_empty_list(self):
        assert Extraction.model_validate({}).entities == []

    def test_extraction_forbids_extra_fields(self):
        with pytest.raises(Exception):
            Extraction.model_validate({"entities": [], "notes": "x"})


# ===========================================================================
# 1. Text rules: fires on broken input / silent on clean input / no damage
# ===========================================================================
class TestStripFences:
    def test_fires_on_json_fence(self):
        assert strip_fences('```json\n{"a": 1}\n```') == ('{"a": 1}', True)

    def test_fires_on_bare_fence(self):
        assert strip_fences('```\n{"a": 1}\n```') == ('{"a": 1}', True)

    def test_silent_on_clean_json(self):
        assert strip_fences('{"a": 1}') == ('{"a": 1}', False)

    def test_silent_when_text_before_fence(self):
        # Anchored at the start on purpose: preambles are extract_json_span's job.
        text = 'Here:\n```json\n{"a": 1}\n```'
        assert strip_fences(text) == (text, False)

    def test_no_damage_to_backticks_inside_a_value(self):
        text = '{"code": "```"}'
        assert strip_fences(text) == (text, False)


class TestExtractJsonSpan:
    def test_fires_on_preamble_and_postamble(self):
        assert extract_json_span('Here you go: {"a": 1} Hope this helps!') == ('{"a": 1}', True)

    def test_fires_on_list(self):
        assert extract_json_span("Result: [1, 2]") == ("[1, 2]", True)

    def test_silent_on_clean_json(self):
        assert extract_json_span('{"a": 1}') == ('{"a": 1}', False)

    def test_silent_when_no_brackets(self):
        assert extract_json_span("no json here") == ("no json here", False)

    def test_silent_when_close_comes_before_open(self):
        assert extract_json_span("} {") == ("} {", False)

    def test_no_damage_to_brace_inside_a_value(self):
        text = '{"a": "}"}'
        assert extract_json_span(text) == (text, False)


class TestTrailingCommaRemoval:
    def test_fires_in_list(self):
        assert trailing_comma_removal("[1, 2,]") == ("[1, 2]", True)

    def test_fires_in_object(self):
        assert trailing_comma_removal('{"a": 1,}') == ('{"a": 1}', True)

    def test_fires_with_whitespace_before_bracket(self):
        assert trailing_comma_removal('{"a": 1, \n }') == ('{"a": 1}', True)

    def test_silent_on_clean_json(self):
        assert trailing_comma_removal("[1, 2]") == ("[1, 2]", False)

    def test_no_damage_to_comma_inside_a_value(self):
        text = '{"note": "a, b"}'
        assert trailing_comma_removal(text) == (text, False)


class TestParsePythonLiteral:
    def test_reads_single_quotes(self):
        assert parse_python_literal("{'a': 1}") == {"a": 1}

    def test_reads_python_none_and_true(self):
        assert parse_python_literal("{'a': None, 'b': True}") == {"a": None, "b": True}

    def test_keeps_apostrophe_inside_string(self):
        assert parse_python_literal("{'text': \"EU's\"}") == {"text": "EU's"}

    def test_scalar_is_rejected(self):
        assert parse_python_literal("5") is None

    def test_garbage_is_rejected(self):
        assert parse_python_literal("not python") is None

    def test_unfinished_is_rejected(self):
        assert parse_python_literal("{'a': ") is None

    def test_never_runs_code(self):
        assert parse_python_literal("__import__('os').getcwd()") is None


# ===========================================================================
# 2. Stage 1: parse_with_repairs
# ===========================================================================
class TestParseWithRepairs:
    def test_clean_json_fires_nothing(self):
        r = parse_with_repairs('{"entities": []}')
        assert r.ok and r.data == {"entities": []}
        assert r.rules_fired == []
        assert r.failed_stage is None

    def test_valid_json_is_never_touched(self):
        # Valid JSON containing ",}" inside a string must come through unchanged.
        r = parse_with_repairs('{"text": "a,}"}')
        assert r.ok and r.data == {"text": "a,}"}
        assert r.rules_fired == []

    @pytest.mark.parametrize("reason", ["length", "max_tokens"])
    def test_truncation_returns_immediately(self, reason):
        r = parse_with_repairs('{"entities": [{"text": "Ger', done_reason=reason)
        assert not r.ok
        assert r.failed_stage == "truncated"
        assert r.rules_fired == []

    def test_normal_stop_is_not_truncation(self):
        r = parse_with_repairs('{"entities": []}', done_reason="stop")
        assert r.ok

    def test_rules_recorded_in_order(self):
        r = parse_with_repairs('```json\n{"a": [1, 2,]}\n```')
        assert r.ok and r.data == {"a": [1, 2]}
        assert r.rules_fired == [STRIP, COMMA]

    def test_python_literal_fallback(self):
        r = parse_with_repairs("{'a': None, 'b': True}")
        assert r.ok and r.data == {"a": None, "b": True}
        assert r.rules_fired == [PYLIT]

    def test_parse_failure_reports_stage(self):
        r = parse_with_repairs('{"entities": "EU"')
        assert not r.ok
        assert r.failed_stage == "parse"
        assert r.data is None

    def test_parse_failure_error_is_text_with_position(self):
        r = parse_with_repairs('{"entities": "EU"')
        assert isinstance(r.error, str)
        assert "line" in r.error and "column" in r.error


# ===========================================================================
# 3. Value rules (single value in, (value, changed) out)
# ===========================================================================
class TestWrapListField:
    def test_fires_on_dict(self):
        assert wrap_list_field({"text": "EU"}) == ([{"text": "EU"}], True)

    def test_silent_on_list(self):
        assert wrap_list_field([{"text": "EU"}]) == ([{"text": "EU"}], False)

    def test_silent_on_none(self):
        assert wrap_list_field(None) == (None, False)

    def test_silent_on_string(self):
        assert wrap_list_field("EU") == ("EU", False)


class TestNormalizeEnum:
    def test_fires_on_lowercase(self):
        assert normalize_enum("org", LABELS) == ("ORG", True)

    def test_fires_on_padded_value(self):
        assert normalize_enum(" misc ", LABELS) == ("MISC", True)

    def test_silent_on_valid_value(self):
        assert normalize_enum("ORG", LABELS) == ("ORG", False)

    def test_leaves_synonyms_alone(self):
        # Mapping "organization" -> "ORG" is a judgment about meaning, not a repair.
        assert normalize_enum("organization", LABELS) == ("organization", False)

    def test_silent_on_none(self):
        assert normalize_enum(None, LABELS) == (None, False)

    def test_silent_on_number(self):
        assert normalize_enum(3, LABELS) == (3, False)


class TestNullify:
    @pytest.mark.parametrize("value", ["", "N/A", "n/a", "NA", "none", "NULL", "unknown", "  null  "])
    def test_fires_on_null_like(self, value):
        assert nullify(value) == (None, True)

    def test_silent_on_real_value(self):
        assert nullify("Brussels") == ("Brussels", False)

    def test_silent_on_none(self):
        assert nullify(None) == (None, False)

    def test_silent_on_non_string(self):
        assert nullify(0) == (0, False)


# ===========================================================================
# 4. The walker
# ===========================================================================
class TestWalk:
    def test_fixes_nested_list_items_in_place(self):
        data = {"entities": [{"text": "EU", "type": "org"}]}
        fired = []
        result = walk(data, Extraction, fired)
        assert result is None                      # walk returns nothing
        assert data["entities"][0]["type"] == "ORG"
        assert fired == [ENUM]

    def test_records_every_occurrence(self):
        data = {"entities": [{"text": "EU", "type": "org"}, {"text": "German", "type": "misc"}]}
        fired = []
        walk(data, Extraction, fired)
        assert fired.count(ENUM) == 2

    def test_wraps_object_where_list_expected(self):
        data = {"entities": {"text": "EU", "type": "ORG"}}
        fired = []
        walk(data, Extraction, fired)
        assert data["entities"] == [{"text": "EU", "type": "ORG"}]
        assert fired == [WRAP]

    def test_wrapped_item_is_also_repaired(self):
        data = {"entities": {"text": "EU", "type": "org"}}
        fired = []
        walk(data, Extraction, fired)
        assert data["entities"] == [{"text": "EU", "type": "ORG"}]
        assert fired == [WRAP, ENUM]

    def test_ignores_keys_not_in_schema(self):
        data = {"entities": [{"text": "EU", "type": "ORG", "confidence": "high"}]}
        fired = []
        walk(data, Extraction, fired)
        assert data["entities"][0]["confidence"] == "high"
        assert fired == []

    def test_nullify_skips_required_fields(self):
        # "NA" can be a real entity. text is not nullable, so it must survive.
        data = {"entities": [{"text": "NA", "type": "ORG"}]}
        fired = []
        walk(data, Extraction, fired)
        assert data["entities"][0]["text"] == "NA"
        assert NULL not in fired

    def test_nullify_on_nullable_field(self):
        data = {"actor": "EU", "action": "rejects", "date": "N/A"}
        fired = []
        walk(data, Event, fired)
        assert data["date"] is None
        assert fired == [NULL]

    def test_recurses_into_single_nested_model(self):
        data = {"actor": "EU", "action": "rejects", "location": {"city": "n/a", "country": "Belgium"}}
        fired = []
        walk(data, Event, fired)
        assert data["location"] == {"city": None, "country": "Belgium"}
        assert fired == [NULL]

    def test_null_like_string_for_nested_model(self):
        data = {"actor": "EU", "action": "rejects", "location": "unknown"}
        fired = []
        walk(data, Event, fired)
        assert data["location"] is None
        assert fired == [NULL]

    def test_literal_or_none_field_normalizes(self):
        data = {"actor": "EU", "action": "rejects", "category": "Trade"}
        fired = []
        walk(data, Event, fired)
        assert data["category"] == "trade"
        assert fired == [ENUM]

    def test_literal_or_none_field_nullifies_first(self):
        data = {"actor": "EU", "action": "rejects", "category": "n/a"}
        fired = []
        walk(data, Event, fired)
        assert data["category"] is None
        assert fired == [NULL]

    def test_follows_schema_field_order(self):
        data = {"category": "Trade", "actor": "EU", "action": "rejects", "date": "N/A"}
        fired = []
        walk(data, Event, fired)
        assert fired == [NULL, ENUM]              # date comes before category in Event

    @pytest.mark.parametrize("bad", [
        {"entities": "EU"},
        {"entities": 3},
        {"entities": ["EU", {"text": "EU", "type": "org"}]},
        {"entities": [[{"text": "EU", "type": "ORG"}]]},
        {"entities": None},
    ])
    def test_wrong_shapes_never_crash(self, bad):
        walk(bad, Extraction, [])

    def test_wrong_shape_items_are_skipped_but_dicts_still_fixed(self):
        data = {"entities": ["EU", {"text": "EU", "type": "org"}]}
        fired = []
        walk(data, Extraction, fired)
        assert data["entities"][1]["type"] == "ORG"


# ===========================================================================
# 5. Stage 2: validate_with_repairs
# ===========================================================================
class TestValidateWithRepairs:
    def test_clean_data_passes_with_no_rules(self):
        r = validate_with_repairs({"entities": [{"text": "EU", "type": "ORG"}]}, Extraction)
        assert r.ok
        assert isinstance(r.value, Extraction)
        assert r.rules_fired == []
        assert r.errors == []

    def test_does_not_modify_its_input(self):
        data = {"entities": [{"text": "EU", "type": "org"}]}
        validate_with_repairs(data, Extraction)
        assert data["entities"][0]["type"] == "org"

    def test_repairs_then_validates(self):
        r = validate_with_repairs({"entities": [{"text": "EU", "type": "org"}]}, Extraction)
        assert r.ok
        assert r.value.entities[0].type == "ORG"
        assert r.rules_fired == [ENUM]

    def test_wraps_bare_list_when_one_list_field(self):
        r = validate_with_repairs([{"text": "EU", "type": "ORG"}], Extraction)
        assert r.ok
        assert r.value.entities[0].text == "EU"
        assert r.rules_fired == [WRAP]

    def test_bare_list_is_also_walked(self):
        r = validate_with_repairs([{"text": "EU", "type": "org"}], Extraction)
        assert r.ok
        assert r.rules_fired == [WRAP, ENUM]

    def test_does_not_guess_with_two_list_fields(self):
        r = validate_with_repairs([{"text": "EU", "type": "ORG"}], TwoLists)
        assert not r.ok
        assert WRAP not in r.rules_fired

    def test_failure_returns_structured_errors(self):
        data = {"entities": [{"text": "EU", "type": "ORG"},
                             {"text": "British", "type": "MISC", "confidence": 0.9}]}
        r = validate_with_repairs(data, Extraction)
        assert not r.ok
        assert r.value is None
        assert len(r.errors) == 1
        assert tuple(r.errors[0]["loc"]) == ("entities", 1, "confidence")

    def test_errors_can_be_saved_as_json(self):
        r = validate_with_repairs({"entities": [{"text": "EU", "type": "organization"}]}, Extraction)
        json.dumps(r.errors)                       # must not raise

    def test_errors_have_no_url(self):
        r = validate_with_repairs({"entities": [{"type": "ORG"}]}, Extraction)
        assert all("url" not in e for e in r.errors)

    def test_repairs_happen_at_every_depth_before_one_validation(self):
        data = {"actor": "EU", "action": "rejects", "date": "",
                "location": {"city": "unknown", "country": "Belgium"}, "category": "Politics"}
        r = validate_with_repairs(data, Event)
        assert r.ok
        assert r.value.date is None
        assert r.value.location.city is None
        assert r.value.category == "politics"
        assert r.rules_fired == [NULL, NULL, ENUM]


# ===========================================================================
# 6. describe_loc
# ===========================================================================
class TestDescribeLoc:
    def test_nested_path_is_one_indexed(self):
        assert describe_loc(("entities", 2, "confidence")) == '"entities" -> item 3 -> "confidence"'

    def test_first_item_is_item_1(self):
        assert describe_loc(("entities", 0, "type")) == '"entities" -> item 1 -> "type"'

    def test_single_key(self):
        assert describe_loc(("date",)) == '"date"'

    def test_empty_location(self):
        assert describe_loc(()) == "the whole object"


# ===========================================================================
# 7. run_pipeline (what D1 calls)
# ===========================================================================
Q1_OUTPUT = (
    'Sure! Here are the entities:\n```json\n'
    '{"entities": [{"text": "EU", "type": "org"}, {"text": "Germany", "type": "MISC"}, '
    '{"text": "British", "type": "MISC", "confidence": 0.9}],}\n```'
)


class TestRunPipeline:
    def test_success_has_no_message(self):
        r = run_pipeline('{"entities": [{"text": "EU", "type": "ORG"}]}', Extraction)
        assert r.ok
        assert isinstance(r.value, Extraction)
        assert r.failed_stage is None
        assert r.rules_fired == []
        assert r.model_message is None

    def test_repaired_success_has_no_message(self):
        r = run_pipeline('```json\n{"entities": [{"text": "EU", "type": "ORG"}]}\n```', Extraction)
        assert r.ok
        assert r.rules_fired == [STRIP]
        assert r.model_message is None

    def test_combines_rules_from_both_stages_in_order(self):
        r = run_pipeline(Q1_OUTPUT, Extraction)
        assert r.rules_fired == [SPAN, COMMA, ENUM]

    def test_knowledge_check_q1(self):
        r = run_pipeline(Q1_OUTPUT, Extraction)
        assert not r.ok
        assert r.value is None
        assert r.failed_stage == "validation"
        assert len(r.errors) == 1
        assert tuple(r.errors[0]["loc"]) == ("entities", 2, "confidence")

    def test_validation_message_content(self):
        msg = run_pipeline(Q1_OUTPUT, Extraction).model_message
        assert isinstance(msg, str)
        assert "item 3" in msg
        assert '"confidence"' in msg
        assert "you wrote 0.9" in msg
        assert "Return the corrected JSON only" in msg

    def test_validation_message_has_one_line_per_error(self):
        raw = '{"entities": [{"type": "org", "extra": 1}, {"text": "EU", "type": "organization"}]}'
        r = run_pipeline(raw, Extraction)
        problem_lines = [ln for ln in r.model_message.splitlines() if ln.startswith("- ")]
        assert len(problem_lines) == len(r.errors) == 3

    def test_long_inputs_are_shortened_in_message(self):
        raw = '{"entities": [{"type": "ORG", "note": "' + "x" * 500 + '"}]}'
        msg = run_pipeline(raw, Extraction).model_message
        assert all(len(line) < 250 for line in msg.splitlines())

    def test_parse_failure(self):
        r = run_pipeline('{"entities": "EU"', Extraction)
        assert not r.ok
        assert r.failed_stage == "parse"
        assert len(r.errors) == 1 and isinstance(r.errors[0], str)

    def test_parse_message_content(self):
        msg = run_pipeline('{"entities": "EU"', Extraction).model_message
        assert isinstance(msg, str)
        assert "line" in msg and "column" in msg
        assert "Return the corrected JSON only" in msg

    @pytest.mark.parametrize("reason", ["length", "max_tokens"])
    def test_truncation(self, reason):
        r = run_pipeline('{"entities": [{"text": "Ger', Extraction, done_reason=reason)
        assert not r.ok
        assert r.failed_stage == "truncated"
        assert r.rules_fired == []
        assert r.model_message is None

    def test_python_literal_through_pipeline(self):
        r = run_pipeline("{'entities': [{'text': \"EU's\", 'type': 'ORG'}]}", Extraction)
        assert r.ok
        assert r.value.entities[0].text == "EU's"
        assert r.rules_fired == [PYLIT]


# ===========================================================================
# 8. Malformed-output corpus (named assignment requirement)
#    (name, raw output, schema, done_reason, expect ok, expect stage, expect rules fired)
# ===========================================================================
CORPUS = [
    ("clean",                    '{"entities": [{"text": "EU", "type": "ORG"}]}', Extraction, None, True, None, []),
    ("empty_object",             '{}', Extraction, None, True, None, []),
    ("fenced_json",              '```json\n{"entities": [{"text": "EU", "type": "ORG"}]}\n```', Extraction, None, True, None, [STRIP]),
    ("fenced_no_language",       '```\n{"entities": [{"text": "EU", "type": "ORG"}]}\n```', Extraction, None, True, None, [STRIP]),
    ("preamble_and_postamble",   'Sure! {"entities": [{"text": "EU", "type": "ORG"}]} Let me know.', Extraction, None, True, None, [SPAN]),
    ("trailing_comma_list",      '{"entities": [{"text": "EU", "type": "ORG"},]}', Extraction, None, True, None, [COMMA]),
    ("trailing_comma_object",    '{"entities": [{"text": "EU", "type": "ORG",}]}', Extraction, None, True, None, [COMMA]),
    ("fence_and_trailing_comma", '```json\n{"entities": [{"text": "EU", "type": "ORG"},]}\n```', Extraction, None, True, None, [STRIP, COMMA]),
    ("comment_after_json",       '{"entities": []} // done', Extraction, None, True, None, [SPAN]),
    ("single_quotes",            "{'entities': [{'text': 'EU', 'type': 'ORG'}]}", Extraction, None, True, None, [PYLIT]),
    ("python_none",              "{'actor': 'EU', 'action': 'rejects', 'date': None}", Event, None, True, None, [PYLIT]),
    ("lowercase_label",          '{"entities": [{"text": "EU", "type": "org"}]}', Extraction, None, True, None, [ENUM]),
    ("bare_object_for_list",     '{"entities": {"text": "EU", "type": "ORG"}}', Extraction, None, True, None, [WRAP]),
    ("bare_list_no_wrapper",     '[{"text": "EU", "type": "ORG"}]', Extraction, None, True, None, [WRAP]),
    ("empty_string_for_none",    '{"actor": "EU", "action": "rejects", "date": ""}', Event, None, True, None, [NULL]),
    ("na_string_for_none",       '{"actor": "EU", "action": "rejects", "date": "N/A"}', Event, None, True, None, [NULL]),
    ("null_like_nested_object",  '{"actor": "EU", "action": "rejects", "location": "unknown"}', Event, None, True, None, [NULL]),
    ("null_like_inside_nested",  '{"actor": "EU", "action": "rejects", "location": {"city": "n/a"}}', Event, None, True, None, [NULL]),
    ("truncated_unflagged",      '{"entities": [{"text": "EU", "type": "ORG"}, {"text": "Ger', Extraction, None, False, "parse", [SPAN]),
    ("truncated_length",         '{"entities": [{"text": "EU", "type": "ORG"}, {"text": "Ger', Extraction, "length", False, "truncated", []),
    ("truncated_max_tokens",     '{"entities": [{"text": "EU", "type": "ORG"}, {"text": "Ger', Extraction, "max_tokens", False, "truncated", []),
    ("empty_output",             '', Extraction, None, False, "parse", []),
    ("whitespace_only",          '   \n  ', Extraction, None, False, "parse", []),
    ("not_json_at_all",          'I could not find any entities.', Extraction, None, False, "parse", []),
    ("comment_inside_json",      '{"entities": [] // none\n}', Extraction, None, False, "parse", []),
    ("wrong_type",               '{"entities": [{"text": "EU", "type": 3}]}', Extraction, None, False, "validation", []),
    ("unknown_label",            '{"entities": [{"text": "EU", "type": "organization"}]}', Extraction, None, False, "validation", []),
    ("missing_required",         '{"entities": [{"type": "ORG"}]}', Extraction, None, False, "validation", []),
    ("extra_field",              '{"entities": [{"text": "EU", "type": "ORG", "confidence": 0.9}]}', Extraction, None, False, "validation", []),
    ("nested_depth_error",       '{"entities": [[{"text": "EU", "type": "ORG"}]]}', Extraction, None, False, "validation", []),
    ("bare_entity_no_wrapper",   '{"text": "EU", "type": "ORG"}', Extraction, None, False, "validation", []),
    ("knowledge_check_q1",       Q1_OUTPUT, Extraction, None, False, "validation", [SPAN, COMMA, ENUM]),
]


@pytest.mark.parametrize(
    "name, raw, schema, done_reason, expect_ok, expect_stage, expect_fired",
    CORPUS,
    ids=[c[0] for c in CORPUS],
)
def test_malformed_corpus(name, raw, schema, done_reason, expect_ok, expect_stage, expect_fired):
    r = run_pipeline(raw, schema, done_reason=done_reason)
    assert r.ok is expect_ok
    assert r.failed_stage == expect_stage
    assert r.rules_fired == expect_fired
    if expect_ok:
        assert isinstance(r.value, schema)
        assert r.model_message is None
    else:
        assert r.value is None