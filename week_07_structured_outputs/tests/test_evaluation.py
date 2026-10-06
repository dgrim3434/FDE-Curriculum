"""Tests for the evaluation and grounding functions (Deliverable 2, scoring side).

Run from week07/:   python -m pytest tests/test_evaluation.py -v

Records here are shaped exactly like the ones run_chart.py saves:
    {"schema": ..., "outcome": ..., "rules_fired": [[...], ...],
     "value": {...} or None,
     "gold": {"sentence": ..., "entities": [[text, type], ...], "relations": [[s, r, o], ...]}}
Gold items are lists (that's what json.load gives back), not tuples.
"""
import json

import pytest

from extraction.evaluation import (
    flat_accuracy,
    gold_pairs,
    gold_triples,
    outcomes_counter,
    prf,
    rule_hits,
    to_pairs,
    to_triples,
)
from extraction.grounding import grounded_rate, ungrounded

SENTENCE = "Peter Blackburn reports for Reuters from Brussels ."

GRAPH_VALUE = {
    "entities": [{"text": "Peter Blackburn", "type": "Peop"},
                 {"text": "Reuters", "type": "Org"},
                 {"text": "Brussels", "type": "Loc"}],
    "relations": [{"subject": "Peter Blackburn", "relation": "Work_For", "object": "Reuters"}],
}


def record(value=None, schema="Extraction", outcome="first_pass", rules=None,
           sentence=SENTENCE, entities=None, relations=None):
    """Build a record the way run_chart.py saves it, then round-trip it through JSON."""
    r = {
        "schema": schema,
        "outcome": outcome,
        "rules_fired": rules if rules is not None else [[]],
        "value": value,
        "gold": {"sentence": sentence,
                 "entities": entities if entities is not None else [],
                 "relations": relations if relations is not None else []},
    }
    return json.loads(json.dumps(r))      # tuples become lists, exactly like the saved files


# ===========================================================================
# 1. Turning values into sets
# ===========================================================================
class TestToPairs:
    def test_none_is_empty(self):
        assert to_pairs(None) == set()

    def test_extraction_value(self):
        value = {"entities": [{"text": "EU", "type": "ORG"}, {"text": "German", "type": "MISC"}]}
        assert to_pairs(value) == {("EU", "ORG"), ("German", "MISC")}

    def test_graph_value_uses_only_entities(self):
        assert to_pairs(GRAPH_VALUE) == {("Peter Blackburn", "Peop"), ("Reuters", "Org"),
                                         ("Brussels", "Loc")}

    def test_no_entities(self):
        assert to_pairs({"entities": []}) == set()

    def test_duplicates_collapse(self):
        value = {"entities": [{"text": "EU", "type": "ORG"}, {"text": "EU", "type": "ORG"}]}
        assert to_pairs(value) == {("EU", "ORG")}


class TestToTriples:
    def test_none_is_empty(self):
        assert to_triples(None) == set()

    def test_graph_value(self):
        assert to_triples(GRAPH_VALUE) == {("Peter Blackburn", "Work_For", "Reuters")}

    def test_no_relations(self):
        assert to_triples({"entities": [], "relations": []}) == set()


class TestGold:
    def test_gold_pairs_are_tuples(self):
        r = record(entities=[["EU", "ORG"], ["German", "MISC"]])
        assert gold_pairs(r) == {("EU", "ORG"), ("German", "MISC")}

    def test_gold_pairs_empty(self):
        assert gold_pairs(record()) == set()

    def test_gold_triples_are_tuples(self):
        r = record(relations=[["Peter Blackburn", "Work_For", "Reuters"]])
        assert gold_triples(r) == {("Peter Blackburn", "Work_For", "Reuters")}

    def test_gold_and_prediction_compare_equal(self):
        r = record(value={"entities": [{"text": "EU", "type": "ORG"}]}, entities=[["EU", "ORG"]])
        assert gold_pairs(r) == to_pairs(r["value"])


# ===========================================================================
# 2. Precision / recall / F1
# ===========================================================================
class TestPRF:
    def test_perfect(self):
        assert prf([{"a", "b"}], [{"a", "b"}]) == pytest.approx((1.0, 1.0, 1.0))

    def test_lesson_9_worked_example(self):
        gold = {("European Commission", "ORG"), ("German", "MISC"), ("British", "MISC")}
        pred = {("European Commission", "ORG"), ("German", "LOC"), ("British", "MISC"),
                ("Thursday", "MISC")}
        assert prf([gold], [pred]) == pytest.approx((0.5, 2 / 3, 4 / 7))

    def test_lesson_9_checkpoint_example(self):
        gold = {("Reuters", "ORG"), ("Brussels", "LOC"), ("Peter Blackburn", "PER")}
        pred = {("Reuters", "ORG"), ("Brussels", "ORG"), ("Peter Blackburn", "PER"),
                ("Blackburn", "PER"), ("EU", "ORG")}
        assert prf([gold], [pred]) == pytest.approx((0.4, 2 / 3, 0.5))

    def test_counts_are_added_across_sentences_before_dividing(self):
        # sentence 1: 1 right of 1.  sentence 2: 0 right of 1 predicted, 3 gold.
        # summed: tp=1, predicted=2, gold=4 -> P=0.5, R=0.25 (an average per sentence would differ)
        golds = [{"a"}, {"b", "c", "d"}]
        preds = [{"a"}, {"x"}]
        assert prf(golds, preds) == pytest.approx((0.5, 0.25, 1 / 3))

    def test_no_predictions_does_not_crash(self):
        assert prf([{"a"}], [set()]) == pytest.approx((0.0, 0.0, 0.0))

    def test_nothing_anywhere_does_not_crash(self):
        assert prf([set()], [set()]) == pytest.approx((0.0, 0.0, 0.0))

    def test_empty_lists(self):
        assert prf([], []) == pytest.approx((0.0, 0.0, 0.0))

    def test_failed_extraction_counts_as_predicting_nothing(self):
        # End-to-end F1: a failed record's value is None -> empty set -> lowers recall only.
        golds = [{("EU", "ORG")}, {("UN", "ORG")}]
        preds = [to_pairs({"entities": [{"text": "EU", "type": "ORG"}]}), to_pairs(None)]
        assert prf(golds, preds) == pytest.approx((1.0, 0.5, 2 / 3))


# ===========================================================================
# 3. Counting outcomes and rule hits
# ===========================================================================
class TestOutcomesCounter:
    def test_counts_per_schema(self):
        records = [record(schema="Graph", outcome="retry"),
                   record(schema="Graph", outcome="retry"),
                   record(schema="Graph", outcome="failed"),
                   record(schema="Extraction", outcome="first_pass")]
        counts = outcomes_counter(records)
        assert counts["Graph"]["retry"] == 2
        assert counts["Graph"]["failed"] == 1
        assert counts["Extraction"]["first_pass"] == 1

    def test_missing_outcome_reads_as_zero(self):
        counts = outcomes_counter([record(schema="Event", outcome="first_pass")])
        assert counts["Event"]["failed"] == 0

    def test_empty(self):
        assert len(outcomes_counter([])) == 0


class TestRuleHits:
    def test_counts_rules(self):
        records = [record(rules=[["strip_fences"]]),
                   record(rules=[["strip_fences", "normalize_enum"]]),
                   record(rules=[[]])]
        hits = rule_hits(records)
        assert hits["strip_fences"] == 2
        assert hits["normalize_enum"] == 1

    def test_rule_firing_on_several_attempts_counts_once(self):
        hits = rule_hits([record(rules=[["strip_fences"], ["strip_fences"], ["strip_fences"]])])
        assert hits["strip_fences"] == 1

    def test_repeat_within_one_attempt_counts_once(self):
        hits = rule_hits([record(rules=[["normalize_enum", "normalize_enum"]])])
        assert hits["normalize_enum"] == 1

    def test_rules_from_different_attempts_all_count(self):
        hits = rule_hits([record(rules=[["strip_fences"], ["nullify"]])])
        assert hits["strip_fences"] == 1 and hits["nullify"] == 1

    def test_no_rules(self):
        assert sum(rule_hits([record(rules=[[]]), record(rules=[[], []])]).values()) == 0


# ===========================================================================
# 4. Flat schema accuracy
# ===========================================================================
FLAT_ALL_FALSE = {"has_person": False, "has_organization": False,
                  "has_location": False, "has_misc": False}


class TestFlatAccuracy:
    def test_all_correct(self):
        value = {"has_person": False, "has_organization": True, "has_location": False, "has_misc": True}
        r = record(value=value, schema="EntityTypes", entities=[["EU", "ORG"], ["German", "MISC"]])
        assert flat_accuracy([r]) == pytest.approx(
            {"has_person": 1.0, "has_organization": 1.0, "has_location": 1.0, "has_misc": 1.0})

    def test_per_field_scores(self):
        # gold has ORG only. Record 1 gets everything right; record 2 misses ORG and invents a PER.
        right = dict(FLAT_ALL_FALSE, has_organization=True)
        wrong = dict(FLAT_ALL_FALSE, has_person=True)
        records = [record(value=right, schema="EntityTypes", entities=[["EU", "ORG"]]),
                   record(value=wrong, schema="EntityTypes", entities=[["EU", "ORG"]])]
        assert flat_accuracy(records) == pytest.approx(
            {"has_person": 0.5, "has_organization": 0.5, "has_location": 1.0, "has_misc": 1.0})

    def test_failed_records_are_skipped(self):
        good = record(value=dict(FLAT_ALL_FALSE, has_organization=True), schema="EntityTypes",
                      entities=[["EU", "ORG"]])
        failed = record(value=None, schema="EntityTypes", outcome="failed", entities=[["EU", "ORG"]])
        assert flat_accuracy([good, failed])["has_organization"] == pytest.approx(1.0)

    def test_no_entities_means_all_false_is_correct(self):
        r = record(value=FLAT_ALL_FALSE, schema="EntityTypes", entities=[])
        assert flat_accuracy([r]) == pytest.approx(
            {"has_person": 1.0, "has_organization": 1.0, "has_location": 1.0, "has_misc": 1.0})

    def test_no_valid_records(self):
        assert flat_accuracy([record(value=None, schema="EntityTypes", outcome="failed")]) == {}

    def test_returns_all_four_fields(self):
        r = record(value=FLAT_ALL_FALSE, schema="EntityTypes")
        assert set(flat_accuracy([r])) == {"has_person", "has_organization", "has_location", "has_misc"}


# ===========================================================================
# 5. Grounding
# ===========================================================================
class TestUngrounded:
    def test_none_is_empty(self):
        assert ungrounded(None, SENTENCE) == []

    def test_all_copied(self):
        assert ungrounded(GRAPH_VALUE, SENTENCE) == []

    def test_invented_entity(self):
        value = {"entities": [{"text": "Reuters", "type": "ORG"}, {"text": "EU", "type": "ORG"}]}
        assert ungrounded(value, SENTENCE) == ["EU"]

    def test_changed_words(self):
        value = {"entities": [{"text": "Britain", "type": "MISC"}]}
        assert ungrounded(value, "EU rejects German call to boycott British lamb .") == ["Britain"]

    def test_partial_span_counts_as_grounded(self):
        # Lesson 9: grounding cannot catch partial spans. That is expected behavior.
        value = {"entities": [{"text": "Blackburn", "type": "PER"}]}
        assert ungrounded(value, SENTENCE) == []

    def test_no_entities(self):
        assert ungrounded({"entities": []}, SENTENCE) == []


class TestGroundedRate:
    def test_all_grounded(self):
        assert grounded_rate([record(value=GRAPH_VALUE, schema="Graph")]) == pytest.approx(1.0)

    def test_one_of_four_invented(self):
        value = {"entities": GRAPH_VALUE["entities"] + [{"text": "EU", "type": "Org"}]}
        assert grounded_rate([record(value=value)]) == pytest.approx(0.75)

    def test_counts_across_records(self):
        r1 = record(value={"entities": [{"text": "Reuters", "type": "ORG"}]})
        r2 = record(value={"entities": [{"text": "EU", "type": "ORG"}]})      # not in SENTENCE
        assert grounded_rate([r1, r2]) == pytest.approx(0.5)

    def test_uses_each_records_own_sentence(self):
        r1 = record(value={"entities": [{"text": "EU", "type": "ORG"}]}, sentence="EU rejects it .")
        r2 = record(value={"entities": [{"text": "Reuters", "type": "ORG"}]}, sentence=SENTENCE)
        assert grounded_rate([r1, r2]) == pytest.approx(1.0)

    def test_failed_records_are_skipped(self):
        records = [record(value=GRAPH_VALUE), record(value=None, outcome="failed")]
        assert grounded_rate(records) == pytest.approx(1.0)

    def test_no_entities_returns_none(self):
        assert grounded_rate([record(value={"entities": []})]) is None
        assert grounded_rate([]) is None