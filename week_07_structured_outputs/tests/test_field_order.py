"""Tests for Deliverable 5 scoring functions (field-order experiment).

Run from week07/:   python -m pytest tests/test_field_order.py -v

Values are passed as plain dicts (what model_dump() gives), and raw model text
as strings, exactly like the run script will pass them.
"""
import pytest

from extraction.evaluation import (
    followed_order,
    key_order,
    noise_margin,
    paired_disagreements,
    quote_grounded,
    topic_correct,
)

EVIDENCE_FIRST = ["evidence", "topic"]
LABEL_FIRST = ["topic", "evidence"]

ARTICLE = ("Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, "
           "Wall Street's dwindling band of ultra-cynics, are seeing green again.")

RAW_EVIDENCE_FIRST = '{"evidence": "Wall Street\'s dwindling band", "topic": "Business"}'
RAW_LABEL_FIRST = '{"topic": "Business", "evidence": "Wall Street\'s dwindling band"}'


# ===========================================================================
# 1. key_order: read from the RAW text, in the order the model wrote it
# ===========================================================================
class TestKeyOrder:
    def test_evidence_first(self):
        assert key_order(RAW_EVIDENCE_FIRST) == EVIDENCE_FIRST

    def test_label_first(self):
        assert key_order(RAW_LABEL_FIRST) == LABEL_FIRST

    def test_order_survives_fence_repair(self):
        assert key_order("```json\n" + RAW_LABEL_FIRST + "\n```") == LABEL_FIRST

    def test_order_survives_python_literal_fallback(self):
        assert key_order("{'topic': 'Sports', 'evidence': 'won the cup'}") == LABEL_FIRST

    def test_extra_keys_are_listed(self):
        raw = '{"topic": "Sports", "confidence": 0.9, "evidence": "won"}'
        assert key_order(raw) == ["topic", "confidence", "evidence"]

    @pytest.mark.parametrize("raw", ["", "not json at all", '{"topic": "Sports", "evid'])
    def test_unparseable_is_none(self, raw):
        assert key_order(raw) is None

    def test_json_list_has_no_keys(self):
        assert key_order('[{"topic": "Sports"}]') is None


# ===========================================================================
# 2. followed_order
# ===========================================================================
class TestFollowedOrder:
    def test_followed(self):
        assert followed_order(RAW_EVIDENCE_FIRST, EVIDENCE_FIRST) is True

    def test_reversed(self):
        assert followed_order(RAW_LABEL_FIRST, EVIDENCE_FIRST) is False

    def test_missing_key_is_not_followed(self):
        assert followed_order('{"topic": "Sports"}', LABEL_FIRST) is False

    def test_extra_key_is_not_followed(self):
        raw = '{"topic": "Sports", "evidence": "won", "confidence": 0.9}'
        assert followed_order(raw, LABEL_FIRST) is False

    def test_unparseable_is_none(self):
        assert followed_order("no json here", EVIDENCE_FIRST) is None


# ===========================================================================
# 3. topic_correct
# ===========================================================================
class TestTopicCorrect:
    def test_right(self):
        assert topic_correct({"evidence": "x", "topic": "Business"}, "Business") is True

    def test_wrong(self):
        assert topic_correct({"evidence": "x", "topic": "World"}, "Business") is False

    def test_failed_extraction_counts_as_wrong(self):
        assert topic_correct(None, "Business") is False

    def test_works_for_either_field_order(self):
        assert topic_correct({"topic": "Sci/Tech", "evidence": "x"}, "Sci/Tech") is True


# ===========================================================================
# 4. quote_grounded
# ===========================================================================
class TestQuoteGrounded:
    def test_exact_quote(self):
        assert quote_grounded({"evidence": "dwindling band of ultra-cynics", "topic": "Business"},
                              ARTICLE) is True

    def test_ignores_capitalization(self):
        assert quote_grounded({"evidence": "WALL STREET'S DWINDLING BAND", "topic": "Business"},
                              ARTICLE) is True

    def test_ignores_surrounding_spaces(self):
        assert quote_grounded({"evidence": "  seeing green again  ", "topic": "Business"},
                              ARTICLE) is True

    def test_invented_quote(self):
        assert quote_grounded({"evidence": "stocks rallied on Monday", "topic": "Business"},
                              ARTICLE) is False

    def test_reworded_quote(self):
        assert quote_grounded({"evidence": "Wall Street's shrinking band", "topic": "Business"},
                              ARTICLE) is False

    @pytest.mark.parametrize("quote", ["", "   "])
    def test_empty_quote_is_not_grounded(self, quote):
        # "" in any_text is True in Python. An empty quote must still count as not grounded.
        assert quote_grounded({"evidence": quote, "topic": "Business"}, ARTICLE) is False

    def test_failed_extraction(self):
        assert quote_grounded(None, ARTICLE) is False


# ===========================================================================
# 5. paired_disagreements
# ===========================================================================
EVIDENCE_ARM = {"ag_0001": True, "ag_0002": True, "ag_0003": False, "ag_0004": True}
LABEL_ARM = {"ag_0001": True, "ag_0002": False, "ag_0003": False, "ag_0004": False}


class TestPairedDisagreements:
    def test_lesson_example(self):
        assert paired_disagreements(EVIDENCE_ARM, LABEL_ARM) == (2, 0)

    def test_order_of_arguments_matters(self):
        assert paired_disagreements(LABEL_ARM, EVIDENCE_ARM) == (0, 2)

    def test_counts_both_directions(self):
        a = {"x": True, "y": False, "z": True}
        b = {"x": False, "y": True, "z": True}
        assert paired_disagreements(a, b) == (1, 1)

    def test_all_agree(self):
        assert paired_disagreements(EVIDENCE_ARM, dict(EVIDENCE_ARM)) == (0, 0)

    def test_only_shared_ids_count(self):
        a = dict(EVIDENCE_ARM, ag_9999=True)          # extra article only in arm A
        b = dict(LABEL_ARM, ag_8888=True)             # extra article only in arm B
        assert paired_disagreements(a, b) == (2, 0)

    def test_no_shared_ids_raises(self):
        # Your design: arms with no articles in common means the run was set up wrong. Fail loudly.
        with pytest.raises(ValueError):
            paired_disagreements({"a": True}, {"b": False})


# ===========================================================================
# 6. noise_margin
# ===========================================================================
class TestNoiseMargin:
    def test_spec_example(self):
        assert noise_margin(0.8, 100) == pytest.approx(0.08)

    def test_fifty_fifty(self):
        assert noise_margin(0.5, 50) == pytest.approx(0.1414, abs=1e-4)

    def test_bigger_sample_is_tighter(self):
        assert noise_margin(0.8, 400) == pytest.approx(0.04)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_certain_rates_have_no_margin(self, p):
        assert noise_margin(p, 100) == pytest.approx(0.0)