"""Tests for agreement-based confidence (Deliverable 2, confidence side).

Run from week07/:   python -m pytest tests/test_confidence.py -v

Uses YOUR FakeLLM / LLMResponse. FakeLLM.calls records the seed and temperature
of every call, so the tests can check that each sample really is different.
"""
import pytest

from extraction.confidence import SelfConsistancy, agreement, calibration_table, majority, sample
from extraction.evaluation import prf
from extraction.schemas import Extraction
from llm import FakeLLM, LLMResponse

SENTENCE = "The European Commission said on Thursday it disagreed with German advice ."
VALID = '{"entities": [{"text": "European Commission", "type": "ORG"}]}'
INVALID = '{"entities": [{"text": "EU", "type": "organization"}]}'   # never repairable

EC, GER_MISC, GER_LOC, BRIT, THU = (("European Commission", "ORG"), ("German", "MISC"),
                                    ("German", "LOC"), ("British", "MISC"), ("Thursday", "MISC"))


def resp(text, tin=10, tout=5):
    return LLMResponse(text=text, stop_reason="end_turn", input_tokens=tin, output_tokens=tout)


# ===========================================================================
# 1. sample
# ===========================================================================
class TestSample:
    def test_returns_self_consistency_result(self):
        r = sample(SENTENCE, Extraction, FakeLLM([resp(VALID)] * 5), n=5)
        assert isinstance(r, SelfConsistancy)

    def test_all_valid(self):
        r = sample(SENTENCE, Extraction, FakeLLM([resp(VALID)] * 5), n=5)
        assert r.valid == 5
        assert len(r.results) == 5
        assert all(isinstance(v, Extraction) for v in r.results)

    def test_makes_n_calls_when_all_valid(self):
        llm = FakeLLM([resp(VALID)] * 3)
        sample(SENTENCE, Extraction, llm, n=3)
        assert len(llm.calls) == 3

    def test_failed_sample_is_left_out(self):
        # sample 1 valid; sample 2 fails after 3 calls (first try + 2 retries); sample 3 valid
        llm = FakeLLM([resp(VALID), resp(INVALID), resp(INVALID), resp(INVALID), resp(VALID)])
        r = sample(SENTENCE, Extraction, llm, n=3)
        assert r.valid == 2
        assert len(r.results) == 2
        assert len(llm.calls) == 5

    def test_each_sample_gets_its_own_seed(self):
        llm = FakeLLM([resp(VALID)] * 5)
        sample(SENTENCE, Extraction, llm, n=5)
        assert [c["seed"] for c in llm.calls] == [42, 43, 44, 45, 46]

    def test_base_seed_is_used(self):
        llm = FakeLLM([resp(VALID)] * 3)
        sample(SENTENCE, Extraction, llm, n=3, base_seed=7)
        assert [c["seed"] for c in llm.calls] == [7, 8, 9]

    def test_retry_inside_a_sample_keeps_that_samples_seed(self):
        llm = FakeLLM([resp(INVALID), resp(VALID), resp(VALID)])
        sample(SENTENCE, Extraction, llm, n=2)
        assert [c["seed"] for c in llm.calls] == [42, 42, 43]

    def test_temperature_is_passed_through(self):
        llm = FakeLLM([resp(VALID)] * 2)
        sample(SENTENCE, Extraction, llm, n=2, temperature=0.7)
        assert all(c["temperature"] == 0.7 for c in llm.calls)

    def test_tokens_add_up_across_every_call(self):
        llm = FakeLLM([resp(INVALID, 100, 20), resp(VALID, 120, 25), resp(VALID, 90, 15)])
        r = sample(SENTENCE, Extraction, llm, n=2)
        assert (r.input_tokens, r.output_tokens) == (310, 60)


# ===========================================================================
# 2. agreement
# ===========================================================================
# Lesson 9 Part 2, checkpoint Q3: sample 2 failed (empty set), N = 5.
Q3_SETS = [
    {EC, GER_MISC, BRIT},
    set(),
    {EC, GER_LOC, BRIT},
    {EC, BRIT},
    {EC, GER_MISC, BRIT, THU},
]


class TestAgreement:
    def test_lesson_9_checkpoint(self):
        assert agreement(Q3_SETS, 5) == pytest.approx(
            {EC: 0.8, BRIT: 0.8, GER_MISC: 0.4, GER_LOC: 0.2, THU: 0.2})

    def test_divides_by_n_not_by_number_of_sets(self):
        # Only the 4 valid samples are passed in, but there were 5 samples.
        valid_only = [s for s in Q3_SETS if s]
        assert agreement(valid_only, 5)[EC] == pytest.approx(0.8)

    def test_one_valid_sample_of_five_is_not_full_confidence(self):
        assert agreement([{EC}], 5) == pytest.approx({EC: 0.2})

    def test_all_agree(self):
        assert agreement([{EC}] * 5, 5) == pytest.approx({EC: 1.0})

    def test_nothing_found(self):
        assert agreement([set(), set()], 5) == {}
        assert agreement([], 5) == {}

    def test_works_for_relation_triples(self):
        t = ("Peter Blackburn", "Work_For", "Reuters")
        assert agreement([{t}, {t}, set()], 3) == pytest.approx({t: 2 / 3})


# ===========================================================================
# 3. majority
# ===========================================================================
class TestMajority:
    def test_lesson_9_checkpoint(self):
        assert set(majority(agreement(Q3_SETS, 5))) == {EC, BRIT}

    def test_exactly_half_is_kept(self):
        assert set(majority({EC: 0.5, BRIT: 0.4})) == {EC}

    def test_custom_threshold(self):
        scores = agreement(Q3_SETS, 5)
        assert set(majority(scores, threshold=0.2)) == {EC, BRIT, GER_MISC, GER_LOC, THU}

    def test_empty(self):
        assert set(majority({})) == set()

    def test_result_can_be_scored_with_prf(self):
        # The majority vote IS the final prediction, so it has to work with prf.
        gold = {EC, GER_MISC, BRIT}
        pred = majority(agreement(Q3_SETS, 5))
        assert prf([gold], [pred]) == pytest.approx((1.0, 2 / 3, 0.8))


# ===========================================================================
# 4. calibration_table
# ===========================================================================
class TestCalibrationTable:
    PAIRS = [(0.8, True), (0.8, True), (0.8, True), (0.8, False),
             (0.2, False), (0.2, True),
             (1.0, True)]

    def test_one_row_per_confidence_level(self):
        assert set(calibration_table(self.PAIRS)) == {0.2, 0.8, 1.0}

    def test_each_row_has_n_and_accuracy(self):
        table = calibration_table(self.PAIRS)
        n, acc = table[0.8]
        assert n == 4
        assert acc == pytest.approx(0.75)

    def test_every_row(self):
        table = calibration_table(self.PAIRS)
        assert table[0.2][0] == 2 and table[0.2][1] == pytest.approx(0.5)
        assert table[1.0][0] == 1 and table[1.0][1] == pytest.approx(1.0)

    def test_n_adds_up_to_all_pairs(self):
        table = calibration_table(self.PAIRS)
        assert sum(n for n, _ in table.values()) == len(self.PAIRS)

    def test_all_wrong(self):
        n, acc = calibration_table([(0.4, False), (0.4, False)])[0.4]
        assert n == 2 and acc == pytest.approx(0.0)

    def test_empty(self):
        assert calibration_table([]) == {}

    def test_end_to_end_from_agreement(self):
        gold = {EC, GER_MISC, BRIT}
        scores = agreement(Q3_SETS, 5)
        pairs = [(conf, item in gold) for item, conf in scores.items()]
        table = calibration_table(pairs)
        assert table[0.8] == (2, pytest.approx(1.0))     # EC and BRIT, both right
        assert table[0.4] == (1, pytest.approx(1.0))     # German MISC, right
        assert table[0.2] == (2, pytest.approx(0.0))     # German LOC and Thursday, both wrong