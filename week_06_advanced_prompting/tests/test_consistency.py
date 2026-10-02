"""
AI Generated Test Cases For Self-Consistancy
"""

import math

import pytest

from prompting.consistency import self_consistency, SCResults
from prompting.llm import FakeLLM, LLMResponse
from prompting.parsing import make_number_parser

MSGS = [{"role": "user", "content": "What is 6 * 3?"}]


def R(text, stop="end_turn", i=10, o=5, t=0.5):
    return LLMResponse(text=text, stop_reason=stop, input_tokens=i, output_tokens=o, latency_s=t)


def A(v):
    return R(f"reasoning... <answer>{v}</answer>")


def run(replies, **kw):
    fake = FakeLLM(replies)
    kw.setdefault("n", len(replies))
    res = self_consistency(fake, "SYS", MSGS, make_number_parser(), **kw)
    return res, fake


# ─────────────────────────────── voting ────────────────────────────────

class TestVoting:
    def test_basic_majority(self):
        res, _ = run([A(18), A(18), A(24), A(18), A(17)])
        assert isinstance(res, SCResults)
        assert res.status == "ok"
        assert res.answer == pytest.approx(18.0)
        assert res.distribution == {18.0: 3, 24.0: 1, 17.0: 1}

    def test_distribution_is_a_dict(self):
        res, _ = run([A(18), A(24), A(18)])
        assert isinstance(res.distribution, dict)

    def test_normalization_before_counting(self):
        # raw strings would make "17" win; normalized numbers make 18 win
        res, _ = run([A("18.0"), A("$18"), A("18"), A("17"), A("17")])
        assert res.answer == pytest.approx(18.0)
        assert res.distribution == {18.0: 3, 17.0: 2}

    def test_tiny_float_differences_are_one_vote(self):
        res, _ = run([A("18"), A("18.0000001"), A("24")])
        assert res.distribution == {18.0: 2, 24.0: 1}

    def test_n_equals_one(self):
        res, fake = run([A(18)], n=1, min_valid=1)
        assert len(fake.calls) == 1
        assert res.status == "ok"
        assert res.answer == pytest.approx(18.0)
        assert res.vote_share_valid == pytest.approx(1.0)
        assert res.margin == pytest.approx(1.0)
        assert res.entropy == pytest.approx(0.0)


# ─────────────────────────────── confidence numbers ────────────────────

class TestConfidence:
    def test_vote_shares_are_single_numbers_for_the_winner(self):
        res, _ = run([A(18), A(18), A(24), A(18), A(17)])
        assert res.vote_share_valid == pytest.approx(3 / 5)
        assert res.vote_share_all == pytest.approx(3 / 5)

    def test_shares_with_invalid_samples(self):
        res, _ = run([A(18), R("no tag here"), A(18), R("still nothing"), A(18)])
        assert res.n_samples == 5
        assert res.n_valid == 3
        assert res.vote_share_valid == pytest.approx(1.0)
        assert res.vote_share_all == pytest.approx(3 / 5)

    def test_margin_with_runner_up(self):
        res, _ = run([A(18), A(18), A(18), A(24), A(24)])
        assert res.margin == pytest.approx((3 - 2) / 5)

    def test_margin_with_single_answer(self):
        res, _ = run([A(18), A(18), A(18)])
        assert res.margin == pytest.approx(1.0)

    def test_entropy_zero_when_all_agree(self):
        res, _ = run([A(18)] * 5)
        assert res.entropy == pytest.approx(0.0)

    def test_entropy_value_and_normalization(self):
        # votes [3, 2] over 5 valid samples, 5 drawn
        res, _ = run([A(18), A(18), A(18), A(24), A(24)])
        h = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
        assert res.entropy == pytest.approx(h / math.log(5))

    def test_entropy_max_when_all_different(self):
        res, _ = run([A(1), A(2), A(3), A(4), A(5)])
        assert res.entropy == pytest.approx(1.0)

    def test_entropy_is_never_negative(self):
        res, _ = run([A(18), A(24), A(18)])
        assert res.entropy >= 0


# ─────────────────────────────── ties & insufficient ───────────────────

class TestStatus:
    def test_tie_detected(self):
        res, _ = run([A(18), A(18), A(24), A(24), A(17)])
        assert res.is_tie is True
        assert res.status == "tie"
        assert res.answer in (18.0, 24.0)

    def test_no_tie_flag_when_clear_winner(self):
        res, _ = run([A(18), A(18), A(24)])
        assert res.is_tie is False

    def test_insufficient_samples(self):
        res, _ = run([A(18), R("x"), R("y"), R("z"), A(18)], min_valid=3)
        assert res.n_valid == 2
        assert res.status == "insufficient_samples"
        assert res.answer is None

    def test_all_invalid_does_not_crash(self):
        res, _ = run([R("nothing")] * 5)
        assert res.n_valid == 0
        assert res.status == "insufficient_samples"
        assert res.answer is None
        assert res.vote_share_valid is None
        assert res.vote_share_all == pytest.approx(0.0)
        assert res.distribution == {}


# ─────────────────────────────── sampling settings ─────────────────────

class TestSampling:
    def test_every_call_uses_the_temperature(self):
        _, fake = run([A(18)] * 5, temperature=0.7)
        assert [c["temperature"] for c in fake.calls] == [0.7] * 5

    def test_seeds_are_distinct_and_offset_by_base(self):
        _, fake = run([A(18)] * 5, base_seed=100)
        assert [c["seed"] for c in fake.calls] == [100, 101, 102, 103, 104]

    def test_prompt_passed_unchanged(self):
        _, fake = run([A(18)] * 3)
        for c in fake.calls:
            assert c["system"] == "SYS"
            assert c["messages"] == MSGS

    def test_exactly_n_calls_without_early_stop(self):
        _, fake = run([A(18)] * 5, early_stop=False)
        assert len(fake.calls) == 5

    def test_truncated_sample_is_invalid_even_with_a_tag(self):
        res, _ = run([R("<answer>99</answer>", stop="max_tokens"), A(18), A(18), A(18)])
        assert res.n_valid == 3
        assert 99.0 not in res.distribution
        assert res.answer == pytest.approx(18.0)


# ─────────────────────────────── early stopping ────────────────────────

class TestEarlyStop:
    def test_stops_once_decided(self):
        # after 3 agreeing samples: 3 - 0 > 2 remaining
        res, fake = run([A(18)] * 5, n=5, early_stop=True)
        assert len(fake.calls) == 3
        assert res.n_samples == 3
        assert res.answer == pytest.approx(18.0)

    def test_stops_at_four_when_one_disagrees(self):
        # [18, 24, 18, 18]: after 4 → 3 - 1 = 2 > 1 remaining
        res, fake = run([A(18), A(24), A(18), A(18), A(18)], n=5, early_stop=True)
        assert len(fake.calls) == 4

    def test_does_not_stop_when_still_open(self):
        _, fake = run([A(18), A(24), A(18), A(24), A(18)], n=5, early_stop=True)
        assert len(fake.calls) == 5

    def test_invalid_samples_count_as_drawn(self):
        # [18, bad, 18, 18]: after 4 drawn → 3 - 0 = 3 > 1 remaining
        _, fake = run([A(18), R("bad"), A(18), A(18), A(18)], n=5, early_stop=True)
        assert len(fake.calls) == 4


# ─────────────────────────────── trace & accounting ────────────────────

class TestAccounting:
    def test_raw_samples_kept_including_invalid(self):
        replies = [A(18), R("no tag"), A(24)]
        res, _ = run(replies)
        assert res.samples == [r.text for r in replies]

    def test_totals_summed(self):
        res, _ = run([R("<answer>1</answer>", i=10, o=3, t=0.5),
                      R("<answer>1</answer>", i=11, o=4, t=0.25),
                      R("<answer>1</answer>", i=12, o=5, t=0.25)])
        assert res.input_tokens == 33
        assert res.output_tokens == 12
        assert res.latency_s == pytest.approx(1.0)