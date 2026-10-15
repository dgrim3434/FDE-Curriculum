import json
from statistics import NormalDist

import numpy as np
import pytest

from retrieval.bootstrap import (align_scores, bonferroni_level, cluster_bootstrap, interpret,
                                 min_detectable_effect, paired_bootstrap)


# ---------------------------------------------------------------- align_scores
def test_align_pairs_by_id_not_by_position():
    a = {"q2": {"ndcg": 0.2}, "q1": {"ndcg": 0.1}, "q3": {"ndcg": 0.3}}
    b = {"q3": {"ndcg": 0.9}, "q1": {"ndcg": 0.7}, "q2": {"ndcg": 0.8}}
    qids, va, vb = align_scores(a, b, "ndcg")
    assert qids == ["q1", "q2", "q3"]
    assert list(va) == [0.1, 0.2, 0.3] and list(vb) == [0.7, 0.8, 0.9]


def test_align_raises_on_different_query_sets():
    with pytest.raises(ValueError):
        align_scores({"q1": {"ndcg": 0.1}}, {"q2": {"ndcg": 0.1}}, "ndcg")


# ---------------------------------------------------------------- paired_bootstrap
def test_identical_runs_give_zero_ci_and_all_ties():
    a = np.random.default_rng(0).random(50)
    r = paired_bootstrap(a, a.copy())
    assert r["mean_diff"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert (r["wins"], r["ties"], r["losses"]) == (0, 50, 0)


def test_constant_shift_gives_exact_ci():
    a = np.random.default_rng(1).random(50)
    r = paired_bootstrap(a, a + 0.1)
    assert r["mean_diff"] == pytest.approx(0.1)
    assert r["ci_low"] == pytest.approx(0.1) and r["ci_high"] == pytest.approx(0.1)


def test_wins_ties_losses():
    r = paired_bootstrap(np.array([0.0, 0.5, 0.5, 1.0]), np.array([0.2, 0.5, 0.4, 1.0]))
    assert (r["wins"], r["ties"], r["losses"]) == (1, 2, 1)


def test_same_seed_same_result_and_json_serializable():
    rng = np.random.default_rng(2)
    a, b = rng.random(80), rng.random(80)
    r1 = paired_bootstrap(a, b, seed=7)
    assert r1 == paired_bootstrap(a, b, seed=7)
    assert (r1["n"], r1["n_boot"], r1["seed"], r1["level"]) == (80, 10_000, 7, 0.95)
    json.dumps(r1)


def test_higher_level_gives_wider_ci():
    rng = np.random.default_rng(3)
    a, b = rng.random(100), rng.random(100)
    r95 = paired_bootstrap(a, b, level=0.95)
    r99 = paired_bootstrap(a, b, level=bonferroni_level(4))
    assert (r99["ci_high"] - r99["ci_low"]) > (r95["ci_high"] - r95["ci_low"])


def test_calibration_coverage_is_about_95_percent():
    rng = np.random.default_rng(42)
    true_diff, hits, sims = 0.05, 0, 200
    for _ in range(sims):
        a = rng.random(100)
        b = a + rng.normal(true_diff, 0.2, size=100)
        r = paired_bootstrap(a, b, n_boot=2000, seed=int(rng.integers(1_000_000)))
        hits += r["ci_low"] <= true_diff <= r["ci_high"]
    assert 0.88 <= hits / sims <= 0.99


# ---------------------------------------------------------------- cluster_bootstrap
def test_cluster_constant_shift():
    a = np.random.default_rng(4).random(12)
    clusters = ["P1", "P2", "P1", "P3", "P2", "P3", "P1", "P2", "P3", "P1", "P2", "P3"]
    r = cluster_bootstrap(a, a + 0.1, clusters)
    assert r["ci_low"] == pytest.approx(0.1) and r["ci_high"] == pytest.approx(0.1)


def test_cluster_groups_non_contiguous_papers():
    a = np.zeros(5)
    b = np.array([0.1, 0.0, 0.1, -0.1, 0.0])
    clusters = ["P1", "P2", "P1", "P3", "P2"]               # P1 and P2 are NOT next to each other
    r = cluster_bootstrap(a, b, clusters)
    assert r["n"] == 5 and r["n_clusters"] == 3
    assert r["mean_diff"] == pytest.approx(0.02)             # mean over QUESTIONS: 0.1 / 5
    assert (r["wins"], r["ties"], r["losses"]) == (2, 2, 1)  # per QUESTION


def test_cluster_ci_is_wider_when_questions_within_a_paper_are_correlated():
    rng = np.random.default_rng(5)
    papers = [f"P{p}" for p in range(30)]
    effects = rng.normal(0.0, 0.3, size=30)
    clusters, d = [], []
    for p, e in zip(papers, effects):
        for _ in range(5):                                   # 5 questions per paper, all the same difference
            clusters.append(p)
            d.append(e)
    a = np.zeros(len(d))
    b = np.array(d)
    q = paired_bootstrap(a, b)
    c = cluster_bootstrap(a, b, clusters)
    assert c["mean_diff"] == pytest.approx(q["mean_diff"])
    assert (c["ci_high"] - c["ci_low"]) > 1.5 * (q["ci_high"] - q["ci_low"])


def test_cluster_same_seed_same_result():
    rng = np.random.default_rng(6)
    a, b = rng.random(20), rng.random(20)
    clusters = [f"P{i % 7}" for i in range(20)]
    assert cluster_bootstrap(a, b, clusters, seed=3) == cluster_bootstrap(a, b, clusters, seed=3)
    json.dumps(cluster_bootstrap(a, b, clusters, seed=3))


# ---------------------------------------------------------------- bonferroni / MDE / interpret
def test_bonferroni_level():
    assert bonferroni_level(4) == pytest.approx(0.9875)
    assert bonferroni_level(1) == pytest.approx(0.95)


def test_min_detectable_effect_formula():
    d = np.random.default_rng(8).normal(0, 0.12, size=300)
    expected = NormalDist().inv_cdf(0.975) * np.std(d, ddof=1) / np.sqrt(300)
    assert min_detectable_effect(d) == pytest.approx(expected)


@pytest.mark.parametrize("lo,hi,expected", [
    (0.006, 0.026, "b_better"),
    (-0.030, -0.010, "a_better"),
    (-0.004, 0.003, "tie"),
    (-0.060, 0.070, "inconclusive"),
    (0.0, 0.0, "tie"),                    # identical configs
    (-0.010, 0.010, "tie"),               # exactly on the margin counts as within it
    (0.0, 0.020, "inconclusive"),         # touches 0, so not "better"; extends past the margin
])
def test_interpret(lo, hi, expected):
    assert interpret(lo, hi, tie_margin=0.01) == expected