import numpy as np
import pytest

from retrieval.metrics import (evaluate_run, evidence_at_budget, mean_scores, merge_spans, ndcg_at_k,
                               overlap_length, precision_at_k, recall_at_k, reciprocal_rank)

# Section 8A.6 worked example
RANK_A = ["d3", "d7", "d1", "d9", "d2"]
QRELS_A = {"d1": 2, "d2": 1, "d8": 1}
# Your Section 8A checkpoint example
RANK_B = ["d4", "d2", "d9", "d1", "d6"]
QRELS_B = {"d2": 1, "d6": 2, "d8": 1}


# ---------------------------------------------------------------- the two worked examples
def test_worked_example_a():
    assert recall_at_k(RANK_A, QRELS_A, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RANK_A, QRELS_A, 5) == pytest.approx(0.4)
    assert reciprocal_rank(RANK_A, QRELS_A, 5) == pytest.approx(1 / 3)
    assert ndcg_at_k(RANK_A, QRELS_A, 5, gain="linear") == pytest.approx(0.44295, abs=1e-4)
    assert ndcg_at_k(RANK_A, QRELS_A, 5, gain="exponential") == pytest.approx(0.45676, abs=1e-4)


def test_worked_example_b_your_checkpoint():
    assert recall_at_k(RANK_B, QRELS_B, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RANK_B, QRELS_B, 5) == pytest.approx(0.4)
    assert reciprocal_rank(RANK_B, QRELS_B, 5) == pytest.approx(0.5)
    assert ndcg_at_k(RANK_B, QRELS_B, 5, gain="linear") == pytest.approx(0.44863, abs=1e-4)
    assert ndcg_at_k(RANK_B, QRELS_B, 5, gain="exponential") == pytest.approx(0.43368, abs=1e-4)


def test_idcg_comes_from_qrels_not_the_ranking():
    # IDCG built from the retrieved docs only would give 0.534 here (your checkpoint)
    assert ndcg_at_k(RANK_B, QRELS_B, 5) != pytest.approx(0.534, abs=1e-3)


# ---------------------------------------------------------------- properties
def test_perfect_ranking_scores_one():
    perfect = ["d6", "d2", "d8"]
    for gain in ["linear", "exponential"]:
        assert ndcg_at_k(perfect, QRELS_B, 5, gain=gain) == pytest.approx(1.0)
    assert recall_at_k(perfect, QRELS_B, 5) == 1.0
    assert reciprocal_rank(perfect, QRELS_B, 5) == 1.0


def test_no_relevant_retrieved_scores_zero():
    ranked = ["x", "y", "z"]
    assert recall_at_k(ranked, QRELS_B, 5) == 0.0
    assert precision_at_k(ranked, QRELS_B, 5) == 0.0
    assert reciprocal_rank(ranked, QRELS_B, 5) == 0.0
    assert ndcg_at_k(ranked, QRELS_B, 5) == 0.0


def test_recall_never_decreases_with_k():
    ranked = ["d4", "d2", "x", "d8", "y", "d6", "z"]
    vals = [recall_at_k(ranked, QRELS_B, k) for k in range(1, 8)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    assert vals[-1] == 1.0


def test_ndcg_always_between_zero_and_one():
    rng = np.random.default_rng(0)
    qrels = {f"d{i}": int(rng.integers(0, 3)) for i in range(20)}
    qrels["d0"] = 2                                         # make sure |R| > 0
    pool = list(qrels) + [f"n{i}" for i in range(20)]
    for _ in range(200):
        ranked = list(rng.permutation(pool))
        for gain in ["linear", "exponential"]:
            assert 0.0 <= ndcg_at_k(ranked, qrels, 10, gain=gain) <= 1.0 + 1e-12


def test_rel_zero_rows_are_not_relevant():
    qrels = {"a": 1, "b": 0}
    ranked = ["b", "a"]
    assert recall_at_k(ranked, qrels, 2) == 1.0               # |R| = 1 (only "a")
    assert precision_at_k(ranked, qrels, 2) == 0.5
    assert reciprocal_rank(ranked, qrels, 2) == 0.5           # "b" at position 1 is not a hit
    assert ndcg_at_k(ranked, qrels, 2) == pytest.approx(1 / np.log2(3))


def test_fewer_results_than_k():
    assert precision_at_k(["d2"], QRELS_B, 5) == pytest.approx(0.2)   # divides by k, not by 1
    assert recall_at_k(["d2"], QRELS_B, 5) == pytest.approx(1 / 3)


def test_k_larger_than_relevant_set():
    assert ndcg_at_k(["a"], {"a": 1}, 10) == pytest.approx(1.0)


@pytest.mark.parametrize("qrels", [{}, {"a": 0, "b": 0}])
def test_no_relevant_documents_raises(qrels):
    with pytest.raises(ValueError):
        recall_at_k(["a"], qrels, 5)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], qrels, 5)


def test_unknown_gain_raises():
    with pytest.raises(ValueError):
        ndcg_at_k(RANK_A, QRELS_A, 5, gain="quadratic")


# ---------------------------------------------------------------- evaluate_run / mean_scores
def test_evaluate_run_loops_over_qrels():
    run = {"q1": RANK_A, "q3": ["anything"]}                 # q3 has no labels → ignored
    qrels = {"q1": QRELS_A, "q2": QRELS_B}                   # q2 missing from run → scores 0
    out = evaluate_run(run, qrels, 5)
    assert set(out) == {"q1", "q2"}
    assert set(out["q1"]) == {"recall", "precision", "rr", "ndcg"}
    assert out["q1"]["ndcg"] == pytest.approx(0.44295, abs=1e-4)
    assert out["q2"] == {"recall": 0.0, "precision": 0.0, "rr": 0.0, "ndcg": 0.0}


def test_mean_scores():
    per_query = {"q1": {"recall": 1.0, "precision": 0.2, "rr": 1.0, "ndcg": 1.0},
                 "q2": {"recall": 0.0, "precision": 0.0, "rr": 0.0, "ndcg": 0.0}}
    m = mean_scores(per_query)
    assert m == {"n": 2, "recall": 0.5, "precision": 0.1, "rr": 0.5, "ndcg": 0.5}
    assert mean_scores({}) == {"n": 0}


# ---------------------------------------------------------------- merge_spans / overlap_length
@pytest.mark.parametrize("spans,expected", [
    ([(0, 100), (80, 180), (200, 260)], [(0, 180), (200, 260)]),
    ([(0, 10), (10, 20)], [(0, 20)]),                         # touching → merged
    ([(0, 100), (20, 30)], [(0, 100)]),                       # contained: the end must not shrink
    ([(200, 260), (0, 100), (80, 180)], [(0, 180), (200, 260)]),   # unsorted input
    ([], []),
])
def test_merge_spans(spans, expected):
    assert merge_spans(spans) == expected


def test_merge_spans_does_not_modify_input():
    spans = [(200, 260), (0, 100)]
    merge_spans(spans)
    assert spans == [(200, 260), (0, 100)]


@pytest.mark.parametrize("a,b,expected", [
    ([(900, 1150), (2000, 2400)], [(1000, 1300)], 150),       # Section 8A.7
    ([(0, 10)], [(20, 30)], 0),
    ([(0, 50)], [(0, 50)], 50),
    ([(0, 100)], [(20, 30)], 10),
    ([(0, 10), (20, 30)], [(5, 25)], 10),
    ([], [(0, 10)], 0),
])
def test_overlap_length(a, b, expected):
    assert overlap_length(a, b) == expected


# ---------------------------------------------------------------- evidence_at_budget (tuples: start, end, n_tokens)
def test_evidence_worked_example():
    out = evidence_at_budget([(900, 1150, 60), (2000, 2400, 90)], [(1000, 1300)], budget=200)
    assert out["evidence_recall"] == pytest.approx(0.5)
    assert out["context_precision"] == pytest.approx(150 / 650)
    assert out["n_selected"] == 2 and out["tokens_used"] == 150


def test_evidence_stops_at_first_chunk_that_does_not_fit():
    ranked = [(0, 100, 50), (100, 300, 200), (300, 350, 10)]  # the 3rd fits, but selection already stopped
    out = evidence_at_budget(ranked, [(300, 350)], budget=100)
    assert out["n_selected"] == 1 and out["tokens_used"] == 50
    assert out["evidence_recall"] == 0.0
    assert out["context_precision"] == 0.0


def test_evidence_exactly_at_budget_is_taken():
    out = evidence_at_budget([(0, 10, 100)], [(0, 10)], budget=100)
    assert out["n_selected"] == 1 and out["evidence_recall"] == 1.0


def test_evidence_nothing_fits():
    out = evidence_at_budget([(0, 100, 300)], [(0, 50)], budget=200)
    assert out["n_selected"] == 0 and out["tokens_used"] == 0
    assert out["evidence_recall"] == 0.0
    assert out["context_precision"] is None


def test_evidence_overlapping_chunks_not_double_counted():
    out = evidence_at_budget([(0, 100, 20), (50, 150, 20)], [(0, 150)], budget=100)
    assert out["evidence_recall"] == pytest.approx(1.0)
    assert out["context_precision"] == pytest.approx(1.0)        # merged denominator: 150, not 200


def test_evidence_duplicate_gold_spans():
    out = evidence_at_budget([(0, 100, 10)], [(0, 100), (0, 100)], budget=50)
    assert out["evidence_recall"] == pytest.approx(1.0)


def test_evidence_empty_gold_raises():
    with pytest.raises(ValueError):
        evidence_at_budget([(0, 100, 10)], [], budget=50)