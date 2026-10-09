"""Tests for search/metrics.py — Week 8.
Deliverable 3: index_recall.   (Deliverable 4 adds retrieval metrics below.)

Run from week_08_vector_search/:
    python -m pytest tests/test_metrics.py -v
"""
import numpy as np
import pytest

from search.metrics import index_recall, recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k, evaluate


class TestIndexRecall:
    def test_concept5_hand_example(self):
        # brute force [7, 3, 12, 9, 40] vs IVF [7, 3, 12, 55, 40]: missed doc 9 → 4/5
        exact = np.array([[7, 3, 12, 9, 40]])
        approx = np.array([[7, 3, 12, 55, 40]])
        mean, per = index_recall(approx, exact, 5)
        assert mean == pytest.approx(0.8)
        assert np.allclose(per, [0.8])

    def test_identical_is_one(self):
        ids = np.array([[1, 2, 3], [4, 5, 6]])
        mean, per = index_recall(ids, ids.copy(), 3)
        assert mean == pytest.approx(1.0)
        assert np.allclose(per, [1.0, 1.0])

    def test_disjoint_is_zero(self):
        mean, per = index_recall(np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), 3)
        assert mean == pytest.approx(0.0)
        assert np.allclose(per, [0.0])

    def test_order_does_not_matter(self):
        exact = np.array([[10, 20, 30, 40]])
        approx = np.array([[40, 30, 20, 10]])
        mean, _ = index_recall(approx, exact, 4)
        assert mean == pytest.approx(1.0)

    def test_k_slices_both_arrays(self):
        exact = np.array([[1, 2, 3, 4]])
        approx = np.array([[2, 1, 9, 8]])
        assert index_recall(approx, exact, 1)[0] == pytest.approx(0.0)   # {2} vs {1}
        assert index_recall(approx, exact, 2)[0] == pytest.approx(1.0)   # {2,1} vs {1,2}
        assert index_recall(approx, exact, 4)[0] == pytest.approx(0.5)   # {2,1,9,8} vs {1,2,3,4}

    def test_padding_counts_as_miss(self):
        # IVF found only 1 real result for k=3; the -1 slots are misses, denominator stays k
        exact = np.array([[5, 6, 7]])
        approx = np.array([[5, -1, -1]])
        mean, _ = index_recall(approx, exact, 3)
        assert mean == pytest.approx(1 / 3)

    def test_per_query_values_and_mean(self):
        exact = np.array([[1, 2], [3, 4]])
        approx = np.array([[1, 2], [3, 9]])
        mean, per = index_recall(approx, exact, 2)
        assert np.allclose(per, [1.0, 0.5])
        assert mean == pytest.approx(0.75)

    def test_per_query_is_float_array(self):
        exact = np.array([[1, 2], [3, 4], [5, 6]])
        _, per = index_recall(exact.copy(), exact, 2)
        assert isinstance(per, np.ndarray)
        assert per.shape == (3,)
        assert np.issubdtype(per.dtype, np.floating)

    def test_mean_is_a_float(self):
        exact = np.array([[1, 2]])
        mean, _ = index_recall(exact.copy(), exact, 2)
        assert isinstance(mean, float)

    def test_mean_equals_average_of_per_query(self, ):
        rng = np.random.default_rng(0)
        exact = np.stack([rng.permutation(50)[:20] for _ in range(30)])
        approx = np.stack([rng.permutation(50)[:20] for _ in range(30)])
        mean, per = index_recall(approx, exact, 20)
        assert mean == pytest.approx(per.mean())

    def test_matches_isin_oracle_on_wide_arrays(self):
        # the Experiment A shape: (B, 100) arrays, reported at several k
        rng = np.random.default_rng(1)
        B = 40
        exact = np.stack([rng.permutation(1000)[:100] for _ in range(B)])
        approx = exact.copy()
        for i in range(B):                      # replace a random number of slots with misses
            n = rng.integers(0, 30)
            approx[i, rng.choice(100, n, replace=False)] = 1000 + np.arange(n)
        for k in [1, 10, 100]:
            expected = np.array([np.isin(approx[i, :k], exact[i, :k]).sum() / k
                                 for i in range(B)])
            mean, per = index_recall(approx, exact, k)
            assert np.allclose(per, expected), f"k={k}"
            assert mean == pytest.approx(expected.mean())

    def test_mismatched_batch_raises(self):
        with pytest.raises(ValueError):
            index_recall(np.array([[1, 2], [3, 4]]), np.array([[1, 2]]), 2)

# ───────────────────── retrieval metrics (Deliverable 4) ─────────────────────
# Concept 9 hand example: relevant D2 (grade 2) and D7 (grade 1)
HAND_RANKED = ["D4", "D2", "D9", "D7", "D1"]
HAND_REL = {"D2": 2, "D7": 1}


class TestRecallAtK:
    def test_hand_example(self):
        assert recall_at_k(HAND_RANKED, HAND_REL, 5) == pytest.approx(1.0)
        assert recall_at_k(HAND_RANKED, HAND_REL, 2) == pytest.approx(0.5)
        assert recall_at_k(HAND_RANKED, HAND_REL, 1) == pytest.approx(0.0)

    def test_only_top_k_counts(self):
        # a relevant doc at rank 4 must NOT count toward recall@3
        assert recall_at_k(["a", "b", "c", "R"], {"R": 1}, 3) == pytest.approx(0.0)

    def test_grade_zero_is_not_relevant(self):
        # "A" is judged NOT relevant; only "B" counts, so finding B = 1/1
        assert recall_at_k(["A", "B", "C"], {"A": 0, "B": 1}, 3) == pytest.approx(1.0)

    def test_checkpoint7_example(self):
        assert recall_at_k(["D1", "D5", "D3", "D8"], {"D5": 1, "D8": 1}, 3) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(recall_at_k(HAND_RANKED, HAND_REL, 5), float)


class TestPrecisionAtK:
    def test_hand_example(self):
        assert precision_at_k(HAND_RANKED, HAND_REL, 5) == pytest.approx(0.4)
        assert precision_at_k(HAND_RANKED, HAND_REL, 2) == pytest.approx(0.5)
        assert precision_at_k(HAND_RANKED, HAND_REL, 1) == pytest.approx(0.0)

    def test_denominator_is_k(self):
        # 1 relevant in the top 3 → 1/3, regardless of how long the ranked list is
        ranked = ["R", "x", "y", "z", "w", "v"]
        assert precision_at_k(ranked, {"R": 1}, 3) == pytest.approx(1 / 3)

    def test_grade_zero_is_not_relevant(self):
        assert precision_at_k(["A", "B", "C"], {"A": 0, "B": 1}, 3) == pytest.approx(1 / 3)

    def test_returns_float(self):
        assert isinstance(precision_at_k(HAND_RANKED, HAND_REL, 5), float)


class TestMRRAtK:
    def test_hand_example(self):
        assert mrr_at_k(HAND_RANKED, HAND_REL, 5) == pytest.approx(0.5)

    def test_first_relevant_beyond_k_is_zero(self):
        # first relevant doc is at rank 2, so MRR@1 must be 0
        assert mrr_at_k(HAND_RANKED, HAND_REL, 1) == pytest.approx(0.0)

    def test_relevant_at_rank_one(self):
        assert mrr_at_k(["R", "x"], {"R": 1}, 2) == pytest.approx(1.0)

    def test_no_relevant_is_zero(self):
        assert mrr_at_k(["a", "b", "c"], {"R": 1}, 3) == pytest.approx(0.0)

    def test_grade_zero_is_not_relevant(self):
        # "A" sits at rank 1 but is judged not relevant; first relevant is B at rank 2
        assert mrr_at_k(["A", "B", "C"], {"A": 0, "B": 1}, 3) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(mrr_at_k(["a", "b"], {"R": 1}, 2), float)


class TestNDCGAtK:
    def test_hand_example(self):
        # DCG = 2/log2(3) + 1/log2(5) = 1.69254 ; ideal = 2 + 1/log2(3) = 2.63093
        assert ndcg_at_k(HAND_RANKED, HAND_REL, 5) == pytest.approx(0.643318, abs=1e-4)

    def test_hand_example_small_k(self):
        assert ndcg_at_k(HAND_RANKED, HAND_REL, 1) == pytest.approx(0.0)             # D4 not relevant
        assert ndcg_at_k(HAND_RANKED, HAND_REL, 2) == pytest.approx(0.479624, abs=1e-4)

    def test_kc9_graded_example(self):
        # relevant D3 (2), D9 (1); ranked [D9, D1, D3]: DCG = 1 + 2/2 = 2 ; ideal = 2.63093
        assert ndcg_at_k(["D9", "D1", "D3"], {"D3": 2, "D9": 1}, 3) == pytest.approx(0.760190, abs=1e-4)

    def test_checkpoint7_binary_example(self):
        assert ndcg_at_k(["D1", "D5", "D3", "D8"], {"D5": 1, "D8": 1}, 3) == pytest.approx(0.386853, abs=1e-4)

    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["hi", "lo", "x"], {"hi": 2, "lo": 1}, 3) == pytest.approx(1.0)

    def test_ideal_is_sorted_by_grade(self):
        # swapping the grade-2 and grade-1 docs must cost something
        assert ndcg_at_k(["lo", "hi"], {"lo": 1, "hi": 2}, 2) == pytest.approx(0.859719, abs=1e-4)

    def test_ideal_is_truncated_to_k(self):
        # 3 relevant docs exist but k=2: two relevant docs in the top 2 is PERFECT
        assert ndcg_at_k(["a", "b", "c"], {"a": 1, "b": 1, "c": 1}, 2) == pytest.approx(1.0)

    def test_grade_zero_is_not_relevant(self):
        # DCG = 0 + 1/log2(3) ; ideal = 1 (only B is relevant)
        assert ndcg_at_k(["A", "B", "C"], {"A": 0, "B": 1}, 3) == pytest.approx(0.630930, abs=1e-4)

    def test_no_relevant_in_top_k_is_zero(self):
        assert ndcg_at_k(["a", "b", "c"], {"R": 1}, 3) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(ndcg_at_k(HAND_RANKED, HAND_REL, 5), float)


class TestEvaluate:
    @pytest.fixture
    def two_queries(self):
        run = {"q1": HAND_RANKED,
               "q2": ["X", "Y", "Z", "W", "V"]}
        qrels = {"q1": HAND_REL,
                 "q2": {"X": 1}}
        return run, qrels

    def test_returns_means_and_per_query(self, two_queries):
        run, qrels = two_queries
        means, per_query = evaluate(run, qrels, k_values=(1, 5))
        assert isinstance(means, dict) and isinstance(per_query, dict)
        assert set(per_query) == {"q1", "q2"}

    def test_metric_keys(self, two_queries):
        run, qrels = two_queries
        means, per_query = evaluate(run, qrels, k_values=(1, 5))
        expected = {f"{m}@{k}" for m in ("recall", "precision", "mrr", "ndcg") for k in (1, 5)}
        assert set(means) == expected
        assert set(per_query["q1"]) == expected

    def test_default_k_values(self, two_queries):
        run, qrels = two_queries
        means, _ = evaluate(run, qrels)
        for k in (1, 10, 100):
            assert f"ndcg@{k}" in means and f"recall@{k}" in means

    def test_hand_computed_means(self, two_queries):
        run, qrels = two_queries
        means, _ = evaluate(run, qrels, k_values=(1, 5))
        assert means["recall@5"] == pytest.approx(1.0)
        assert means["recall@1"] == pytest.approx(0.5)                    # (0 + 1) / 2
        assert means["mrr@5"] == pytest.approx(0.75)                      # (0.5 + 1) / 2
        assert means["precision@5"] == pytest.approx(0.3)                 # (0.4 + 0.2) / 2
        assert means["ndcg@5"] == pytest.approx((0.643318 + 1.0) / 2, abs=1e-4)

    def test_means_are_averages_of_per_query(self, two_queries):
        run, qrels = two_queries
        means, per_query = evaluate(run, qrels, k_values=(1, 5))
        for key, value in means.items():
            assert value == pytest.approx(np.mean([per_query[q][key] for q in per_query]))
            assert isinstance(value, float)

    def test_extra_run_queries_are_ignored(self, two_queries):
        run, qrels = two_queries
        base, _ = evaluate(run, qrels, k_values=(5,))
        run_extra = dict(run, q_unlabeled=["X", "D2"])                   # not in qrels
        means, per_query = evaluate(run_extra, qrels, k_values=(5,))
        assert "q_unlabeled" not in per_query
        assert means == pytest.approx(base)

    def test_missing_qrels_query_raises(self, two_queries):
        run, qrels = two_queries
        with pytest.raises(ValueError):
            evaluate({"q1": run["q1"]}, qrels)                            # q2 missing from run