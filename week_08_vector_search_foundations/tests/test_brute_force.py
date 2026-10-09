"""Tests for search/brute_force.py — Week 8, Deliverable 1.

Run from week_08_vector_search/:
    python -m pytest tests/test_brute_force.py -v
"""
import numpy as np
import pytest

from search.brute_force import normalize, top_k, top_k_batch, BruteForceIndex


# ───────────────────────── fixtures & helpers ─────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def corpus(rng):
    # 200 random 16-dim vectors. Continuous values -> no exact score ties,
    # so exact ordering comparisons are safe.
    return rng.normal(size=(200, 16)).astype(np.float32)


@pytest.fixture
def index(corpus):
    return BruteForceIndex(corpus)


def naive_search(corpus, q, k):
    """Obviously-correct oracle: explicit cosine in float64 + a FULL argsort."""
    c = corpus.astype(np.float64)
    qq = np.asarray(q, dtype=np.float64)
    cos = (c @ qq) / (np.linalg.norm(c, axis=1) * np.linalg.norm(qq))
    order = np.argsort(-cos)[:k]
    return order, cos[order]


# Concept 3 hand example: cosines vs q are [1.0, 0.96, 0.6, 0.6]
HAND_CORPUS = np.array([[30, 40], [4, 3], [5, 0], [100, 0]], dtype=np.float32)
HAND_QUERY = np.array([3, 4], dtype=np.float32)


# ───────────────────────────── normalize ─────────────────────────────

class TestNormalize:
    def test_rows_have_unit_norm(self, corpus):
        out = normalize(corpus)
        assert out.shape == corpus.shape
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_1d_input(self):
        out = normalize(np.array([3.0, 4.0]))
        assert out.shape == (2,)
        assert np.allclose(out, [0.6, 0.8], atol=1e-6)

    def test_zero_row_stays_zero(self):
        X = np.array([[0, 0, 0], [1, 2, 2]], dtype=np.float32)
        out = normalize(X)
        assert not np.isnan(out).any()
        assert np.array_equal(out[0], [0, 0, 0])
        assert np.allclose(out[1], [1 / 3, 2 / 3, 2 / 3], atol=1e-6)

    def test_returns_float32_from_float64_input(self, rng):
        out = normalize(rng.normal(size=(10, 4)))      # float64 in
        assert out.dtype == np.float32

    def test_does_not_modify_input(self, corpus):
        before = corpus.copy()
        normalize(corpus)
        assert np.array_equal(corpus, before)

    def test_idempotent(self, corpus):
        once = normalize(corpus)
        assert np.allclose(normalize(once), once, atol=1e-6)

    def test_scale_does_not_change_direction(self):
        v = np.array([1.0, 2.0, 2.0])
        assert np.allclose(normalize(v), normalize(10 * v), atol=1e-6)


# ─────────────────────────────── top_k ───────────────────────────────

class TestTopK:
    def test_hand_example(self):
        s = np.array([0.2, 0.9, 0.1, 0.7, 0.5], dtype=np.float32)
        idx, vals = top_k(s, 3)
        assert idx.tolist() == [1, 3, 4]
        assert np.allclose(vals, [0.9, 0.7, 0.5])

    def test_indices_point_into_original_scores(self, rng):
        s = rng.random(1000).astype(np.float32)
        idx, vals = top_k(s, 10)
        # each returned value must BE the score at its returned index
        assert np.array_equal(vals, s[idx])

    def test_matches_full_argsort(self, rng):
        s = rng.random(1000)
        for k in [1, 5, 50, 999]:
            idx, _ = top_k(s, k)
            assert idx.tolist() == np.argsort(-s)[:k].tolist(), f"k={k}"

    def test_sorted_best_first(self, rng):
        _, vals = top_k(rng.random(500), 20)
        assert np.all(np.diff(vals) <= 0)

    def test_k_equals_n(self, rng):
        s = rng.random(7)
        idx, vals = top_k(s, 7)
        assert sorted(idx.tolist()) == list(range(7))
        assert np.all(np.diff(vals) <= 0)

    def test_k_larger_than_n_clamps(self, rng):
        idx, vals = top_k(rng.random(4), 10)
        assert len(idx) == 4 and len(vals) == 4

    def test_negative_scores(self):
        s = np.array([-0.5, -0.1, -0.9, -0.3])
        idx, _ = top_k(s, 2)
        assert idx.tolist() == [1, 3]


# ──────────────────────────── top_k_batch ────────────────────────────

class TestTopKBatch:
    def test_shapes(self, rng):
        idx, vals = top_k_batch(rng.random((6, 100)), 5)
        assert idx.shape == (6, 5) and vals.shape == (6, 5)

    def test_matches_top_k_row_by_row(self, rng):
        S = rng.random((12, 300))
        idx, vals = top_k_batch(S, 8)
        for r in range(S.shape[0]):
            i_r, v_r = top_k(S[r], 8)
            assert idx[r].tolist() == i_r.tolist(), f"row {r}"
            assert np.allclose(vals[r], v_r)

    def test_indices_point_into_original_rows(self, rng):
        S = rng.random((5, 200))
        idx, vals = top_k_batch(S, 10)
        assert np.array_equal(vals, np.take_along_axis(S, idx, axis=1))

    def test_rows_sorted_best_first(self, rng):
        _, vals = top_k_batch(rng.random((5, 200)), 10)
        assert np.all(np.diff(vals, axis=1) <= 0)

    def test_rows_are_independent(self):
        # each row's best entries sit in DIFFERENT columns
        S = np.array([[0.9, 0.1, 0.5, 0.3],
                      [0.1, 0.2, 0.3, 0.8]])
        idx, _ = top_k_batch(S, 2)
        assert idx.tolist() == [[0, 2], [3, 2]]

    def test_k_larger_than_n_clamps(self, rng):
        idx, vals = top_k_batch(rng.random((3, 4)), 10)
        assert idx.shape == (3, 4) and vals.shape == (3, 4)

    def test_k_equals_n(self, rng):
        idx, _ = top_k_batch(rng.random((3, 6)), 6)
        for r in range(3):
            assert sorted(idx[r].tolist()) == list(range(6))


# ─────────────────────── BruteForceIndex: build ───────────────────────

class TestIndexInit:
    def test_attributes(self, index):
        assert index.N == 200 and index.d == 16
        assert index.E.shape == (200, 16)

    def test_corpus_normalized_and_float32(self, index):
        assert index.E.dtype == np.float32
        assert np.allclose(np.linalg.norm(index.E, axis=1), 1.0, atol=1e-5)

    def test_does_not_modify_input(self, corpus):
        before = corpus.copy()
        BruteForceIndex(corpus)
        assert np.array_equal(corpus, before)

    def test_float64_input_stored_as_float32(self, rng):
        assert BruteForceIndex(rng.normal(size=(20, 4))).E.dtype == np.float32

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            BruteForceIndex(np.ones(16, dtype=np.float32))


# ─────────────────────── BruteForceIndex: search ──────────────────────

class TestSearch:
    def test_hand_example(self):
        idx = BruteForceIndex(HAND_CORPUS)
        ids, scores = idx.search(HAND_QUERY, 3)
        assert ids[0] == 0 and ids[1] == 1
        assert ids[2] in (2, 3)                     # docs 2 and 3 tie at 0.6
        assert np.allclose(scores, [1.0, 0.96, 0.6], atol=1e-5)

    def test_matches_naive_oracle(self, corpus, index, rng):
        for _ in range(20):
            q = rng.normal(size=16).astype(np.float32)
            ids, scores = index.search(q, 10)
            exp_ids, exp_scores = naive_search(corpus, q, 10)
            assert ids.tolist() == exp_ids.tolist()
            assert np.allclose(scores, exp_scores, atol=1e-5)

    def test_returned_scores_match_returned_ids(self, index, rng):
        q = rng.normal(size=16).astype(np.float32)
        ids, scores = index.search(q, 10)
        recomputed = index.E[ids] @ (q / np.linalg.norm(q))
        assert np.allclose(scores, recomputed, atol=1e-5)

    def test_self_retrieval(self, corpus, index):
        for i in [0, 17, 99, 199]:
            ids, scores = index.search(corpus[i], 1)
            assert ids[0] == i
            assert np.isclose(scores[0], 1.0, atol=1e-5)

    def test_query_scale_does_not_matter(self, index, rng):
        q = rng.normal(size=16).astype(np.float32)
        a, _ = index.search(q, 10)
        b, _ = index.search(q * 50.0, 10)
        assert a.tolist() == b.tolist()

    def test_scores_in_cosine_range(self, index, rng):
        _, scores = index.search(rng.normal(size=16), 200)
        assert np.all(scores <= 1.0 + 1e-5) and np.all(scores >= -1.0 - 1e-5)

    def test_k_larger_than_n(self, index, rng):
        ids, scores = index.search(rng.normal(size=16), 1000)
        assert len(ids) == 200 and len(scores) == 200
        assert sorted(ids.tolist()) == list(range(200))

    def test_zero_query_gives_no_nan(self, index):
        _, scores = index.search(np.zeros(16, dtype=np.float32), 5)
        assert not np.isnan(scores).any()

    def test_wrong_dimension_raises(self, index):
        with pytest.raises(ValueError):
            index.search(np.ones(15, dtype=np.float32), 5)

    def test_2d_query_raises(self, index):
        with pytest.raises(ValueError):
            index.search(np.ones((1, 16), dtype=np.float32), 5)

    @pytest.mark.parametrize("k", [0, -3])
    def test_bad_k_raises(self, index, k):
        with pytest.raises(ValueError):
            index.search(np.ones(16, dtype=np.float32), k)


# ──────────────────── BruteForceIndex: search_batch ───────────────────

class TestSearchBatch:
    def test_shapes_and_dtypes(self, index, rng):
        Q = rng.normal(size=(37, 16)).astype(np.float32)
        ids, scores = index.search_batch(Q, 5)
        assert ids.shape == (37, 5) and scores.shape == (37, 5)
        assert ids.dtype == np.int64 and scores.dtype == np.float32

    def test_equals_single_queries(self, index, rng):
        # 37 queries, chunk 10 -> three full chunks + one partial chunk of 7
        Q = rng.normal(size=(37, 16)).astype(np.float32)
        ids, scores = index.search_batch(Q, 5, chunk_size=10)
        for i in range(37):
            s_ids, s_scores = index.search(Q[i], 5)
            assert ids[i].tolist() == s_ids.tolist(), f"query {i}"
            assert np.allclose(scores[i], s_scores, atol=1e-5)

    @pytest.mark.parametrize("chunk_size", [1, 7, 37, 1000])
    def test_chunk_size_does_not_change_results(self, index, rng, chunk_size):
        Q = rng.normal(size=(37, 16)).astype(np.float32)
        ref_ids, ref_scores = index.search_batch(Q, 5, chunk_size=256)
        ids, scores = index.search_batch(Q, 5, chunk_size=chunk_size)
        assert np.array_equal(ids, ref_ids)
        assert np.allclose(scores, ref_scores, atol=1e-5)

    def test_matches_naive_oracle(self, corpus, index, rng):
        Q = rng.normal(size=(15, 16)).astype(np.float32)
        ids, scores = index.search_batch(Q, 10)
        for i in range(15):
            exp_ids, exp_scores = naive_search(corpus, Q[i], 10)
            assert ids[i].tolist() == exp_ids.tolist(), f"query {i}"
            assert np.allclose(scores[i], exp_scores, atol=1e-5)

    def test_self_retrieval(self, corpus, index):
        ids, scores = index.search_batch(corpus, 1)
        assert ids[:, 0].tolist() == list(range(200))
        assert np.allclose(scores[:, 0], 1.0, atol=1e-5)

    def test_k_larger_than_n(self, index, rng):
        ids, scores = index.search_batch(rng.normal(size=(3, 16)), 1000)
        assert ids.shape == (3, 200) and scores.shape == (3, 200)

    def test_zero_query_row_gives_no_nan(self, index, rng):
        Q = rng.normal(size=(3, 16)).astype(np.float32)
        Q[1] = 0.0
        _, scores = index.search_batch(Q, 5)
        assert not np.isnan(scores).any()

    def test_wrong_dimension_raises(self, index):
        with pytest.raises(ValueError):
            index.search_batch(np.ones((4, 15), dtype=np.float32), 5)

    def test_1d_input_raises(self, index):
        with pytest.raises(ValueError):
            index.search_batch(np.ones(16, dtype=np.float32), 5)

    @pytest.mark.parametrize("k", [0, -3])
    def test_bad_k_raises(self, index, k):
        with pytest.raises(ValueError):
            index.search_batch(np.ones((2, 16), dtype=np.float32), k)