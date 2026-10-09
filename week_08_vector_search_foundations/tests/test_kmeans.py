"""Tests for search/kmeans.py — Week 8, Deliverable 2.

Run from week_08_vector_search/:
    python -m pytest tests/test_kmeans.py -v
"""
import numpy as np
import pytest

from search.kmeans import kmeans, assign
from search.brute_force import normalize


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_blobs(rng, directions, n_per=50, spread=0.05):
    """Tight groups of points around given DIRECTIONS (spherical k-means
    clusters by direction). Returns (X float32, true group per row)."""
    pts, truth = [], []
    for j, c in enumerate(directions):
        c = np.asarray(c, dtype=np.float64)
        c = c / np.linalg.norm(c)
        pts.append(c + spread * rng.normal(size=(n_per, len(c))))
        truth += [j] * n_per
    return np.vstack(pts).astype(np.float32), np.array(truth)


# ─────────────────────────────── assign ───────────────────────────────

class TestAssign:
    def test_hand_example(self):
        X = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
        C = np.array([[1, 0], [0, 1]], dtype=np.float32)
        labels, best = assign(X, C)
        assert labels.tolist() == [0, 1, 1]               # 0.8 beats 0.6
        assert np.allclose(best, [1.0, 1.0, 0.8], atol=1e-6)

    def test_shapes_and_dtypes(self, rng):
        X = normalize(rng.normal(size=(100, 8)))
        C = normalize(rng.normal(size=(5, 8)))
        labels, best = assign(X, C)
        assert labels.shape == (100,) and best.shape == (100,)
        assert labels.dtype == np.int64
        assert best.dtype == np.float32

    def test_matches_argmax_oracle(self, rng):
        X = normalize(rng.normal(size=(300, 8)))
        C = normalize(rng.normal(size=(7, 8)))
        labels, best = assign(X, C)
        S = X.astype(np.float64) @ C.astype(np.float64).T
        assert labels.tolist() == S.argmax(axis=1).tolist()
        assert np.allclose(best, S.max(axis=1), atol=1e-5)

    def test_best_is_score_of_assigned_centroid(self, rng):
        X = normalize(rng.normal(size=(50, 8)))
        C = normalize(rng.normal(size=(4, 8)))
        labels, best = assign(X, C)
        assert np.allclose(best, np.sum(X * C[labels], axis=1), atol=1e-5)

    @pytest.mark.parametrize("chunk_size", [1, 7, 64, 10_000])
    def test_chunk_size_does_not_change_results(self, rng, chunk_size):
        X = normalize(rng.normal(size=(150, 8)))
        C = normalize(rng.normal(size=(6, 8)))
        ref_labels, ref_best = assign(X, C, chunk_size=8192)
        labels, best = assign(X, C, chunk_size=chunk_size)
        assert np.array_equal(labels, ref_labels)
        assert np.allclose(best, ref_best, atol=1e-6)


# ─────────────────────────────── kmeans ───────────────────────────────

class TestKMeans:
    def test_shapes_and_dtypes(self, rng):
        X = rng.normal(size=(200, 8)).astype(np.float32)
        C, labels, history = kmeans(X, 5)
        assert C.shape == (5, 8) and C.dtype == np.float32
        assert labels.shape == (200,) and labels.dtype == np.int64
        assert isinstance(history, list) and len(history) >= 1

    def test_centroids_unit_norm_no_nan(self, rng):
        C, _, _ = kmeans(rng.normal(size=(300, 8)), 10)
        assert not np.isnan(C).any()
        assert np.allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-5)

    def test_labels_in_range(self, rng):
        _, labels, _ = kmeans(rng.normal(size=(300, 8)), 10)
        assert labels.min() >= 0 and labels.max() < 10

    def test_recovers_two_blobs(self, rng):
        X, truth = make_blobs(rng, [[1, 1], [-1, -1]])
        _, labels, _ = kmeans(X, 2)
        a, b = set(labels[truth == 0].tolist()), set(labels[truth == 1].tolist())
        assert len(a) == 1 and len(b) == 1 and a != b      # one label per blob, different labels

    def test_centroids_point_at_blob_directions(self, rng):
        dirs = np.eye(3, dtype=np.float32)
        X, _ = make_blobs(rng, dirs, n_per=60)
        C, _, _ = kmeans(X, 3)
        # every true direction has a centroid within a few degrees of it
        assert np.all((dirs @ C.T).max(axis=1) > 0.99)

    def test_history_never_decreases(self, rng):
        X = rng.normal(size=(500, 8)).astype(np.float32)
        _, _, history = kmeans(X, 12, max_iter=25)
        assert np.all(np.diff(np.array(history)) >= -1e-6), history

    def test_history_values_are_cosines(self, rng):
        X, _ = make_blobs(rng, [[1, 1], [-1, -1]])
        _, _, history = kmeans(X, 2)
        assert all(-1.0 - 1e-6 <= h <= 1.0 + 1e-6 for h in history)
        assert history[-1] > 0.9          # tight blobs → mean best-cosine near 1

    def test_stops_early_when_converged(self, rng):
        X, _ = make_blobs(rng, [[1, 1], [-1, -1]])
        _, _, history = kmeans(X, 2, max_iter=25)
        assert len(history) < 25

    def test_respects_max_iter(self, rng):
        X = rng.normal(size=(500, 8)).astype(np.float32)
        _, _, history = kmeans(X, 12, max_iter=3)
        assert len(history) <= 3

    def test_labels_match_centroids_after_convergence(self, rng):
        X, _ = make_blobs(rng, np.eye(3))
        C, labels, _ = kmeans(X, 3)
        expected, _ = assign(normalize(X), C)
        assert np.array_equal(labels, expected)

    def test_same_seed_same_result(self, rng):
        X = rng.normal(size=(300, 8)).astype(np.float32)
        C1, l1, _ = kmeans(X, 6, seed=3)
        C2, l2, _ = kmeans(X, 6, seed=3)
        assert np.array_equal(C1, C2) and np.array_equal(l1, l2)

    def test_input_scale_does_not_matter(self, rng):
        X = rng.normal(size=(300, 8)).astype(np.float32)
        _, l1, _ = kmeans(X, 6, seed=1)
        _, l2, _ = kmeans(X * 25.0, 6, seed=1)
        assert np.array_equal(l1, l2)

    def test_does_not_modify_input(self, rng):
        X = rng.normal(size=(100, 8)).astype(np.float32)
        before = X.copy()
        kmeans(X, 4)
        assert np.array_equal(X, before)

    def test_empty_clusters_never_produce_nan(self):
        # 20 copies of one point + 20 of another with k=4: at least two
        # centroids start as identical vectors, so some cluster goes empty
        X = np.vstack([np.tile([1.0, 0.0], (20, 1)),
                       np.tile([0.0, 1.0], (20, 1))]).astype(np.float32)
        C, labels, history = kmeans(X, 4, max_iter=10)
        assert not np.isnan(C).any()
        assert np.allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-5)
        assert not np.isnan(history).any()

    @pytest.mark.parametrize("k", [0, -2])
    def test_bad_k_raises(self, rng, k):
        with pytest.raises(ValueError):
            kmeans(rng.normal(size=(10, 4)), k)

    def test_k_larger_than_n_raises(self, rng):
        with pytest.raises(ValueError):
            kmeans(rng.normal(size=(10, 4)), 11)

    def test_labels_match_centroids_when_max_iter_hit(self, rng):
        # random data + tiny max_iter: k-means cannot converge in 2 iterations
        X = rng.normal(size=(500, 8)).astype(np.float32)
        C, labels, history = kmeans(X, 12, max_iter=2)
        assert len(history) == 2                      # confirms we really hit the cap
        expected, _ = assign(normalize(X), C)
        assert np.array_equal(labels, expected)