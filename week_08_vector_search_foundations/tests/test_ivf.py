"""Tests for search/ivf.py — Week 8, Deliverable 2.

Run from week_08_vector_search/:
    python -m pytest tests/test_ivf.py -v
"""
import numpy as np
import pytest

from search.ivf import IVFIndex
from search.kmeans import assign
from search.brute_force import BruteForceIndex, normalize

N, D, NLIST = 1000, 16, 10


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def corpus(rng):
    return rng.normal(size=(N, D)).astype(np.float32)


@pytest.fixture
def ivf(corpus):
    return IVFIndex(corpus, NLIST, seed=0)


@pytest.fixture
def bf(corpus):
    return BruteForceIndex(corpus)


@pytest.fixture
def queries(rng):
    return rng.normal(size=(25, D)).astype(np.float32)


def index_recall(ids, true_ids):
    return len(set(list(ids)) & set(list(true_ids))) / len(true_ids)


# ──────────────────────────────── build ───────────────────────────────

class TestBuild:
    def test_attributes(self, ivf):
        assert ivf.N == N and ivf.d == D and ivf.nlist == NLIST
        assert ivf.E.shape == (N, D) and ivf.E.dtype == np.float32
        assert np.allclose(np.linalg.norm(ivf.E, axis=1), 1.0, atol=1e-5)
        assert ivf.centroids.shape == (NLIST, D)
        assert np.allclose(np.linalg.norm(ivf.centroids, axis=1), 1.0, atol=1e-5)
        assert ivf.labels.shape == (N,)
        assert len(ivf.lists) == NLIST

    def test_lists_partition_the_corpus(self, ivf):
        all_ids = np.concatenate(ivf.lists)
        assert len(all_ids) == N                          # nothing missing, nothing doubled
        assert np.array_equal(np.sort(all_ids), np.arange(N))

    def test_list_members_carry_that_label(self, ivf):
        for c, members in enumerate(ivf.lists):
            assert np.all(ivf.labels[members] == c), f"cluster {c}"

    def test_labels_are_nearest_centroid(self, ivf):
        expected, _ = assign(ivf.E, ivf.centroids)
        assert np.array_equal(ivf.labels, expected)

    def test_labels_are_nearest_centroid_when_kmeans_stops_early(self, corpus):
        ivf = IVFIndex(corpus, NLIST, max_iter=1, seed=0)
        expected, _ = assign(ivf.E, ivf.centroids)
        assert np.array_equal(ivf.labels, expected)

    def test_cluster_sizes(self, ivf):
        sizes = ivf.cluster_sizes()
        assert sizes.shape == (NLIST,)
        assert sizes.sum() == N
        assert sizes.tolist() == [len(l) for l in ivf.lists]

    def test_training_on_a_sample_still_indexes_everything(self, corpus):
        ivf = IVFIndex(corpus, NLIST, train_size=200, seed=0)
        assert ivf.cluster_sizes().sum() == N
        assert np.array_equal(np.sort(np.concatenate(ivf.lists)), np.arange(N))
        expected, _ = assign(ivf.E, ivf.centroids)
        assert np.array_equal(ivf.labels, expected)

    def test_train_size_at_least_n_uses_all(self, corpus):
        ivf = IVFIndex(corpus, NLIST, train_size=N + 100, seed=0)
        assert ivf.cluster_sizes().sum() == N

    def test_same_seed_same_index(self, corpus):
        a = IVFIndex(corpus, NLIST, train_size=300, seed=4)
        b = IVFIndex(corpus, NLIST, train_size=300, seed=4)
        assert np.array_equal(a.centroids, b.centroids)
        assert np.array_equal(a.labels, b.labels)

    def test_does_not_modify_input(self, corpus):
        before = corpus.copy()
        IVFIndex(corpus, NLIST)
        assert np.array_equal(corpus, before)

    @pytest.mark.parametrize("nlist", [0, -1, N + 1])
    def test_bad_nlist_raises(self, corpus, nlist):
        with pytest.raises(ValueError):
            IVFIndex(corpus, nlist)


# ─────────────────────────────── search ───────────────────────────────

class TestSearch:
    def test_probing_every_list_equals_brute_force(self, ivf, bf, queries):
        for q in queries:
            ids, scores = ivf.search(q, 10, nprobe=NLIST)
            bf_ids, bf_scores = bf.search(q, 10)
            assert list(ids) == list(bf_ids)
            assert np.allclose(scores, bf_scores, atol=1e-5)

    def test_nprobe_above_nlist_clamps(self, ivf, bf, queries):
        ids, _ = ivf.search(queries[0], 10, nprobe=NLIST * 5)
        bf_ids, _ = bf.search(queries[0], 10)
        assert list(ids) == list(bf_ids)

    def test_returns_global_ids(self, ivf, queries):
        q = queries[0]
        ids, scores = ivf.search(q, 10, nprobe=3)
        assert np.allclose(scores, ivf.E[ids] @ normalize(q), atol=1e-5)

    def test_results_come_from_probed_clusters(self, ivf, queries):
        q = queries[0]
        ids, _ = ivf.search(q, 10, nprobe=2)
        probed = set(np.argsort(-(ivf.centroids @ normalize(q)))[:2].tolist())
        assert set(ivf.labels[ids].tolist()) <= probed

    def test_sorted_best_first_and_unique(self, ivf, queries):
        ids, scores = ivf.search(queries[0], 20, nprobe=3)
        assert np.all(np.diff(scores) <= 0)
        assert len(set(list(ids))) == len(ids)

    def test_index_recall_never_drops_as_nprobe_grows(self, ivf, bf, queries):
        prev = 0.0
        for nprobe in range(1, NLIST + 1):
            r = np.mean([index_recall(ivf.search(q, 10, nprobe)[0], bf.search(q, 10)[0])
                         for q in queries])
            assert r >= prev - 1e-12, f"nprobe={nprobe}: {r} < {prev}"
            prev = r
        assert prev == 1.0                                 # all lists probed → exact

    def test_fewer_candidates_than_k(self, ivf, queries):
        ids, scores = ivf.search(queries[0], N, nprobe=1)  # one list can't hold all N
        assert len(ids) == len(scores) < N
        assert len(set(list(ids))) == len(ids)

    def test_all_probed_lists_empty_returns_empty(self, ivf, queries):
        ivf.lists = [np.array([], dtype=np.int64) for _ in range(NLIST)]
        ids, scores = ivf.search(queries[0], 5, nprobe=2)
        assert len(ids) == 0 and len(scores) == 0

    def test_wrong_dimension_raises(self, ivf):
        with pytest.raises(ValueError):
            ivf.search(np.ones(D - 1, dtype=np.float32), 5)

    def test_2d_query_raises(self, ivf):
        with pytest.raises(ValueError):
            ivf.search(np.ones((1, D), dtype=np.float32), 5)

    @pytest.mark.parametrize("k", [0, -1])
    def test_bad_k_raises(self, ivf, k):
        with pytest.raises(ValueError):
            ivf.search(np.ones(D, dtype=np.float32), k)

    @pytest.mark.parametrize("nprobe", [0, -1])
    def test_bad_nprobe_raises(self, ivf, nprobe):
        with pytest.raises(ValueError):
            ivf.search(np.ones(D, dtype=np.float32), 5, nprobe=nprobe)


# ──────────────────────────── search_batch ────────────────────────────

class TestSearchBatch:
    def test_shapes_and_dtypes(self, ivf, queries):
        ids, scores = ivf.search_batch(queries, 5, nprobe=2)
        assert ids.shape == (25, 5) and scores.shape == (25, 5)
        assert ids.dtype == np.int64 and scores.dtype == np.float32

    def test_equals_single_queries(self, ivf, queries):
        ids, scores = ivf.search_batch(queries, 10, nprobe=2)
        for i, q in enumerate(queries):
            s_ids, s_scores = ivf.search(q, 10, nprobe=2)
            m = len(s_ids)
            assert list(ids[i, :m]) == list(s_ids), f"query {i}"
            assert np.allclose(scores[i, :m], s_scores, atol=1e-5)

    def test_probing_every_list_equals_brute_force(self, ivf, bf, queries):
        ids, scores = ivf.search_batch(queries, 10, nprobe=NLIST)
        bf_ids, bf_scores = bf.search_batch(queries, 10)
        assert np.array_equal(ids, bf_ids)
        assert np.allclose(scores, bf_scores, atol=1e-5)

    def test_short_rows_are_padded(self, ivf, queries):
        ids, scores = ivf.search_batch(queries, N, nprobe=1)
        assert ids.shape == (25, N)
        pad = ids == -1
        assert pad.any()                                   # one list can't fill N slots
        assert np.all(np.isneginf(scores[pad]))            # −1 and −inf go together
        assert not np.isneginf(scores[~pad]).any()
        for i, q in enumerate(queries[:5]):
            s_ids, _ = ivf.search(q, N, nprobe=1)
            assert (~pad[i]).sum() == len(s_ids)           # real results = single search
            assert np.all(pad[i, len(s_ids):])             # padding only at the END

    def test_all_lists_empty_gives_all_padding(self, ivf, queries):
        ivf.lists = [np.array([], dtype=np.int64) for _ in range(NLIST)]
        ids, scores = ivf.search_batch(queries, 5, nprobe=2)
        assert np.all(ids == -1) and np.all(np.isneginf(scores))

    def test_wrong_dimension_raises(self, ivf):
        with pytest.raises(ValueError):
            ivf.search_batch(np.ones((3, D - 1), dtype=np.float32), 5)

    def test_1d_input_raises(self, ivf):
        with pytest.raises(ValueError):
            ivf.search_batch(np.ones(D, dtype=np.float32), 5)

    @pytest.mark.parametrize("k", [0, -1])
    def test_bad_k_raises(self, ivf, queries, k):
        with pytest.raises(ValueError):
            ivf.search_batch(queries, k)

    @pytest.mark.parametrize("nprobe", [0, -1])
    def test_bad_nprobe_raises(self, ivf, queries, nprobe):
        with pytest.raises(ValueError):
            ivf.search_batch(queries, 5, nprobe=nprobe)