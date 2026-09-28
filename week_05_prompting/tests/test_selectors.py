"""
AI Generated Tests
"""

import numpy as np
import pandas as pd
import pytest

from prompting.sampler.sample_generator import (
    capped_label,
    random_sampler,
    top_diverse,
    top_similarity,
)

# ---------------------------------------------------------------------------
# Hand-built pool (4-dim unit vectors). Query points along the first axis.
#
#   name  label                   sim(query)   notes
#   A     card_arrival            0.951        best match
#   B     card_arrival            0.942        near-duplicate of A   (A·B = 0.998)
#   C     card_delivery_estimate  0.799        relevant but different (A·C = 0.620)
#   D     exchange_rate           0.000        irrelevant
#   E     top_up_failed           0.300        weak
#
# MMR with lam=0.7, after picking A:
#   B: 0.7*0.942 - 0.3*0.998 = 0.360
#   C: 0.7*0.799 - 0.3*0.620 = 0.373   ← C wins, although B is closer to the query
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


QUERY_VEC = _unit([1, 0, 0, 0])
POOL = np.stack([
    _unit([0.95, 0.31, 0.00, 0.0]),   # A
    _unit([0.94, 0.33, 0.05, 0.0]),   # B
    _unit([0.80, -0.45, 0.40, 0.0]),  # C
    _unit([0.00, 0.00, 0.00, 1.0]),   # D
    _unit([0.30, -0.90, 0.30, 0.1]),  # E
])
NAMES = ["A", "B", "C", "D", "E"]
LABELS = ["card_arrival", "card_arrival", "card_delivery_estimate", "exchange_rate", "top_up_failed"]


class FakeModel:
    """Stands in for SentenceTransformer: returns a fixed vector for the query."""

    def __init__(self, vec=QUERY_VEC):
        self.vec = vec
        self.calls = 0

    def encode(self, text, normalize_embeddings=True):
        self.calls += 1
        return self.vec.copy()


@pytest.fixture
def data():
    # `pos` = row position, so tests can check which rows came back and in what order
    return pd.DataFrame({"pos": range(5), "name": NAMES, "label_name": LABELS})


@pytest.fixture
def E():
    return POOL.copy()


def names(rows):
    return list(rows["name"])


# All four selectors behind one signature, for the shared property tests below.
SELECTORS = {
    "random": lambda q, d, m, e, k: random_sampler(q, d, m, e, samples=k),
    "top_similarity": lambda q, d, m, e, k: top_similarity(q, d, m, e, samples=k),
    "top_diverse": lambda q, d, m, e, k: top_diverse(q, d, m, e, samples=k),
    "capped_label": lambda q, d, m, e, k: capped_label(q, d, m, e, "label_name", samples=k),
}


# ---------------------------------------------------------------------------
# 1. Properties every selector must have
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SELECTORS)
def test_returns_requested_number_of_rows(name, data, E):
    rows, emb, _ = SELECTORS[name]("q", data, FakeModel(), E, 3)
    assert len(rows) == 3 and len(emb) == 3


@pytest.mark.parametrize("name", SELECTORS)
def test_no_duplicate_rows(name, data, E):
    rows, _, _ = SELECTORS[name]("q", data, FakeModel(), E, 4)
    assert len(set(rows["pos"])) == len(rows)


@pytest.mark.parametrize("name", SELECTORS)
def test_returned_embeddings_line_up_with_returned_rows(name, data, E):
    """Row i of the returned DataFrame must be the example whose vector is row i of the returned E.
    If these get out of order, the heatmap and any downstream use of the vectors is silently wrong."""
    rows, emb, _ = SELECTORS[name]("q", data, FakeModel(), E, 4)
    expected = POOL[rows["pos"].to_numpy()]
    assert np.allclose(emb, expected), f"{name}: rows and embeddings are in different orders"


@pytest.mark.parametrize("name", SELECTORS)
def test_pool_embeddings_are_not_modified(name, data, E):
    before = E.copy()
    SELECTORS[name]("q", data, FakeModel(), E, 3)
    assert np.array_equal(E, before)


@pytest.mark.parametrize("name", SELECTORS)
def test_returns_the_query_embedding(name, data, E):
    _, _, q = SELECTORS[name]("q", data, FakeModel(), E, 3)
    assert np.allclose(q, QUERY_VEC)


@pytest.mark.parametrize("name", SELECTORS)
def test_samples_larger_than_pool_returns_whole_pool(name, data, E):
    rows, emb, _ = SELECTORS[name]("q", data, FakeModel(), E, 50)
    if name == "capped_label":
        assert 1 <= len(rows) <= 5          # the cap may legitimately return fewer
    else:
        assert sorted(rows["pos"]) == [0, 1, 2, 3, 4]
    assert len(emb) == len(rows)


@pytest.mark.parametrize("name", SELECTORS)
def test_samples_equal_to_pool_size(name, data, E):
    rows, _, _ = SELECTORS[name]("q", data, FakeModel(), E, 5)
    assert len(rows) <= 5 and len(set(rows["pos"])) == len(rows)


@pytest.mark.parametrize("name", SELECTORS)
def test_labels_travel_with_rows(name, data, E):
    rows, _, _ = SELECTORS[name]("q", data, FakeModel(), E, 3)
    for _, r in rows.iterrows():
        assert r["label_name"] == LABELS[r["pos"]]


# ---------------------------------------------------------------------------
# 2. random_sampler
# ---------------------------------------------------------------------------

def test_random_same_seed_same_rows(data, E):
    a, _, _ = random_sampler("q", data, FakeModel(), E, samples=3, seed=7)
    b, _, _ = random_sampler("q", data, FakeModel(), E, samples=3, seed=7)
    assert list(a["pos"]) == list(b["pos"])


def test_random_different_seeds_can_differ():
    big = pd.DataFrame({"pos": range(100), "name": [str(i) for i in range(100)],
                        "label_name": ["x"] * 100})
    E_big = np.eye(100)
    picks = {tuple(random_sampler("q", big, FakeModel(np.eye(100)[0]), E_big, samples=5, seed=s)[0]["pos"])
             for s in range(5)}
    assert len(picks) > 1


# ---------------------------------------------------------------------------
# 3. top_similarity
# ---------------------------------------------------------------------------

def test_top_similarity_picks_highest_scores(data, E):
    rows, _, _ = top_similarity("q", data, FakeModel(), E, samples=3)
    assert set(names(rows)) == {"A", "B", "C"}


def test_top_similarity_most_similar_is_last(data, E):
    rows, _, _ = top_similarity("q", data, FakeModel(), E, samples=3)
    assert names(rows) == ["C", "B", "A"]          # ascending similarity


def test_top_similarity_k1(data, E):
    rows, _, _ = top_similarity("q", data, FakeModel(), E, samples=1)
    assert names(rows) == ["A"]


def test_top_similarity_full_pool_ordering(data, E):
    rows, _, _ = top_similarity("q", data, FakeModel(), E, samples=5)
    assert names(rows) == ["D", "E", "C", "B", "A"]


# ---------------------------------------------------------------------------
# 4. top_diverse (MMR)
# ---------------------------------------------------------------------------

def test_mmr_first_pick_is_most_similar(data, E):
    rows, _, _ = top_diverse("q", data, FakeModel(), E, samples=1)
    assert names(rows) == ["A"]


def test_mmr_skips_near_duplicate(data, E):
    """Hand-computed above: after A, C (0.373) beats the near-duplicate B (0.360)."""
    rows, _, _ = top_diverse("q", data, FakeModel(), E, lam=0.7, samples=2)
    assert set(names(rows)) == {"A", "C"}
    top_rows, _, _ = top_similarity("q", data, FakeModel(), E, samples=2)
    assert set(names(top_rows)) == {"A", "B"}      # plain top-k would have taken B


def test_mmr_most_similar_is_last(data, E):
    rows, _, _ = top_diverse("q", data, FakeModel(), E, lam=0.7, samples=2)
    assert names(rows)[-1] == "A"


def test_mmr_lambda_one_equals_top_k(data, E):
    mmr_rows, _, _ = top_diverse("q", data, FakeModel(), E, lam=1.0, samples=3)
    top_rows, _, _ = top_similarity("q", data, FakeModel(), E, samples=3)
    assert names(mmr_rows) == names(top_rows)


# ---------------------------------------------------------------------------
# 5. capped_label
# ---------------------------------------------------------------------------

def test_capped_respects_cap(data, E):
    # samples=2, cap=0.5 → max 1 per label → A taken, B (same label) skipped, C taken
    rows, _, _ = capped_label("q", data, FakeModel(), E, "label_name", cap=0.5, samples=2)
    assert set(names(rows)) == {"A", "C"}
    assert rows["label_name"].value_counts().max() == 1


def test_capped_most_similar_is_last(data, E):
    rows, _, _ = capped_label("q", data, FakeModel(), E, "label_name", cap=0.5, samples=2)
    assert names(rows)[-1] == "A"


def test_capped_k1_returns_one_row(data, E):
    rows, _, _ = capped_label("q", data, FakeModel(), E, "label_name", cap=0.5, samples=1)
    assert names(rows) == ["A"]


def test_capped_count_never_exceeds_max_reps(data, E):
    rows, _, _ = capped_label("q", data, FakeModel(), E, "label_name", cap=0.5, samples=4)
    max_reps = max(1, int(4 * 0.5))
    assert rows["label_name"].value_counts().max() <= max_reps


def test_capped_single_label_pool_returns_fewer_without_crashing(E):
    same = pd.DataFrame({"pos": range(5), "name": NAMES, "label_name": ["x"] * 5})
    rows, emb, _ = capped_label("q", same, FakeModel(), E, "label_name", cap=0.5, samples=3)
    assert len(rows) == 1 and len(emb) == 1        # max 1 per label, only one label exists


def test_capped_missing_label_column_raises(data, E):
    with pytest.raises(ValueError):
        capped_label("q", data, FakeModel(), E, "not_a_column", samples=2)


# ---------------------------------------------------------------------------
# 6. Model usage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SELECTORS)
def test_query_encoded_once_per_call(name, data, E):
    model = FakeModel()
    SELECTORS[name]("q", data, model, E, 3)
    assert model.calls == 1