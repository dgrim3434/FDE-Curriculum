import numpy as np


def _encode(query, model, query_vec):
    return query_vec if query_vec is not None else model.encode(query, normalize_embeddings=True)


def _check_k(samples: int, E) -> int:
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    return min(samples, len(E))


def _pack(data, E, order, q):
    idx = np.asarray(order, dtype=int)
    return data.iloc[idx], E[idx], q


def random_sampler(query, data, model, E, samples=5, seed=42, query_vec=None):
    q = _encode(query, model, query_vec)
    k = _check_k(samples, E)
    idx = np.random.default_rng(seed).choice(len(E), size=k, replace=False)
    return _pack(data, E, idx, q)


def top_similarity(query, data, model, E, samples=5, query_vec=None):
    q = _encode(query, model, query_vec)
    k = _check_k(samples, E)
    scores = E @ q
    top = np.argpartition(-scores, k - 1)[:k] if k < len(E) else np.arange(len(E))
    best_first = top[np.argsort(-scores[top])]
    return _pack(data, E, best_first[::-1], q)


def top_diverse(query, data, model, E, lam=0.7, samples=5, query_vec=None):
    
    q = _encode(query, model, query_vec)
    k = _check_k(samples, E)
    relevance = E @ q
    max_sim = np.full(len(E), -np.inf)     
    picked = []
    for _ in range(k):
        score = relevance if not picked else lam * relevance - (1 - lam) * max_sim
        score = score.copy()
        score[picked] = -np.inf
        best = int(np.argmax(score))
        picked.append(best)
        max_sim = np.maximum(max_sim, E @ E[best])
    return _pack(data, E, picked[::-1], q)


def capped_label(query, data, model, E, label_column, cap=0.5, samples=5, query_vec=None):
    if label_column not in data.columns:
        raise ValueError(f"label column {label_column!r} not in data")
    q = _encode(query, model, query_vec)
    k = _check_k(samples, E)
    max_reps = max(1, int(k * cap))
    labels = data[label_column].to_numpy()

    counts: dict = {}
    picked = []
    for i in np.argsort(-(E @ q)):
        lab = labels[i]
        if counts.get(lab, 0) < max_reps:
            counts[lab] = counts.get(lab, 0) + 1
            picked.append(int(i))
            if len(picked) == k:
                break
    return _pack(data, E, picked[::-1], q)