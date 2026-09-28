import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"


def embed_pool(data, text_column: str, model_name: str = MODEL_NAME):

    if text_column not in data.columns:
        raise ValueError(f"column {text_column!r} not in data (have {list(data.columns)})")

    model = SentenceTransformer(model_name) 

    texts = data[text_column].astype(str).tolist()
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{model_name.replace('/', '_')}_{len(texts)}_{digest}.npy"

    if cache_path.exists():
        return np.load(cache_path), model

    E = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    np.save(cache_path, E)
    return E, model


def embed_queries(data, data_name, query_column, model_name=MODEL_NAME):

    return embed_pool(data, query_column, model_name)