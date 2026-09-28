from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
MODEL_NAME = "all-MiniLM-L6-v2"


CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"

def embed_queries(data, data_name, query_column, model_name=MODEL_NAME):
    
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        raise ValueError("ERROR: Invalid model name input")
    
    
    norm_model_name = model_name.replace("/", "_")
    
    CACHE_PATH = CACHE_DIR / f"{norm_model_name}_{data_name}.npy"
    
    if CACHE_PATH.exists():
        E = np.load(CACHE_PATH)
        
        if E.shape[0] != len(data):
            raise ValueError(f"ERROR: cache matrix is size: {E.shape[0]} and provided data is size: {len(data)}")
        
        return E, model
    
    queries = data[query_column].tolist()
    
    E = model.encode(
        queries,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    )
    
    np.save(CACHE_PATH, E)
    
    return E, model
