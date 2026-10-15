import numpy as np
import os
from metadata.store import load_vectors, save_vectors, check_aligned

def make_embed_fn(model, prefix="", batch_size=64):
    
    def embed(texts):
        
        return model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    
    return embed

def texts_for_embedding(chunks) -> list[str]:
    
    texts = []
    
    for chunk in chunks:
        
        if 'embed_text' in chunk.metadata:
            
            texts.append(chunk.metadata['embed_text'])
        else:
            
            texts.append(chunk.text)
    
    return texts

def embed_with_cache(chunks, embed_fn, cache_path) -> np.ndarray:
    
    if os.path.isfile(cache_path):
        
        vectors, chunk_ids = load_vectors(cache_path)

        check_aligned(chunks, chunk_ids)
    else:
        
        vectors = embed_fn(texts_for_embedding(chunks))
        
        save_vectors(cache_path, vectors, [c.chunk_id for c in chunks])
    
    return vectors

def doc_layout(chunks) -> tuple[list[str], np.ndarray]:
    
    if len(chunks) == 0:
        return [], np.array([])
    
    seen = set()
    doc_ids = []
    doc_starts = []
    
    curr_doc = chunks[0].doc_id
    seen.add(curr_doc)
    doc_ids.append(curr_doc)
    doc_starts.append(0)
    
    for i in range(1, len(chunks)):
        
        chunk = chunks[i]
        if chunk.doc_id == curr_doc:
            continue
        
        if chunk.doc_id in seen:
            raise ValueError("ERROR: Chunks are not contiguos")
        
        curr_doc = chunk.doc_id
        seen.add(curr_doc)
        doc_starts.append(i)
        doc_ids.append(curr_doc)
        
        i += 1
        
    return doc_ids, np.array(doc_starts)

def maxp(S, doc_starts) -> np.ndarray:
    
    return np.maximum.reduceat(S, doc_starts, axis=1)

def rank_documents(Q, C, chunks, query_ids, k) -> dict:
    
    S = Q @ C.T
    
    doc_ids, doc_starts = doc_layout(chunks)
    
    D = maxp(S, doc_starts)

    order = np.argsort(-D, axis=1, kind='stable')[:, :k]
    
    return {query_ids[i]: [doc_ids[j] for j in order[i]] for i in range(len(query_ids))}

def rank_chunks(q_vec, C_paper) -> list[int]:
    
    S = C_paper @ q_vec
    
    return list(np.argsort(-S, kind='stable'))
