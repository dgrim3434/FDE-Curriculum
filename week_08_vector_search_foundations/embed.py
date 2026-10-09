from sentence_transformers import SentenceTransformer
import numpy as np
import time
import torch
from data_loading import load_corpus, load_queries,load_qrels
from data.constants import quora_path
import os
import json

def embed_texts(model_name, texts, prefix="", batch_size=256):
    
    model = SentenceTransformer(model_name, device="cuda")
    model_dim = model.get_sentence_embedding_dimension()
    
    info = {'model_name': model_name, 'dim': int(model_dim), 'max_length': int(model.max_seq_length), 'prefix': prefix, 'batch_size': int(batch_size), 'N': int(len(texts))}
    

    text_p = [prefix + t for t in texts]
    lengths = [len(ids) for ids in model.tokenizer(text_p, truncation=False)['input_ids']]
    #throw away warmup
    emb = model.encode(text_p[:2 * batch_size], batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,show_progress_bar=False)
    
    t0 = time.perf_counter()
    emb = model.encode(text_p, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,show_progress_bar=True)
    t1 = time.perf_counter() - t0
    
    
    info['seconds'] = float(t1)
    info['passages_per_sec'] = float(len(texts) / t1)
    info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    info['mean_tokens'] = np.mean(lengths)
    info['truncation_rate'] = (len([t for t in lengths if t > model.max_seq_length]) / len(lengths)) * 100
    info['tokens_per_sec'] = float(sum([min(t ,model.max_seq_length) for t in lengths]) / t1)
    
    
    return emb, info

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    
    corpus = load_corpus(quora_path / "corpus.jsonl")
    query = load_queries(quora_path / "queries.jsonl")
    
    corpus_ids = [c[0] for c in corpus]
    corpus_text = [c[1] for c in corpus]
    
    test_ids   = sorted(load_qrels(quora_path / "qrels" / "test.tsv").keys())
    
    query_ids = test_ids
    query_text = query_text = [query[qid] for qid in test_ids]
    
    os.makedirs("data/embeddings", exist_ok=True)
    
    with open("data/embeddings/quora_minilm_corpus_ids.json", "w", encoding="utf-8") as f:
        json.dump(corpus_ids, f) 
    
    with open("data/embeddings/quora_minilm_query_ids.json", "w", encoding="utf-8") as f:
            json.dump(query_ids, f)
    
    
    corpus_embeddings, corpus_info = embed_texts(MODEL_NAME, corpus_text)
    query_embeddings, query_info = embed_texts(MODEL_NAME, query_text)
    
    np.save("data/embeddings/quora_minilm_corpus.npy", corpus_embeddings)
    np.save("data/embeddings/quora_minilm_queries.npy", query_embeddings)
    
    with open("results/embed_quora_minilm.json", 'w', encoding='utf-8') as f:
        
        json.dump({'corpus': corpus_info, 'query': query_info}, f, indent=2)
    
"""
np.save("data/embeddings/quora_minilm_corpus.npy", corpus_emb)    # (522931, 384)
np.save("data/embeddings/quora_minilm_queries.npy", query_emb)
# + quora_minilm_corpus_ids.json, quora_minilm_query_ids.json (lists of str, row order)
# + quora_minilm_meta.json (the info dicts)

"""
