from dataclasses import asdict
from chunking.base import Chunk
import json
import numpy as np

def save_chunks(chunks, path):
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
    
        for chunk in chunks:
        
            record = asdict(chunk)
            record['chunk_id'] = chunk.chunk_id
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

def load_chunks(path):
    
    chunks = []
    
    with open(path, 'r', encoding='utf-8') as f:
        
        for line in f:
            
            record = json.loads(line)
            record.pop("chunk_id", None)
            chunks.append(Chunk(**record))
    
    return chunks

def save_vectors(path, vectors, chunk_ids):
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(path, vectors=vectors, chunk_ids=np.array(chunk_ids))

def load_vectors(path):
    
    data = np.load(path)
    
    return data['vectors'], data['chunk_ids'].tolist()

def check_aligned(chunks, chunk_ids) -> None:
    
    ids = [c.chunk_id for c in chunks]
    
    if ids != list(chunk_ids):
        raise ValueError("ERROR: indicies are misaligned")