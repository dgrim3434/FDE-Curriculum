import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"
BANKING_NAME = "banking77"


def random_sampler(query, data, model, E, samples=5, seed=42):
    
    embedding = model.encode(query, normalize_embeddings=True)
    
    rng = np.random.default_rng(seed=seed)
    
    idx = rng.choice(len(data), size = min(samples, len(data)), replace=False)
    
    return data.iloc[idx], E[idx], embedding


def top_similarity(query, data, model, E, samples=5):
    
    embedding = model.encode(query, normalize_embeddings=True)
    
    scores = E @ embedding
    
    samples = min(samples, len(E))
    
    if samples < len(E):
        top = np.argpartition(-scores, samples)[:samples]
    
        sort = top[np.argsort(-scores[top])]
    else:
        sort = np.argsort(-scores)[:samples]
    
    idx = np.flip(sort)
    
    return data.iloc[idx], E[idx], embedding

def top_diverse(query, data, model, E, lam =0.7 , samples=5):
    
    embedding = model.encode(query, normalize_embeddings=True)
    idx = []
    # Number of embeddings which have been selected
    count = 0
    
    scores = E @ embedding
    
    loops = min(samples, len(E))
    
    while count < loops:
        
        if count == 0:
            
            best = int(np.argmax(scores))
            idx.append(best)
        else:
            redundancy = (E @ E[idx].T).max(axis = 1)
            
            upd_score = lam * scores - redundancy * (1 - lam)
            
            upd_score[idx] = float("-inf")
            
            best = int(np.argmax(upd_score))
            idx.append(best)
                
        count += 1
    
    results = np.flip(idx)
    return data.iloc[results], E[results], embedding

"""

"""
def capped_label(query, data, model, E, label_column, cap = 0.5, samples=5):
    
    embedding = model.encode(query, normalize_embeddings=True)
    
    if label_column not in set(data.columns):
        raise ValueError("ERROR: Label Column Must be present within the dataframe")
    
    labels = data[label_column].to_numpy()
    
    samples = min(samples, len(E))
    
    max_reps = max(1, int(samples * cap))
    
    scores = E @ embedding
    counts = {}
    
    order = np.argsort(-scores)
    idx = []
    
    for i in order:
        
        label = labels[i]
        
        if label not in counts:
            
            counts[label] = 0
        
        if counts[label] < max_reps:
            counts[label] += 1
            idx.append(int(i))
        
        if len(idx) >= samples:
            break
    
    idx = np.flip(idx)
    
    return data.iloc[idx], E[idx], embedding
            
        
            
            
        
    
    
    