import numpy as np

def normalize(matrix, dims=-1):
    
    norms = np.linalg.norm(matrix, axis=dims, keepdims=True)
    
    return(matrix / np.maximum(norms, 1e-12)).astype(np.float32)
    
def top_k(scores, k):
    
    k = min(len(scores), k)
    
    top_k = np.argpartition(-scores, kth=k-1)[:k]
    
    results = np.argsort(-scores[top_k])
    
    return top_k[results], scores[top_k[results]]

def top_k_batch(S, k):
    
    k = min(S.shape[-1], k)
    
    top_k = np.argpartition(-S, kth=k-1, axis=-1)[:, :k]
    #breakpoint()
    top = np.take_along_axis(S, top_k)
    
    results = np.argsort(-top, axis=-1)
    
    indicies = np.take_along_axis(top_k, results)
    return indicies, np.take_along_axis(S, indicies)

class BruteForceIndex:
    
    def __init__(self, vectors):
        
        if vectors.ndim != 2:
            raise ValueError('Embedding Matrix must 2D')
        
        self.N = vectors.shape[0]
        self.d = vectors.shape[1]
        
        self.E = normalize(vectors)
    
    def search(self, q, k):
        
        if q.ndim != 1 or q.shape[0] != self.d:
            
            if q.ndim != 1:
                raise ValueError(f"ERROR: The query matrix must be 1D. Detected Dimensions where: {q.ndim}")
            raise ValueError(f"ERROR: The dimensions of the query matrix must match the embedding dimensions. Got: {q.shape[0]}, expected: {self.d}")
        
        if k <= 0:
            raise ValueError(f"ERROR: k must be atleast 1. Received k = {k}")
        
        scores = self.E @ normalize(q)
        
        return top_k(scores, k)
    
    def search_batch(self, Q, k, chunk_size=256):
        
        if Q.ndim != 2 or Q.shape[-1] != self.d:
            
            if Q.ndim != 2:
                raise ValueError(f"Error: The query matrix must be 2d. Detected Dimensions where: {Q.ndim}")
            
            raise ValueError(f"ERROR: The dimensions of the query matrix must match the embedding dimensions. Got: {Q.shape[0]}, expected: {self.d}")
        
        B = Q.shape[0]
        
        if k <= 0:
            raise ValueError(f"ERROR: k must be atleast 1. Received k = {k}")
        
        k = min(k, self.N)

        out_ids = np.zeros((B, k), dtype=np.int64)
        out_values = np.zeros((B, k), dtype=np.float32)
        
        for start in range(0, B, chunk_size):
            
            Qc = Q[start: start + chunk_size]
            
            # Shape is (chunk_size, D) @ (N, D)
            scores = normalize(Qc) @ self.E.T
            
            idx, values = top_k_batch(scores, k)
            
            out_ids[start: start + len(idx)] = idx
            out_values[start: start + len(values)] = values
        
        return out_ids, out_values