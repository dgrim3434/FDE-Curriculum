from search.kmeans import kmeans, assign
from search.brute_force import normalize, top_k, top_k_batch
import numpy as np

class IVFIndex:
    
    def __init__(self, vectors, nlist, train_size=None, max_iter=25, seed=0):
        
        self.E = normalize(vectors)
        self.N = vectors.shape[0]
        self.d = vectors.shape[1]
        
        if nlist < 1:
            raise ValueError("ERROR: The nlist parameter must be atleast 1")
        
        self.nlist = nlist
        
        self.max_iter = max_iter
        self.labels = None
        self.lists = np.zeros((self.nlist, ))
        self.rng = np.random.default_rng(seed=seed)
        
        if train_size is not None and train_size < 1:
            raise ValueError("ERROR: train_size must be a positive integer")
        
        
        if train_size is None:
            centroids, _, history = kmeans(self.E, self.nlist, max_iter=self.max_iter, seed=seed)
            
            labels, best = assign(self.E, centroids)
        else:
            
            train_size = min(self.N, train_size)
            selected = self.rng.choice(self.N, train_size, replace=False)
            
            centroids, _, history = kmeans(self.E[selected], self.nlist, max_iter=self.max_iter, seed=seed)
            
            labels, best = assign(self.E, centroids)
            
        self.centroids = centroids
        self.labels = labels
        self.history = history
        
        self.train_size = self.N if train_size is None else train_size
        
        order = np.argsort(self.labels, kind='stable')
        counts = np.bincount(self.labels, minlength=self.nlist)
        self.lists = np.split(order, np.cumsum(counts)[:-1])
    
    def cluster_sizes(self):
        
        return np.bincount(self.labels, minlength=self.nlist)
    
    def search(self, q, k, nprobe=1):
        
        # q shape -> (d), centroids shape: (n_probe, d)
        if q.ndim > 1:
            raise ValueError("ERROR: the search method only handles 1D vectors")
        if q.shape[-1] != self.d:
            raise ValueError("ERROR: q must match the dimensions of other embeddings")
        if nprobe < 1:
            raise ValueError("ERROR: nprobe must be atleast 1")
        if k < 1:
            raise ValueError("ERROR: k value must be atleast 1")
        
        q_norm = normalize(q)
        
        centroid_scores = self.centroids @ q_norm
        
        n_probe, _ = top_k(centroid_scores, k=nprobe)
        
        candidates = np.concatenate([self.lists[c] for c in n_probe])
        
        if len(candidates) < 1:
            return [], []
        
        cand_scores = self.E[candidates] @ q_norm
        
        pos, vals = top_k(cand_scores, k=k)
        
        return candidates[pos], vals
    
    def search_batch(self, Q, k, nprobe=1):
        
        if Q.ndim != 2:
            raise ValueError("ERROR: Q matrix must be 2 dims")
        if Q.shape[-1] != self.d:
            raise ValueError("ERROR: Q matrix must match the shape of the training data")
        if nprobe < 1:
            raise ValueError("ERROR: nprobe must be atleast 1")
        if k < 1:
            raise ValueError("ERROR: K value must be atleast 1")
        
        B = Q.shape[0]
        k = min(k, self.N)
        
        q_norm = normalize(Q)
        
        # Q shape is (B, D), centroids is (C, D) -> (B, C)

        scores = q_norm @ self.centroids.T
        
        n_probes, _ = top_k_batch(scores, k=nprobe)
        
        out_ids = np.full((B, k), -1, dtype=np.int64)
        out_scores = np.full((B,k), -np.inf, dtype=np.float32)
        
        for i in range(B):
            
            cand = np.concatenate([self.lists[c] for c in n_probes[i]])
            
            if len(cand) == 0:
                continue
            
            cand_scores = self.E[cand] @ q_norm[i]
            
            pos, vals = top_k(cand_scores, k=k)
            
            m = len(pos)
            
            out_ids[i, :m] = cand[pos]
            out_scores[i, :m] = vals

        return out_ids, out_scores