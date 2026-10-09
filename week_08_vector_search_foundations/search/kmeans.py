import numpy as np
from search.brute_force import normalize

        
def kmeans(X, k, max_iter=25, seed = 0):
    
    rng = np.random.default_rng(seed=seed)
    history = []
    
    if k <= 0:
        raise ValueError("ERROR: k value must be greater than or equal to 1")
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("ERROR: k must be of type integer")
    if k > X.shape[0]:
        raise ValueError("ERROR: k cannot be larger than the training size")
        
    X_norm = normalize(X)
    
    centroids = initialize(X_norm, k, rng)
    labels = None
       
    for i in range(max_iter):
            
        sums = np.zeros((k, X.shape[-1]), dtype=np.float32)
            
        labels, best = assign(X_norm, centroids)
        
        history.append(np.mean(best))
            
            
        np.add.at(sums, labels, X_norm)
        counts = np.bincount(labels, minlength=k)
        upd = np.maximum(counts, 1)
        update = sums / upd[:, np.newaxis]
        update = normalize(update)
            
            
        if np.allclose(centroids, update):
            return centroids, labels, history
        
        centroids = update
        
        zeroed = np.where(counts == 0)[0]
            
        if len(zeroed) > 0:
            centroids = re_initialize(zeroed, centroids, X_norm, rng)
    
    return centroids, assign(normalize(X), centroids)[0] , history
            
def initialize(X, k, rng):
        
    centers = rng.choice(range(0, len(X)), k, replace=False)
        
    return X[centers].copy()
            
def assign(X, centroids, chunk_size=8192):
    
    labels = np.zeros(len(X), dtype=np.int64)
    best = np.zeros(len(X), dtype=np.float32)
    
    for start in range(0, len(X), chunk_size):
        
        Xc = X[start: start + chunk_size]
        scores = Xc @ centroids.T
        
        label = scores.argmax(axis=1)
        b = scores.max(axis=1)
        
        labels[start: start + len(label)] = label
        best[start: start + len(label)] = b
        
    return labels, best
    
def re_initialize(zeroed, centroids, X, rng):
        
    count = len(zeroed)
    updated = rng.choice(range(0, len(X)), count, replace=False)
        
    centroids[zeroed] = X[updated].copy()
    
    return centroids
        
        