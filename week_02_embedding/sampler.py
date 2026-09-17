"""
The Sampler File:
Responsible for both generating the pairs and the negative sampling.

Pair Generation:
    - Uses the random number generator to generate a number between 1 and the max window length. 
        - This allows for closer words to recieve higher weighting naturally since they are more likely to be within the window compared to words which are futher from the center
    - When selecting the max window size there are a couple of important things to understand. If you want to keep the embeddings extremely tight which means the words capture only extremely close 
    semantics then a smaller window is best assuming you are doing subsampling. The reasoning for this is the larger the window size the more clouded the embedding becomes since more words are being 
    factored in which means the embedding will not be as definite. So if you are doing embeddings for fact retreival you might want a smaller max_window than if you are focused more on the overall
    sentence quality. Smaller corpuses also require larger max_windows because we must ensure each word is trained enough to ensure the semantics are properly extracted. The pairs should be generated
    per epoch which also means that each center word is not gaurenteed to see the same set of words each time which is also good.
"""
import numpy as np
from collections import Counter


class Sampler:
    
    """
    Since Vocab Creation has already been completed before the sampler class is used we will re-use the frequency which was
    computed within the class
    """
    def __init__(self, frequencies, seed = 0):
        
        self.frequencies = frequencies
        self.rng = np.random.default_rng(seed)
        self.probabilities = None
        self.cdf = None
    """
    The pairs get computed for each epoch and subsampling is done for each epoch to.
    Since computing the pairs relies on the results from the subsampling it must take the ids from the current epoch as an input when
    computing the pairs.
    """
    # Function updated to preallocate. Used AI to update implementation (Much more efficient memory compared to regulare python lists)
    def compute_pairs(self, ids, max_window = 5, rng = None):
        
        rng = rng or np.random.default_rng()
        
        n = len(ids)
        
        num_pairs = rng.integers(1, max_window + 1, size=n)
        cap = n * 2 * max_window
        centers = np.empty(cap, np.int32)
        contexts = np.empty(cap, np.int32)
        
        pos_pointer = 0
        for i in range(n):
            
            b_low = np.maximum(0, i - num_pairs[i])
            b_high = np.minimum(n, i + num_pairs[i] + 1)
            
            for j in range(b_low, b_high):
                # Skipping over the center index we don't want center as it's own context
                if j == i:
                    continue
                
                centers[pos_pointer] = ids[i]
                contexts[pos_pointer] = ids[j]
                pos_pointer += 1
        
        return centers[:pos_pointer], contexts[:pos_pointer]

    """
    PreComputes all of the statistics for the data so that sampling can be done quickly when needed.
    The negative sampling is called for each id to conserve the memory so We must compute all needed dataset wide statistics 
    which should be done once instead of everytime we need negative samples. The negative sampling is also not relient on 
    """
    def negative_sampling(self, power = .75):
        
        p = self.frequencies ** power
        self.probabilities = p / sum(p)
        self.cdf = np.cumsum(self.probabilities)
        self.cdf[-1] = 1
    
    def negative_draw(self, batch_size, k):
        
        # if self.cdf == None or self.probabilities == None:
        #     raise ValueError("ERROR: negative sampling function must be called before negative drawl")
        
        u = self.rng.random((batch_size, k))
        return np.searchsorted(self.cdf, u).astype(int)
        
        