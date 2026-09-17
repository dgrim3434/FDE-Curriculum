"""
Creates the sinusiodal positional encoding matrix:
    - Takes two inputs:
        - dims: The length of each embedding which must be an even number
        - max_len: The maximum sequence length for the model. So the maximum number of tokens which can be ingested at a single time
    
    The result is a matrix of shape (max_len, dims)

    this positional matrix is than added to each embedding depending on it's position within the sequence. This creates a rotation which allows the model to learn position within a 
    sequence. 
"""

import numpy as np

def sinusiodal_position_encodings(max_len, dims):
    
    # Check to make sure the dimensions are even
    assert dims % 2 == 0
    
    # Create a placeholder position matrix defualted to zeros
    pe = np.zeros((max_len, dims), dtype=np.float32)
    
    # create the positions
    pos = np.arange(max_len, dtype=np.float32)[:, None]
    
    # create the pairs (2i, 2i+ 1)
    twoi = np.arange(0, dims, 2)
    
    # calcuate the inverse of the denominator for i pair
    dems = np.exp(twoi * (-np.log(10000) / dims))
    
    # Generating the positional encoding for the even indices (sin)
    pe[:, 0::2] = np.sin(pos * dems)
    pe[:, 1::2] = np.cos(pos * dems)
    
    return pe
    
    
    