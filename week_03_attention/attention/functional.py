"""
Scaled dot product attention:
    - Where all of the attention math happens
    - 3 matrices Query (Q), Key (K), Value (V)
    - An input sequence of size (B, T, D): Batch_size, Sequence length, dimensions (length of embeddings)
    - Supports both self and cross attention
    
"""
import torch
from attention.softmax import stable_softmax
import math

def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None, single_head = False):
    
    d_q = Q.shape[-1]
    # Shape of Q matrix: B,H, T_q, D
    # Shape of K matrix: B, H, T_k, D
    # Result shape of q_k: B, H, T_q, T_k
    if single_head:
        scores = torch.einsum('BQD, BKD -> BQK', Q, K) / math.sqrt(d_q)
    else:
        scores = torch.einsum('BHQD, BHKD -> BHQK', Q, K) / math.sqrt(d_q)
    # Perform masking if not non. The mask must be True -> keep. We negate the mask
    
    if mask is not None:
        
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax after masking. Result is the weights
    weights = stable_softmax(scores,dims =-1)
    
    # Apply dropout if not None
    if dropout is not None:
        
        weights = dropout(weights)
    
    return weights @ V, weights

