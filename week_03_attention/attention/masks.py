"""
Handles both the causal and padding masks.

These masks are used on result of the Q @ K matrix before being passed into the softmax function. This order is important because the mask sets the positions to -inf which 
the softmax function then converts into 0.

Causal mask: Ensures tokens only see previous tokens. The transformer is trained with T total predictions for sequence length T. Without a causal mask the token can use the next
token within it's calculations which will cause the loss to drop but will result in meaningless weights due to the target leakage.

Padding: In order for the calculations to be done in parallel all batchs must have the same sequence length. In order to fit smaller sequences into the training set you must add
padding to the extra positions.
"""

import torch


def causal_mask(d_q, d_k = None, device = None):
    
    # If d_k is none it's self-attention
    d_k = d_q if d_k is None else d_k
    
    return torch.tril(torch.ones(d_q, d_k, dtype=torch.bool, device=device), diagonal= d_k - d_q)

def padding_mask_to_attention(mask):
    
    return mask[:, None, None, :]

