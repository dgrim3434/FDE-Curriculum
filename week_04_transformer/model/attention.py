import torch
import math
import torch.nn as nn


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

import torch


def causal_mask(d_q, d_k = None, device = None):
    
    # If d_k is none it's self-attention
    d_k = d_q if d_k is None else d_k
    
    return torch.tril(torch.ones(d_q, d_k, dtype=torch.bool, device=device), diagonal= d_k - d_q)

def padding_mask_to_attention(mask):
    
    return mask[:, None, None, :]




class Multi_Head_Attention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout = 0.0, bias=True):
        super().__init__()
        if d_model % num_heads != 0:
            
            raise ValueError("ERROR: The model dimensions must be divisible by the number of heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.shapes = None
        self.att_dropout = nn.Dropout(dropout)
    
    # Turns an BTD shape matrix into a BHTd_k
    def _split(self, matrix):
        
        B, T, _ = matrix.shape
        return matrix.reshape(B, T, self.num_heads, self.d_k).transpose(1,2)
    
    # Turns a BHTd_k matrix into shape BTD
    def _merge(self, matrix):
        
        B, _, T, _ = matrix.shape
        return matrix.transpose(1, 2).reshape(B,T, -1)
    
    def forward(self, x, context = None, mask=None, record_shapes = False):
        
        context = x if context is None else context
        
        Q = self._split(self.W_q(x))
        K = self._split(self.W_k(context))
        V = self._split(self.W_v(context))
        
        if mask is not None and mask.dim() == 2:
            mask = mask[None, None, :, :]
            
        out, weights = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.att_dropout)
        
        merged = self._merge(out)
        
        y = self.W_o(merged)
        
        if record_shapes:
            
            self.shapes = {'x': x.shape, 'q': self.W_q(x).shape, 'Q': Q.shape, 'K': K.shape, 'scores': weights.shape, 'out_heads': out.shape, 'merged': merged.shape, 'y': y.shape}
        
        return y, weights

def stable_softmax(scores, dims=-1):
    
    # Extracting the row max to ensure no overflow
    row_max = torch.amax(scores, dim=dims, keepdim=True)
    # Converting -inf into 0 to guard against -inf - -inf -> NaN
    row_max = torch.where(torch.isneginf(row_max), torch.zeros_like(row_max), row_max)
    
    # Calculating the exponential function
    exp = torch.exp(scores - row_max)
    
    # Calculating denominatior summing across all columns
    denom = exp.sum(dim=dims, keepdim=True)
    
    return exp / denom.clamp_min_(torch.finfo(exp.dtype).tiny)