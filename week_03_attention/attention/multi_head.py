"""
Multi Head Attention
"""

import torch
import torch.nn as nn
from attention.functional import scaled_dot_product_attention


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
            