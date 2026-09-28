"""
Single Head Attention using the torch.nn.module class for all linear transformations:
    - X @ Q
    - X(Context) @ K
    - X(Context) @ V
    These transformations are simple and I am learning py torch. These results are then passed into the dot product attention function which returns residuals and weights

"""
import torch
from torch import nn
from attention.functional import scaled_dot_product_attention

class Single_Head_Attention(nn.Module):
    
    def __init__(self, d_model, d_k = None, bias = True):
        super().__init__()
        d_k = d_model if d_k is None else d_k
        
        # Defining Linear transformations
        self.W_q = nn.Linear(d_model, d_k, bias=bias)
        self.W_k = nn.Linear(d_model, d_k, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
    
    def forward(self, x, context = None, mask = None):
        
        context = x if context is None else context
        
        return (scaled_dot_product_attention(self.W_q(x), self.W_k(context), self.W_v(context), mask=mask, single_head=True))