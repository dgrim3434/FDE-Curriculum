from model.attention import Multi_Head_Attention
from model.ffn import Forward_Feed_Network

import torch.nn as nn

class Transformer_Block(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff = None, bias=True, dropout = 0.0, norm='pre'):
        
        super().__init__()
        
        if norm not in set(['pre', 'post']):
            raise ValueError("ERROR: Invalid input for parameter: norm. Must be either 'pre' or 'post' ")
        
        self.norm = norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.attention = Multi_Head_Attention(d_model, num_heads, dropout, bias)
        self.ffn = Forward_Feed_Network(d_model, d_ff, dropout, bias)
    
    def forward(self, x, mask = None, return_att = False):
        
        if self.norm == 'pre':
            
            y, w = self.attention.forward(self.ln1(x), mask=mask)
            x = x + y
            
            x = x + self.ffn.forward(self.ln2(x))
        else:
            
            y, w = self.attention.forward(x, mask = mask)
            x = self.ln1(x + y)
            
            x = self.ln2(x + self.ffn.forward(x))
        
        if return_att:
            return (x, w)
        
        return (x, None)