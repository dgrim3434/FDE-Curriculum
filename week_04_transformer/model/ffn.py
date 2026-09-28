"""
The forward feed network class is responsible for the non-linear transformations which occur within the transformer architecture. There are three main peices of the class the
is the trasformation which scales the input to dimensions d_ff. The goal of this is to increase the dimensionality in which non linear transformations can be learned. The resuls
of this transformation are then passed into the GeLU activation function which is responsible for the actual non-linear transformation. The result of this is then passed into a final
transformation which converts the dimensions back to d_model which is then added to the residual stream. This is where most of the parameters within the transformer live with each 
transformate using d_model x d_ff learned parameters plus d_ff dimension bias term fro the upscale transformation and a d_model bias term for the downscale tranformation. Since
d_ff defaults to 4 x d_model the total parameters within each sublayer is 8d_model^2 + 5d_model which accounts for about 2/3 of the total parameters within the transformer depending 
on the total amount of layers.
"""


import torch.nn as nn

class Forward_Feed_Network(nn.Module):
    
    def __init__(self, d_model, d_ff = None, dropout = 0.0, bias=True):
        
        super().__init__()
        d_ff = d_model * 4 if d_ff is None else d_ff
        
        self.fn_u = nn.Linear(d_model, d_ff, bias=bias)
        self.fn_l = nn.Linear(d_ff, d_model, bias=bias)
        self.activation = nn.GELU(approximate='tanh')
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        return self.dropout(self.fn_l(self.activation(self.fn_u(x))))