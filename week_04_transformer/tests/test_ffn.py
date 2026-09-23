import pytest
import torch
import torch.nn as nn
from model.ffn import Forward_Feed_Network as FeedForward
D, T, B = 64, 12, 2

def test_shapes_and_default_expansion():
    ffn = FeedForward(D)
    assert ffn(torch.randn(B, T, D)).shape == (B, T, D)
    assert ffn.fn_u.weight.shape == (4 * D, D)       # nn.Linear stores (out, in)
    assert ffn.fn_l.weight.shape == (D, 4 * D)
 
 
def test_parameter_count():
    d_ff = 4 * D
    ffn = FeedForward(D)
    n = sum(p.numel() for p in ffn.parameters())
    assert n == D * d_ff + d_ff + d_ff * D + D
    assert sum(p.numel() for p in FeedForward(D, d_ff=128).parameters()) == D * 128 + 128 + 128 * D + D
 
 
def test_is_position_wise():
    """Editing one position must change NO other position, exactly.
    .eval() is mandatory: in train mode dropout draws a new mask per call."""
    ffn = FeedForward(D, dropout=0.1).eval()
    x = torch.randn(1, T, D)
    y = ffn(x)
    x2 = x.clone(); x2[0, 5] = torch.randn(D)
    y2 = ffn(x2)
    keep = [i for i in range(T) if i != 5]
    assert torch.equal(y[0, keep], y2[0, keep])       # exactly unchanged
    assert not torch.equal(y[0, 5], y2[0, 5])         # negative control
 
 
def test_permutation_equivariance():
    ffn = FeedForward(D).eval()
    x = torch.randn(1, T, D)
    perm = torch.randperm(T)
    assert torch.allclose(ffn(x[:, perm]), ffn(x)[:, perm], atol=1e-6)
 
 
def test_activation_is_between_the_two_linears_not_after():
    """W2(act(W1 x)) -- not act(W2(act(W1 x))). The wrong version biases every
    residual write positive and trains slightly worse, silently."""
    ffn = FeedForward(D).eval()
    with torch.no_grad():
        nn.init.constant_(ffn.fn_u.weight, -1.0)
        nn.init.zeros_(ffn.fn_l.bias)
    out = ffn(torch.randn(4, T, D))
    assert (out < 0).any(), "output is non-negative: an activation is applied after fc_out"
 
 
def test_dropout_is_off_in_eval_and_on_in_train():
    ffn = FeedForward(D, dropout=0.5)
    x = torch.randn(B, T, D)
    ffn.eval()
    assert torch.equal(ffn(x), ffn(x))
    ffn.train()
    assert not torch.equal(ffn(x), ffn(x))             # negative control
 
 
def test_gradients_reach_both_layers():
    ffn = FeedForward(D)
    ffn(torch.randn(B, T, D)).sum().backward()
    for name, p in ffn.named_parameters():
        assert p.grad is not None and p.grad.abs().max() > 0, name
 