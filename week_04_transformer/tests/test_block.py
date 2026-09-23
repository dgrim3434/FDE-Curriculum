import pytest
import torch
import torch.nn as nn
from model.block import Transformer_Block as TransformerBlock
from model.attention import causal_mask

D, H, T, B = 64, 4, 12, 2

@pytest.fixture
def mask():
    return causal_mask(T)

def test_shape_is_preserved(mask):
    """If this fails you cannot stack, and you find out at layer 2 instead of layer 1."""
    for norm in ("pre", "post"):
        blk = TransformerBlock(D, H, norm=norm).eval()
        x = torch.randn(B, T, D)
        y, _ = blk(x, mask=mask)
        assert y.shape == x.shape
 
 
def test_prenorm_with_zero_branches_is_exactly_the_identity(mask):
    """THE residual test. Zero both branch outputs; a wired pre-norm block must
    return its input bit-for-bit. Fails with max deviation ~3.7 if the residual
    adds are missing."""
    blk = TransformerBlock(D, H, norm="pre").eval()
    with torch.no_grad():
        for m in (blk.attention.W_o, blk.ffn.fn_l):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    x = torch.randn(B, T, D)
    y, _ = blk(x, mask=mask)
    assert torch.equal(y, x)
 
 
def test_postnorm_with_zero_branches_is_NOT_the_identity(mask):
    """Negative control AND a real architectural fact: post-norm normalizes the
    identity path too, so it cannot pass the test above."""
    blk = TransformerBlock(D, H, norm="post").eval()
    with torch.no_grad():
        for m in (blk.attention.W_o, blk.ffn.fn_l):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    x = torch.randn(B, T, D) * 3 + 1
    y, _ = blk(x, mask=mask)
    assert not torch.allclose(y, x, atol=1e-3)
 
 
def test_block_is_causal(mask):
    blk = TransformerBlock(D, H).eval()
    x = torch.randn(1, T, D)
    x2 = x.clone(); x2[0, 6] = torch.randn(D)
    y1, _ = blk(x, mask=mask)
    y2, _ = blk(x2, mask=mask)
    assert torch.equal(y1[:, :6], y2[:, :6])          # bit-identical
    assert not torch.equal(y1[:, 6], y2[:, 6])        # negative control
 
 
def test_without_a_mask_the_block_leaks():
    """Negative control for the test above: prove the leak test CAN fail."""
    blk = TransformerBlock(D, H).eval()
    x = torch.randn(1, T, D)
    x2 = x.clone(); x2[0, 6] = torch.randn(D)
    y1, _ = blk(x, mask=None)
    y2, _ = blk(x2, mask=None)
    assert not torch.equal(y1[:, :6], y2[:, :6])
 
 
def test_attention_weights_are_returned_and_normalized(mask):
    blk = TransformerBlock(D, H).eval()
    _, w = blk(torch.randn(B, T, D), mask=mask, return_att=True)
    w = w.detach()
    assert w.shape == (B, H, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(B, H, T), atol=1e-6)
    assert float(w.triu(1).abs().max()) == 0.0        # strictly lower-triangular
    assert (w.diagonal(dim1=-2, dim2=-1) > 0).all()   # every token sees itself
 
 
def test_parameter_count_is_12_d_squared(mask):
    d_ff = 4 * D
    blk = TransformerBlock(D, H)
    expected = 4 * (D * D + D) + (D * d_ff + d_ff) + (d_ff * D + D) + 2 * (2 * D)
    assert sum(p.numel() for p in blk.parameters()) == expected
    assert expected / (D * D) == pytest.approx(12, abs=0.3)
 
 
def test_invalid_head_count_raises():
    with pytest.raises(ValueError):
        TransformerBlock(D, 7)                         # 64 % 7 != 0
 
 
def test_every_parameter_gets_a_gradient(mask):
    blk = TransformerBlock(D, H)
    y, _ = blk(torch.randn(B, T, D), mask=mask)
    y.sum().backward()
    dead = [n for n, p in blk.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert dead == [], dead
 