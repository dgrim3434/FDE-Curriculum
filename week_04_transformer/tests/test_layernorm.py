import math
 
import pytest
import torch
import torch.nn as nn
 
D = 64

def test_matches_hand_computation():
    """x=[2,4,4,4,5,5,7,9]: mean 5, BIASED var 32/8=4, std 2 -> [-1.5,-.5,-.5,-.5,0,0,1,2]"""
    x = torch.tensor([2., 4., 4., 4., 5., 5., 7., 9.])
    ln = nn.LayerNorm(8)
    with torch.no_grad():
        out = ln(x)
    expected = torch.tensor([-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0])
    assert torch.allclose(out, expected, atol=1e-5)
 
 
def test_uses_biased_variance_not_unbiased():
    """torch.var() defaults to 1/(d-1). At d=8 that makes every output 6.9% too small."""
    x = torch.tensor([2., 4., 4., 4., 5., 5., 7., 9.])
    mu = x.mean()
    biased = ((x - mu) ** 2).mean()
    unbiased = x.var(unbiased=True)
    with torch.no_grad():
        out = nn.LayerNorm(8)(x)
    assert torch.allclose(out, (x - mu) / torch.sqrt(biased + 1e-5), atol=1e-6)
    assert not torch.allclose(out, (x - mu) / torch.sqrt(unbiased + 1e-5), atol=1e-3)
 
 
def test_gamma_and_beta_are_ones_and_zeros_at_init():
    ln = nn.LayerNorm(D)
    assert torch.equal(ln.weight, torch.ones(D))
    assert torch.equal(ln.bias, torch.zeros(D))
    assert sum(p.numel() for p in ln.parameters()) == 2 * D      # 2d, not d
 
 
def test_output_lies_on_the_sphere_of_radius_sqrt_d():
    x = torch.randn(4, 7, D) * 5 + 3
    with torch.no_grad():
        y = nn.LayerNorm(D)(x)
    assert y.sum(-1).abs().max() < 1e-4                           # zero-mean hyperplane
    assert torch.allclose(y.norm(dim=-1), torch.full(y.shape[:-1], math.sqrt(D)), atol=1e-2)
 
 
def test_is_invariant_to_scaling_and_shifting():
    x = torch.tensor([1., 5., 11., 7.])
    ln = nn.LayerNorm(4)
    with torch.no_grad():
        base, scaled, shifted = ln(x), ln(3 * x), ln(x + 10)
    assert torch.allclose(base, scaled, atol=1e-5)                # scale invariant
    assert torch.allclose(base, shifted, atol=1e-5)               # shift invariant
 
 
def test_eps_goes_inside_the_square_root():
    """On a near-constant vector, eps outside the root amplifies noise ~60x."""
    x = torch.tensor([1.0, 1.0, 1.0, 1.0001])
    mu, var, eps = x.mean(), ((x - x.mean()) ** 2).mean(), 1e-5
    inside = (x - mu) / torch.sqrt(var + eps)
    outside = (x - mu) / (torch.sqrt(var) + eps)
    with torch.no_grad():
        out = nn.LayerNorm(4)(x)
    assert torch.allclose(out, inside, atol=1e-6)
    assert not torch.allclose(out, outside, atol=1e-2)
    assert torch.isfinite(nn.LayerNorm(4)(torch.tensor([3., 3., 3., 3.]))).all()
 
 
def test_is_per_token_not_per_batch():
    """One example's output must not depend on the others in the batch."""
    ln = nn.LayerNorm(D)
    x = torch.randn(8, 5, D)
    with torch.no_grad():
        full, single = ln(x), ln(x[2:3])
    assert torch.allclose(full[2:3], single, atol=1e-6)
 
 
def test_normalizing_over_the_time_axis_breaks_causality():
    """The bug: nn.LayerNorm(T) on a transposed tensor. Editing a later position
    changes an earlier one. Correct feature-axis LN changes it by exactly 0."""
    B, T, d = 2, 5, 6
    x = torch.randn(B, T, d) * 2 + 1
    x2 = x.clone(); x2[:, 3] += 10.0
 
    ln_d = nn.LayerNorm(d)
    with torch.no_grad():
        assert torch.equal(ln_d(x)[:, 0], ln_d(x2)[:, 0])          # correct: exactly 0
 
    ln_t = nn.LayerNorm(T)
    with torch.no_grad():
        a = ln_t(x.transpose(1, 2)).transpose(1, 2)
        b = ln_t(x2.transpose(1, 2)).transpose(1, 2)
    assert (a[:, 0] - b[:, 0]).abs().max() > 1e-3                  # leak
    assert a.sum(-1).abs().max() > 1e-3                            # and sums are not 0