"""
Single Head Test Suit
"""
import math, pytest, torch
import torch.nn.functional as F
from attention.masks import causal_mask
from attention.single_head import Single_Head_Attention

D = 12
B = 3
T = 6


@pytest.fixture
def sha():
    torch.manual_seed(0)
    return Single_Head_Attention(D)

@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.rand(B, T, D)

def test_output_and_weight_shapes(sha, x):
    
    y, w = sha.forward(x)
    
    assert y.shape == (B,T, D)
    assert w.shape == (B, T, T)

def test_dtype_is_preserved(sha, x):
    sha = sha.double()
    y, _ = sha(x.double())
    
    assert y.dtype == torch.float64

def test_module_is_deterministic(sha, x):
    assert torch.equal(sha(x)[0], sha(x)[0])

def test_weights_are_a_distribution(sha, x):
    _, w = sha(x)
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-6)
    assert (w >= 0).all()

def test_causal_weights_are_lower_triangular(sha, x):
    cm = causal_mask(T)
    _, w = sha(x, mask=cm)
    assert (w[..., ~cm] == 0).all()                            
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-6) 
    assert (w.diagonal(dim1=-2, dim2=-1) > 0).all()            

def test_batch_items_are_independent(sha, x):
    x2 = x.clone(); x2[0] = torch.randn(T, D)
    assert torch.equal(sha(x)[0][1:], sha(x2)[0][1:])

def test_permutation_equivariance_without_mask(sha, x):
    perm = torch.randperm(T)
    assert torch.allclose(sha(x)[0][:, perm], sha(x[:, perm])[0], atol=1e-5)

def test_no_future_leak(sha, x):
    cm = causal_mask(T)
    x2 = x.clone(); x2[:, 4] = torch.randn(D)
    assert torch.equal(sha(x, mask=cm)[0][:, :4], sha(x2, mask=cm)[0][:, :4])

def test_leak_test_can_fail(sha, x):
    x2 = x.clone(); x2[:, 4] = torch.randn(D)
    assert not torch.equal(sha(x)[0][:, :4], sha(x2)[0][:, :4])

def test_no_future_leak_in_gradients(sha, x):
    x = x.clone().requires_grad_(True)
    sha(x, mask=causal_mask(T))[0][:, :4].sum().backward()
    assert (x.grad[:, 4:] == 0).all()

def test_projections_are_applied_and_distinct(sha, x):
    Q, K, V = sha.W_q(x), sha.W_k(x), sha.W_v(x)
    expected_w = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)), -1)
    y, w = sha(x)
    assert torch.allclose(w, expected_w, atol=1e-6)
    assert torch.allclose(y, expected_w @ V, atol=1e-6)

def test_matches_torch_reference(sha, x):
    Q, K, V = sha.W_q(x), sha.W_k(x), sha.W_v(x)
    y, _ = sha(x, mask=causal_mask(T))
    assert torch.allclose(y, F.scaled_dot_product_attention(Q, K, V, is_causal=True), atol=1e-5)

def test_parameters_receive_gradients(sha, x):
    sha(x, mask=causal_mask(T))[0].pow(2).mean().backward()
    for name, p in sha.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name

def test_context_defaults_to_self_attention(sha, x):
    assert torch.equal(sha(x)[0], sha(x, context=x)[0])

def test_context_is_actually_used(sha, x):
    ctx = torch.randn(B, 9, D)
    y, w = sha(x, context=ctx)
    assert y.shape == (B, T, D) and w.shape == (B, T, 9)
    assert not torch.allclose(y, sha(x)[0])

def test_queries_come_from_x_not_context(sha, x):
    ctx = torch.randn(B, 9, D)
    assert sha(x[:, :2], context=ctx)[0].shape == (B, 2, D)

def test_single_token_sequence(sha):
    y, w = sha(torch.randn(B, 1, D), mask=causal_mask(1))
    assert y.shape == (B, 1, D) and torch.allclose(w, torch.ones(B, 1, 1))

def test_fully_masked_row_gives_zeros_not_nan(sha, x):
    mask = causal_mask(T).clone()
    mask[0, 0] = False
    y, w = sha(x, mask=mask)
    assert not torch.isnan(y).any()
    assert (w[:, 0] == 0).all()