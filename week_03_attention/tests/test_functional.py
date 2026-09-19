"""
Dot product test Suit
"""
import pytest
import torch
from attention.functional import scaled_dot_product_attention
import math
from attention.softmax import stable_softmax
from attention.masks import causal_mask
def test_sdpa_by_hand():
    
    q = torch.arange(10, dtype=torch.float32).reshape(1,1,2,5)
    k = torch.arange(0, 8, 0.5, dtype=torch.float32)[:-1].reshape(1,1,3,5)
    v = torch.arange(0, 3, 0.5, dtype=torch.float32).reshape(1,1,3,2)
    
    d_k = q.shape[-1]
    
    res = torch.einsum('BHQD, BHKD -> BHQK', q, k) / math.sqrt(d_k)
    
    weights = stable_softmax(res)
    residuals = weights @ v
    
    test_residual, test_weights = scaled_dot_product_attention(q,k,v)
    
    assert torch.allclose(weights, test_weights, atol=1e-6)
    assert torch.allclose(residuals, test_residual, atol=1e-6)

def test_weights_no_mask():
    
    q = torch.arange(10, dtype=torch.float32).reshape(1,1,2,5)
    k = torch.arange(0, 8, 0.5, dtype=torch.float32)[:-1].reshape(1,1,3,5)
    v = torch.arange(0, 3, 0.5, dtype=torch.float32).reshape(1,1,3,2)
    
    _, weights = scaled_dot_product_attention(q,k,v)
    
    assert all(weights.sum(dim=-1).ravel() == 1)

def testing_mask_all_true():
    
    q = torch.arange(10, dtype=torch.float32).reshape(1,1,2,5)
    k = torch.arange(0, 8, 0.5, dtype=torch.float32)[:-1].reshape(1,1,3,5)
    v = torch.arange(0, 3, 0.5, dtype=torch.float32).reshape(1,1,3,2)
    
    mask = torch.ones(2,3, dtype=torch.bool)
    
    assert torch.allclose(scaled_dot_product_attention(q,k,v)[0], scaled_dot_product_attention(q,k,v, mask)[0], atol=1e-6)

def test_masked_key():
    keep = torch.ones(5, dtype=torch.bool); keep[2] = False
    q = torch.arange(50, dtype=torch.float32).reshape(1,2,5,5)
    k = torch.arange(1, 26, 0.5, dtype=torch.float32).reshape(1,2,5,5)
    v = torch.arange(1, 11, 0.2, dtype=torch.float32).reshape(1,2,5,5)
    
    o1, _ = scaled_dot_product_attention(q, k, v, keep[None,None,None,:].expand(1,2,5,5))   # mask out key 2
    o2, _ = scaled_dot_product_attention(q, k[:,:,keep], v[:,:,keep])                        # delete key 2 entirely
    assert torch.allclose(o1, o2, atol=1e-6)

def test_leak():
    q = torch.arange(50, dtype=torch.float32).reshape(1,2,5,5)
    k = torch.arange(1, 26, 0.5, dtype=torch.float32).reshape(1,2,5,5)
    v = torch.arange(1, 11, 0.2, dtype=torch.float32).reshape(1,2,5,5)
    K2, V2 = k.clone(), v.clone()
    K2[:,:,4] = torch.randn(5); V2[:,:,4] = torch.randn(5)
    cm = causal_mask(5, 5)
    
    assert torch.equal(scaled_dot_product_attention(q,k,v,cm)[0][...,:4,:], scaled_dot_product_attention(q,K2,V2,cm)[0][...,:4,:])   # bit-identical
