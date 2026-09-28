"""
masks test suit:
    - padding
    - causal
"""
import pytest
import torch
from attention.masks import causal_mask


def test_causal_mask():
    
    mask = causal_mask(3)
    
    res = torch.tensor([[True, False, False],
                        [True, True, False],
                        [True, True, True]])
    
    assert torch.equal(mask, res)

def test_cross_attention_mask():
    
    mask = causal_mask(3,2)
    
    res = torch.tensor([[False, False],
                        [True, False],
                        [True, True]])

    assert torch.equal(mask, res)
