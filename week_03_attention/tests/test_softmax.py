"""
Softmax test Suit
"""

import pytest
import torch
from attention.softmax import stable_softmax


def test_softmax_sum():
    
    arr = torch.arange(10).reshape(5,2)
    
    soft_arr = stable_softmax(arr)
    
    assert all(soft_arr.sum(dim=-1) == 1)

def test_softmax_neg_inf():
    
    arr = torch.arange(10, dtype=torch.float32).reshape(5,2)
    arr[0] = torch.tensor([-torch.inf, -torch.inf])
    
    soft_arr = stable_softmax(arr)
    assert soft_arr[0].sum() == 0
    assert all(soft_arr[1:,].sum(dim=-1) == 1)
    
def test_against_pytorch():
    
    # dims = 2 x 4 x 5 x 3
    mat = torch.rand(120).reshape(2,4,5,-1)
    
    soft_1 = stable_softmax(mat)
    soft_2 = torch.softmax(mat, dim=-1)
    
    assert torch.allclose(soft_1, soft_2, atol=1e-6)

def test_overflow():
    
    large_nums = torch.arange(100, 200, 2).reshape(10, 5)
    
    soft = stable_softmax(large_nums)
    
    assert all(soft.sum(dim=-1) == 1)