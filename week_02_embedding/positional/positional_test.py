from encoding import sinusiodal_position_encodings
import pytest
import numpy as np

@pytest.fixture(scope='module')
def pe():
    
    pe = sinusiodal_position_encodings(100, 26)
    
    return pe

def test_positional_norm(pe):
    
    norms = np.linalg.norm(pe, axis=1)
    assert np.allclose(norms, np.sqrt(26/2), atol=1e-6)

def test_positional_value_bounds(pe):
    
    assert pe.min() >= -1.0 and pe.max() <= 1
    
def test_dot_product(pe):
    
    for k in [1,5,10,20]:
        dots = np.array([pe[p] @ pe[p + k] for p in range(26)])
        assert np.allclose(dots, dots[0], atol=1e-6)

def test_against_naive_loop(pe):
    
    ref = np.zeros((100, 26))
    
    for p in range(100):
        for i in range(26 // 2):
            w = 1.0 / (10000 ** (2*i/ 26))
            ref[p, 2*i] = np.sin(p * w)
            ref[p, 2*i + 1] = np.cos(p * w)
    
    assert np.allclose(pe, ref, atol=1e-9)