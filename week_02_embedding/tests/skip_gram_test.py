import pytest
from skip_gram import Skip_Gram
import numpy as np

@pytest.fixture(scope='module')
def skip_gram():
    
    skip = Skip_Gram(20, 4 , seed=1)

    return skip
@pytest.fixture
def fresh_sg():
    
    skip = Skip_Gram(20, 4, seed=1)
    return skip

def test_skip_gram_sigmoid(skip_gram):
    
    num_sig = {0.0: 0.5, 2.0: 0.8808, -2.0: 0.1192, -8.0: 0.000335}
    
    assert all(abs(skip_gram.sigmoid(j) - i) < .00001 for j,i in num_sig.items())

def test_gradients_numerically(fresh_sg):
    
    fresh_sg.w_out = np.random.default_rng(0).normal(0,0.1,fresh_sg.w_out.shape).astype(np.float32)
    centers = np.array([3,7], np.int32)
    targets = np.array([[5,1,9,2],[4,0,6,8]], np.int32)
    labels = np.zeros((2,4)); labels[:,0] = 1.0
    def L():
        _,_,s = fresh_sg.forward(centers, targets); return fresh_sg.loss(s)
    v,u,s = fresh_sg.forward(centers, targets)
    ana = np.einsum('bk,bkd->bd', fresh_sg.sigmoid(s)-labels, u)
    eps = 1e-4
    for b,c in enumerate(centers):
        for j in range(fresh_sg.D):
            old = fresh_sg.w_in[c,j].copy()
            fresh_sg.w_in[c,j] = old+eps; Lp = L()
            fresh_sg.w_in[c,j] = old-eps; Lm = L()
            fresh_sg.w_in[c,j] = old
            num = (Lp-Lm)/(2*eps) * len(centers)      # loss() returns a MEAN
            assert abs(num - ana[b,j]) < 1e-3