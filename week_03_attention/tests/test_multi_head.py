"""
Multi-head test suit
"""
import math, pytest, torch
from attention.masks import causal_mask
from attention.softmax import stable_softmax
from attention.multi_head import Multi_Head_Attention

D, H, B, T = 16, 4, 3, 6
DK = D // H

@pytest.fixture
def mha():
    torch.manual_seed(0)
    return Multi_Head_Attention(D, H)

@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, T, D)

def test_shapes_at_every_step(mha, x):
    mha(x, mask=causal_mask(T), record_shapes=True)
    s = mha.shapes
    assert s["q"] == (B, T, D)
    assert s["Q"] == (B, H, T, DK) and s["K"] == (B, H, T, DK)
    assert s["scores"] == (B, H, T, T)                   # T_q × T_k — no d_k here
    assert s["out_heads"] == (B, H, T, DK)
    assert s["merged"] == (B, T, D)
    assert s["y"] == (B, T, D)

def test_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        Multi_Head_Attention(10, 3)

def test_parameter_count(mha):
    assert sum(p.numel() for p in mha.parameters()) == 4 * (D * D + D)

def test_module_is_deterministic(mha, x):
    assert torch.equal(mha(x)[0], mha(x)[0])

def test_every_head_produces_a_distribution(mha, x):
    _, w = mha(x)
    assert torch.allclose(w.sum(-1), torch.ones(B, H, T), atol=1e-6)   # note the H
    assert (w >= 0).all()

def test_causal_mask_applies_to_every_head(mha, x):
    cm = causal_mask(T)
    _, w = mha(x, mask=cm)
    assert (w[..., ~cm] == 0).all()
    assert torch.allclose(w.sum(-1), torch.ones(B, H, T), atol=1e-6)
    assert (w.diagonal(dim1=-2, dim2=-1) > 0).all()

def test_matches_nn_multiheadattention(mha, x):
    ref = torch.nn.MultiheadAttention(D, H, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight]))
        ref.in_proj_bias.copy_(torch.cat([mha.W_q.bias, mha.W_k.bias, mha.W_v.bias]))
        ref.out_proj.weight.copy_(mha.W_o.weight); ref.out_proj.bias.copy_(mha.W_o.bias)
    cm = causal_mask(T)
    y_mine, w_mine = mha(x, mask=cm)
    y_ref, w_ref = ref(x, x, x, attn_mask=~cm, average_attn_weights=False)
    assert torch.allclose(y_mine, y_ref, atol=1e-5)
    assert torch.allclose(w_mine, w_ref, atol=1e-6)      # per-head weights too

def test_h1_equals_single_head(x):
    torch.manual_seed(3)
    m = Multi_Head_Attention(D, 1).eval()
    with torch.no_grad():
        m.W_o.weight.copy_(torch.eye(D)); m.W_o.bias.zero_()
    Q, K, V = m.W_q(x), m.W_k(x), m.W_v(x)
    expected = torch.softmax(Q @ K.transpose(-2,-1)/math.sqrt(D), -1) @ V
    assert torch.allclose(m(x)[0], expected, atol=1e-6)

def test_single_token_sequence(mha):
    y, w = mha(torch.randn(B, 1, D), mask=causal_mask(1))
    assert y.shape == (B, 1, D) and torch.allclose(w, torch.ones(B, H, 1, 1))

def test_d_k_of_one(x):
    torch.manual_seed(4)
    m = Multi_Head_Attention(D, D).eval()               # h == d_model, so d_k == 1
    y, w = m(x, mask=causal_mask(T))
    assert y.shape == (B, T, D) and w.shape == (B, D, T, T)

def test_padding_and_causal_combined_no_nan(mha, x):
    pad = torch.ones(B, T, dtype=torch.bool); pad[1, :2] = False       # LEFT padding
    mask = pad[:, None, None, :] & causal_mask(T)
    y, w = mha(x, mask=mask)
    assert not torch.isnan(y).any()
    assert (w[1, :, 0] == 0).all() and (w[1, :, 1] == 0).all()

def test_dropout_only_active_in_train_mode(x):
    torch.manual_seed(5)
    m = Multi_Head_Attention(D, H, dropout=0.5)
    m.train(); assert not torch.equal(m(x)[0], m(x)[0])
    m.eval();  assert torch.equal(m(x)[0], m(x)[0])