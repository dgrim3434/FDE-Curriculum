import math
 
import pytest
import torch
import torch.nn as nn

from model.attention import causal_mask
from model.gpt import GPT, GPTConfig

D, H, T, B, V = 64, 4, 12, 2, 65
@pytest.fixture(autouse=True)
def _determinism():
    torch.manual_seed(0)
 
 
@pytest.fixture
def cfg():
    return GPTConfig(vocab_size=V, block_size=32, n_layer=3, n_head=H,
                     d_model=D, dropout=0.0)
 
 
@pytest.fixture
def model(cfg):
    return GPT(cfg)
 
 
@pytest.fixture
def idx():
    return torch.randint(0, V, (B, T))
 
 
@pytest.fixture
def targets():
    return torch.randint(0, V, (B, T))


def test_forward_shapes(model, idx, targets):
    out = model(idx, targets=targets, return_atten=True)
    assert out["logits"].shape == (B, T, V)
    assert out["loss"].shape == torch.Size([])            # scalar; backward() needs it
    assert len(out["attn"]) == model.cfg.n_layer
    assert out["attn"][0].shape == (B, H, T, T)
    assert torch.allclose(out["attn"][0].sum(-1), torch.ones(B, H, T), atol=1e-6)
 
 
def test_loss_is_none_without_targets(model, idx):
    assert model(idx)["loss"] is None
 
 
def test_init_loss_is_ln_vocab(cfg):
    """Two-sided check. Much higher -> init scale too large. Much lower -> a leak,
    or the number is not cross-entropy over your vocabulary."""
    torch.manual_seed(0)
    m = GPT(cfg)
    loss = m(torch.randint(0, V, (8, T)), targets=torch.randint(0, V, (8, T)))["loss"]
    assert abs(loss.item() - math.log(V)) < 0.15, loss.item()
 
 
def test_parameter_count_matches_the_formula(cfg):
    m = GPT(cfg)
    d, L, d_ff = cfg.d_model, cfg.n_layer, 4 * cfg.d_model
    per_block = (4 * (d * d + d)                 # W_q, W_k, W_v, W_o with biases
                 + (d * d_ff + d_ff)             # fc_in
                 + (d_ff * d + d)                # fc_out
                 + 2 * (2 * d))                  # ln1 + ln2, 2d EACH
    expected = per_block * L + V * d + cfg.block_size * d + 2 * d    # tied head, +ln_f
    assert m.n_params() == expected
 
 
def test_untied_adds_exactly_v_times_d():
    a = GPT(GPTConfig(vocab_size=V, block_size=32, n_layer=2, n_head=H, d_model=D,
                      tie_weights=True))
    b = GPT(GPTConfig(vocab_size=V, block_size=32, n_layer=2, n_head=H, d_model=D,
                      tie_weights=False))
    assert b.n_params() - a.n_params() == V * D
    assert a.lm_head.weight is a.tok_emb.weight
    assert b.lm_head.weight is not b.tok_emb.weight
 
 
def test_block_size_guard_fires(cfg):
    m = GPT(cfg)
    with pytest.raises(AssertionError):
        m(torch.randint(0, V, (1, cfg.block_size + 1)))
 
 
def test_mismatched_targets_are_rejected(model, idx):
    with pytest.raises(AssertionError):
        model(idx, targets=torch.randint(0, V, (B, T + 1)))
 
 
def test_no_future_leak(cfg):
    """Editing token k must leave every logit BEFORE k bit-identical."""
    m = GPT(cfg).eval()
    idx = torch.randint(0, V, (1, T))
    idx2 = idx.clone(); idx2[0, 7] = (idx[0, 7] + 1) % V
    with torch.no_grad():
        a, b = m(idx)["logits"], m(idx2)["logits"]
    assert torch.equal(a[:, :7], b[:, :7])
    assert not torch.equal(a[:, 7], b[:, 7])              # negative control
 
 
def test_gradient_does_not_flow_backwards_in_time(cfg):
    m = GPT(cfg).eval()
    idx = torch.randint(0, V, (1, T))
    emb = m.tok_emb(idx).detach().requires_grad_(True)
    x = emb + m.pos_emb(torch.arange(T))
    mask = causal_mask(T)
    for blk in m.blocks:
        x, _ = blk(x, mask=mask)
    m.lm_head(m.ln_final(x))[:, :7].sum().backward()
    assert (emb.grad[:, 7:] == 0).all()
 
 
def test_positional_embedding_is_actually_added(model):
    """The same token at two positions must NOT produce the same stream vector.
    A zeroed or unwired PE passes every other test in this file."""
    ids = torch.full((1, 4), 7, dtype=torch.long)
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(4))
    assert not torch.equal(x[0, 0], x[0, 1])
 
 
def test_every_parameter_gets_a_gradient(model, idx, targets):
    model(idx, targets=targets)["loss"].backward()
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or p.grad.abs().max() == 0]
    assert dead == [], dead
 
 
def test_tied_embedding_gradient_is_dense(cfg):
    """Tying makes the table dense: the softmax touches every vocabulary row."""
    m = GPT(cfg)
    idx = torch.randint(0, V, (2, 8))
    m(idx, targets=torch.randint(0, V, (2, 8)))["loss"].backward()
    rows = (m.tok_emb.weight.grad.norm(dim=1) > 0).sum().item()
    assert rows == V
 
 
def test_untied_embedding_gradient_is_sparse():
    """nn.Embedding backward is a scatter-add: only rows you indexed get gradient."""
    m = GPT(GPTConfig(vocab_size=V, block_size=32, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0, tie_weights=False))
    idx = torch.randint(0, V, (2, 8))
    m(idx, targets=torch.randint(0, V, (2, 8)))["loss"].backward()
    used = len(set(idx.flatten().tolist()))
    assert (m.tok_emb.weight.grad.norm(dim=1) > 0).sum().item() == used
    assert (m.lm_head.weight.grad.norm(dim=1) > 0).sum().item() == V
 
 
def test_depth_is_configurable(cfg):
    for n_layer in (1, 2, 6):
        m = GPT(GPTConfig(vocab_size=V, block_size=32, n_layer=n_layer, n_head=H,
                          d_model=D, dropout=0.0))
        assert len(m.blocks) == n_layer
        assert m(torch.randint(0, V, (1, 8)))["logits"].shape == (1, 8, V)
 
 
def test_eval_mode_is_deterministic_and_train_mode_is_not():
    m = GPT(GPTConfig(vocab_size=V, block_size=32, n_layer=2, n_head=H, d_model=D,
                      dropout=0.1))
    idx, tgt = torch.randint(0, V, (B, T)), torch.randint(0, V, (B, T))
    m.eval()
    assert m(idx, targets=tgt)["loss"].item() == m(idx, targets=tgt)["loss"].item()
    m.train()
    assert m(idx, targets=tgt)["loss"].item() != m(idx, targets=tgt)["loss"].item()
 
 
def test_can_overfit_fifty_tokens():
    """THE gate. A model that cannot drive 50 tokens to near-zero loss has a wiring
    bug, not a hyperparameter problem. Expect ~0.003; anything under 0.05 passes."""
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0))
    ids = torch.randint(0, V, (1, 51))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(400):
        loss = m(x, targets=y)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.05, loss.item()
 
 
def test_generate_shape_and_context_cropping():
    """block_size 8, 40 new tokens: without idx[:, -block_size:] this raises."""
    m = GPT(GPTConfig(vocab_size=V, block_size=8, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0))
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), 40)
    assert out.shape == (1, 41)
    assert out.min() >= 0 and out.max() < V
 
 
def test_generate_is_reproducible_under_a_seed():
    m = GPT(GPTConfig(vocab_size=V, block_size=16, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0))
    prompt = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(0); a = m.generate(prompt, 20)
    torch.manual_seed(0); b = m.generate(prompt, 20)
    torch.manual_seed(1); c = m.generate(prompt, 20)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
 
 
def test_low_temperature_is_greedy():
    """T -> 0 collapses onto the argmax. Fails if the division is applied after
    the softmax instead of to the logits."""
    m = GPT(GPTConfig(vocab_size=V, block_size=16, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0)).eval()
    prompt = torch.randint(0, V, (1, 8))
    with torch.no_grad():
        greedy = m(prompt)["logits"][0, -1].argmax().item()
    assert m.generate(prompt, 1, temperature=0.01)[0, -1].item() == greedy
 
 
def test_top_k_restricts_the_support():
    m = GPT(GPTConfig(vocab_size=V, block_size=16, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0)).eval()
    prompt = torch.randint(0, V, (1, 8))
    with torch.no_grad():
        allowed = set(m(prompt)["logits"][0, -1].topk(3).indices.tolist())
    for _ in range(30):
        assert m.generate(prompt, 1, top_k=3)[0, -1].item() in allowed
 