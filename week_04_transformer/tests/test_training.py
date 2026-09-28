"""train/loop.py -- learning-rate schedule, optimizer grouping, evaluation, checkpoints."""
import math

import pytest
import torch

from train.loop import lr_at, configure_optimizer, evaluate
from train.data import load_shakespeare
from model.gpt import GPT, GPTConfig

D, H, V = 64, 4, 65
LR, LR_MIN, WARMUP, TOTAL = 1e-3, 1e-4, 100, 3000


# --------------------------------------------------------------------------- #
# learning-rate schedule
# --------------------------------------------------------------------------- #
def test_warmup_starts_nonzero_and_reaches_full_lr():
    """step 0 must move the weights. `step/warmup` instead of `(step+1)/warmup`
    makes the first step a no-op and nothing errors."""
    assert lr_at(0) == pytest.approx(LR / WARMUP)          # 1e-5
    assert lr_at(0) > 0
    assert lr_at(WARMUP - 1) == pytest.approx(LR)


def test_schedule_is_continuous_at_the_warmup_boundary():
    """A discontinuity here shows up as a loss spike exactly at step `warmup`."""
    assert lr_at(WARMUP - 1) == pytest.approx(lr_at(WARMUP), rel=1e-12)


def test_cosine_midpoint_is_the_arithmetic_mean():
    """p = 0.5 -> cos(pi/2) = 0 -> lr_min + 0.5*(lr - lr_min). Closed form, so assert it."""
    mid = WARMUP + (TOTAL - WARMUP) // 2
    assert lr_at(mid) == pytest.approx((LR + LR_MIN) / 2, rel=1e-6)


def test_warmup_increases_and_cosine_decreases():
    warm = [lr_at(s) for s in range(WARMUP)]
    cos = [lr_at(s) for s in range(WARMUP, TOTAL)]
    assert all(b > a for a, b in zip(warm, warm[1:]))
    assert all(b < a for a, b in zip(cos, cos[1:]))


def test_schedule_stays_inside_its_bounds():
    vals = [lr_at(s) for s in range(TOTAL)]
    assert min(vals) >= LR / WARMUP - 1e-12
    assert max(vals) <= LR + 1e-12
    assert lr_at(TOTAL - 1) == pytest.approx(LR_MIN, abs=1e-6)


def test_schedule_past_total_does_not_ramp_back_up():
    """cos(pi*p) keeps oscillating for p > 1. Extend a run or resume with a stale
    step counter and the lr climbs back to FULL value -- unclamped, lr_at(5900)
    returns 1.0e-3, the peak, at what you think is the end of training.
    Fix: p = min(1.0, (step - warmup) / (total - warmup))."""
    for step in (TOTAL, TOTAL + 1000, TOTAL + 2900, TOTAL + 5900):
        assert lr_at(step) == pytest.approx(LR_MIN, abs=1e-9), \
            f"lr at step {step} is {lr_at(step):.2e}, expected it pinned at lr_min"


def test_degenerate_configs_do_not_silently_produce_garbage():
    with pytest.raises((ZeroDivisionError, ValueError, AssertionError)):
        lr_at(100, warmup=100, total=100)                   # total == warmup
    assert lr_at(0, warmup=0) == pytest.approx(LR)          # warmup disabled


def test_negative_step_is_rejected():
    """A resume bug can hand you step < 0. Unclamped, lr_at(-5) returns a NEGATIVE
    learning rate, which ascends the loss."""
    with pytest.raises((ValueError, AssertionError)):
        lr_at(-5)


# --------------------------------------------------------------------------- #
# optimizer configuration
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_model():
    return GPT(GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H,
                         d_model=D, dropout=0.0))


def test_weight_decay_applies_to_matrices_only(small_model):
    """Decaying a LayerNorm gain toward zero shrinks the normalization the whole
    model depends on. Biases and gains belong in a zero-decay group."""
    opt = configure_optimizer(small_model, lr=1e-3, weight_decay=0.1)
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    for name, p in small_model.named_parameters():
        if p.dim() >= 2:
            assert id(p) in decayed, f"{name} should be decayed"
        else:
            assert id(p) not in decayed, f"{name} is 1-D and must NOT be decayed"


def test_every_parameter_is_in_the_optimizer_exactly_once(small_model):
    """A parameter missing from the optimizer never trains, and nothing errors."""
    groups = configure_optimizer(small_model, lr=1e-3).param_groups
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter appears in two groups"
    assert set(ids) == {id(p) for p in small_model.parameters() if p.requires_grad}


def test_first_adam_step_is_lr_sized(small_model):
    """m_hat/sqrt(v_hat) = g/|g| = +-1 on step 1, so every element moves by ~lr
    regardless of its gradient. This is what warmup protects against."""
    x = torch.randint(0, V, (4, 16))
    before = small_model.blocks[0].attention.W_q.weight.detach().clone()
    opt = configure_optimizer(small_model, lr=1e-3)
    small_model(x, targets=torch.randint(0, V, (4, 16)))["loss"].backward()
    opt.step()
    delta = (small_model.blocks[0].attention.W_q.weight.detach() - before).abs()
    assert delta.mean().item() == pytest.approx(1e-3, rel=0.1)


def test_betas_are_set_for_language_modelling(small_model):
    """PyTorch defaults to beta2=0.999; LM training uses 0.95."""
    opt = configure_optimizer(small_model, lr=1e-3)
    for g in opt.param_groups:
        assert g["betas"] == (0.9, 0.95)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def splits():
    train, val, _, _ = load_shakespeare()
    return {"train": train, "val": val}


def test_evaluate_returns_one_number_per_split(splits):
    m = GPT(GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0))
    out = evaluate(m, splits, block_size=32, batch_size=4, n_batches=2)
    assert set(out) == {"train", "val"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in out.values())


def test_evaluate_is_deterministic(splits):
    """Fixed eval batches + eval() mode. If this is flaky your val curve is
    measuring dropout noise, not learning."""
    m = GPT(GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                      dropout=0.1))
    a = evaluate(m, splits, block_size=32, batch_size=4, n_batches=3)
    b = evaluate(m, splits, block_size=32, batch_size=4, n_batches=3)
    assert a == b


def test_evaluate_restores_training_mode(splits):
    """Forgetting model.train() at the end silently disables dropout for the rest
    of the run."""
    m = GPT(GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                      dropout=0.1))
    m.train()
    evaluate(m, splits, block_size=32, batch_size=4, n_batches=2)
    assert m.training


def test_evaluate_does_not_accumulate_gradients(splits):
    """Missing @torch.no_grad() builds a graph over every eval batch."""
    m = GPT(GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0))
    for p in m.parameters():
        p.grad = None
    evaluate(m, splits, block_size=32, batch_size=4, n_batches=2)
    assert all(p.grad is None for p in m.parameters())


def test_untrained_model_evaluates_near_ln_vocab(splits):
    m = GPT(GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0))
    out = evaluate(m, splits, block_size=32, batch_size=8, n_batches=5)
    assert abs(out["val"] - math.log(V)) < 0.2


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def test_checkpoint_round_trips(tmp_path):
    """A checkpoint you cannot reload is not a checkpoint."""
    cfg = GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                    dropout=0.0)
    m = GPT(cfg).eval()
    x = torch.randint(0, V, (2, 16))
    before = m(x)["logits"]

    path = tmp_path / "ckpt.pt"
    torch.save(m.state_dict(), path)

    fresh = GPT(cfg).eval()
    assert not torch.equal(fresh(x)["logits"], before)       # negative control
    fresh.load_state_dict(torch.load(path))
    assert torch.equal(fresh(x)["logits"], before)           # bit-exact


def test_weight_tying_survives_a_checkpoint_round_trip(tmp_path):
    """Loading a state dict can leave two tensors that drift apart, changing the
    parameter count by V*d without warning."""
    cfg = GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                    tie_weights=True)
    m = GPT(cfg)
    path = tmp_path / "tied.pt"
    torch.save(m.state_dict(), path)
    m2 = GPT(cfg)
    m2.load_state_dict(torch.load(path))
    assert m2.lm_head.weight is m2.tok_emb.weight
    assert m2.n_params() == m.n_params()


# --------------------------------------------------------------------------- #
# the integration test: a few real steps must actually reduce the loss
# --------------------------------------------------------------------------- #
def test_short_run_reduces_the_loss(splits):
    """50 steps on real text. Not a convergence test -- a wiring test at the level
    of the whole pipeline, including get_batch."""
    from train.data import get_batch
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=H, d_model=D,
                      dropout=0.0))
    opt = configure_optimizer(m, lr=1e-3)
    g = torch.Generator().manual_seed(0)
    first = last = None
    for i in range(50):
        x, y = get_batch(splits["train"], 64, 16, generator=g)
        loss = m(x, targets=y)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert first == pytest.approx(math.log(V), abs=0.2)
    assert last < first - 0.5, f"{first:.3f} -> {last:.3f}: the pipeline is not learning"


def test_clip_grad_norm_returns_the_preclip_value(small_model):
    """Log this value. It is the tiebreaker between 'lr too high' and 'wiring bug'."""
    x = torch.randint(0, V, (4, 16))
    small_model(x, targets=torch.randint(0, V, (4, 16)))["loss"].backward()
    gnorm = torch.nn.utils.clip_grad_norm_(small_model.parameters(), 1.0)
    assert float(gnorm) > 0 and math.isfinite(float(gnorm))