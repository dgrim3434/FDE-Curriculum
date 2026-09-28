"""Numbers that describe an attention pattern. Look at these BEFORE the pictures."""
import torch

def row_entropy(w, eps=1e-12):
    """H = -sum p log p, per query row. 0 = one key, ln(n) = uniform over n keys."""
    return -(w.clamp_min(eps).log() * w).sum(-1)

def normalized_row_entropy(w, causal=True):
    """Entropy / ln(#allowed keys). In [0,1]. 1.0 = uniform over what it COULD see.

    Without this normalization early rows always look 'focused' - row 0 has one
    option, so its entropy is 0 no matter what the model does. That is arithmetic,
    not behavior.
    """
    T = w.shape[-1]
    n = torch.arange(1, T + 1, device=w.device).float() if causal else torch.full((T,), float(T))
    return row_entropy(w) / n.log().clamp_min(1e-12)

def prev_token_score(w):
    """Mean weight on position i-1. High => a previous-token head."""
    return torch.stack([w[..., i, i - 1] for i in range(1, w.shape[-1])], -1).mean(-1)

def self_score(w):
    return w.diagonal(dim1=-2, dim2=-1)[..., 1:].mean(-1)

def sink_score(w):
    """Mean weight on key 0 from every query except query 0."""
    return w[..., 1:, 0].mean(-1)

def attention_received(w, causal=True):
    """Per KEY: total weight received / number of queries allowed to see it.

    The exposure correction matters: under a causal mask key j is visible to
    T-j queries, so raw column sums make early keys look important for free.
    """
    T = w.shape[-1]
    exposure = (torch.arange(T, 0, -1) if causal else torch.full((T,), T)).float().to(w.device)
    return w.sum(-2) / exposure

def uniform_baseline_received(T):
    """What attention_received looks like if every head is perfectly uniform."""
    w = torch.tril(torch.ones(T, T))
    w = w / w.sum(-1, keepdim=True)
    return attention_received(w)

def summary(w, causal=True):
    return {
        "norm_entropy": normalized_row_entropy(w, causal)[..., 1:].mean(-1),
        "prev_token":   prev_token_score(w),
        "self":         self_score(w),
        "sink":         sink_score(w),
    }