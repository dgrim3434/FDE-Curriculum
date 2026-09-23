import torch.nn as nn
from model.block import Transformer_Block
import math
import torch
from model.attention import causal_mask
import torch.nn.functional as F
from model.attention import stable_softmax
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = None
    dropout: float = 0.1
    norm: str = "pre"
    tie_weights: bool = True

class GPT(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        
        self.dropout = nn.Dropout(cfg.dropout)
        self.ln_final = nn.LayerNorm(cfg.d_model)
        
        self.blocks = nn.ModuleList([Transformer_Block(cfg.d_model, cfg.n_head, cfg.d_ff, dropout=cfg.dropout, norm = cfg.norm) for _ in range(cfg.n_layer)])
        
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        
        if cfg.tie_weights:
            
            self.lm_head.weight = self.tok_emb.weight
        
        for n, p in self.named_parameters():
            
            if n.endswith('W_o.weight') or n.endswith('fc_out.weight'):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))
                
                
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    """
    - Takes an input of ids in shape -> (B, T)
    - Extracts both the positional and semantic embeddings
    -
    """
    def forward(self, ids, targets = None, return_atten=False):
        
        B, T = ids.shape
        
        assert T <= self.cfg.block_size
        
        if targets is not None:
            assert targets.shape == ids.shape
        
        x = self.dropout(self.pos_emb(torch.arange(T, device=ids.device)) + self.tok_emb(ids))
        
        mask = causal_mask(T, device=ids.device)
        attens = []
        for blk in self.blocks:
            
            (x, atten) = blk(x, mask, return_atten)
            
            if return_atten:
                attens.append(atten)
        
        logits = self.lm_head(self.ln_final(x)) ## Shape -> (B, T, Vocab)
        
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        
        out = {'logits': logits, 'loss': loss}
        
        if return_atten: out['attn'] = attens
        
        return out
    
    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k = None):
        was_training = self.training
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)['logits'][:, -1, :] / temperature
            
            if top_k is not None:
                
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            probs = stable_softmax(logits, dims=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        if was_training:
            self.train()
        
        return idx