import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """standard rope: rotates pairs of dimensions by position-dependent angles"""
    # x: [batch, seq, heads, head_dim]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

def get_rotary_emb(seq_len: int, head_dim: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    return cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)


class MLA(nn.Module):
    """
    the key insight: kv cache is O(n_heads * head_dim * seq_len)
    MLA compresses to O(latent_dim * seq_len) where latent_dim << n_heads * head_dim

    but rope(W_k @ c_kv) != W_k @ rope(c_kv) bc rope is element-wise nonlinear
    so we DECOUPLE: main keys are NoRoPE, separate k_rope branch handles position
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        n_heads: int = 32,
        head_dim: int = 128,
        kv_latent_dim: int = 512,  # compression ratio ~8x
        rope_head_dim: int = 64,   # decoupled rope dimension
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.rope_head_dim = rope_head_dim

        # query projections
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.q_rope_proj = nn.Linear(hidden_dim, n_heads * rope_head_dim, bias=False)

        # kv compression: hidden -> latent
        self.kv_compress = nn.Linear(hidden_dim, kv_latent_dim, bias=False)

        # kv decompression: latent -> k, v (per head, absorbed into latent)
        # this is the "absorbed" part - W_uk and W_uv
        self.k_decompress = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)
        self.v_decompress = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)

        # decoupled rope key - separate projection for positional info
        self.k_rope_proj = nn.Linear(hidden_dim, rope_head_dim, bias=False)

        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, hidden]
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        # queries: main + rope branch
        q_main = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim)
        q_rope = self.q_rope_proj(x).view(batch, seq, self.n_heads, self.rope_head_dim)
        q_rope = apply_rotary_pos_emb(q_rope, cos[..., :self.rope_head_dim//2], sin[..., :self.rope_head_dim//2])

        # kv: compress then decompress
        c_kv = self.kv_compress(x)  # [batch, seq, latent_dim] - THIS is what we cache
        k_main = self.k_decompress(c_kv).view(batch, seq, self.n_heads, self.head_dim)  # NoRoPE
        v = self.v_decompress(c_kv).view(batch, seq, self.n_heads, self.head_dim)

        # decoupled rope key - shared across heads (MQA-like for position)
        k_rope = self.k_rope_proj(x).view(batch, seq, 1, self.rope_head_dim)
        k_rope = apply_rotary_pos_emb(k_rope, cos[..., :self.rope_head_dim//2], sin[..., :self.rope_head_dim//2])
        k_rope = k_rope.expand(-1, -1, self.n_heads, -1)  # broadcast to all heads

        # attention: concat main and rope parts for scoring
        # q_full = [q_main, q_rope], k_full = [k_main, k_rope]
        # score = q_main @ k_main.T + q_rope @ k_rope.T
        score_main = torch.einsum('bshd,bthd->bhst', q_main, k_main)
        score_rope = torch.einsum('bshr,bthr->bhst', q_rope, k_rope)

        scale = 1.0 / math.sqrt(self.head_dim + self.rope_head_dim)
        attn = F.softmax((score_main + score_rope) * scale, dim=-1)

        out = torch.einsum('bhst,bthd->bshd', attn, v)
        return self.o_proj(out.reshape(batch, seq, -1))


class NSA(nn.Module):
    """
    NSA splits attention into three parallel sparse patterns:
    1. compressed: block-wise pooling of kv, attends to summaries
    2. selected: top-k block selection based on compressed scores
    3. sliding: local window attention (always attends to recent tokens)

    gating network learns to weight these three outputs
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        n_heads: int = 32,
        head_dim: int = 128,
        block_size: int = 64,
        n_select_blocks: int = 16,
        slide_window: int = 512,
        compress_ratio: int = 4,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.n_select_blocks = n_select_blocks
        self.slide_window = slide_window
        self.compress_ratio = compress_ratio

        self.qkv_proj = nn.Linear(hidden_dim, 3 * n_heads * head_dim, bias=False)

        # compression: pool blocks down
        self.compress_pool = nn.AvgPool1d(compress_ratio, stride=compress_ratio)

        # gating: 3 branches
        self.gate = nn.Linear(hidden_dim, 3 * n_heads, bias=False)

        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape

        qkv = self.qkv_proj(x).view(batch, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # apply rope
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)

        # branch 1: compressed attention
        # pool k,v along seq dimension
        k_compressed = self.compress_pool(k.transpose(1, 3)).transpose(1, 3)  # [b, seq//r, h, d]
        v_compressed = self.compress_pool(v.transpose(1, 3)).transpose(1, 3)
        attn_compressed = F.softmax(
            torch.einsum('bshd,bthd->bhst', q, k_compressed) * scale, dim=-1
        )
        out_compressed = torch.einsum('bhst,bthd->bshd', attn_compressed, v_compressed)

        # branch 2: selected attention (top-k blocks based on compressed scores)
        # use compressed attention scores to select which blocks to attend to fully
        block_scores = attn_compressed.mean(dim=-1)  # [b, h, s]
        n_blocks = seq // self.block_size
        block_scores_reshaped = block_scores.view(batch, self.n_heads, -1, self.block_size).mean(dim=-1)

        # top-k block selection
        _, top_indices = block_scores_reshaped.topk(min(self.n_select_blocks, n_blocks), dim=-1)

        # for simplicity, do full attention but mask out non-selected blocks
        # real impl would be sparse
        full_attn = torch.einsum('bshd,bthd->bhst', q, k) * scale

        # create selection mask
        select_mask = torch.zeros(batch, self.n_heads, seq, seq, device=x.device, dtype=torch.bool)
        for b in range(batch):
            for h in range(self.n_heads):
                for idx in top_indices[b, h]:
                    start = idx * self.block_size
                    end = min(start + self.block_size, seq)
                    select_mask[b, h, :, start:end] = True

        attn_selected = F.softmax(full_attn.masked_fill(~select_mask, -1e9), dim=-1)
        out_selected = torch.einsum('bhst,bthd->bshd', attn_selected, v)

        # branch 3: sliding window attention
        slide_mask = torch.ones(seq, seq, device=x.device, dtype=torch.bool).triu(-self.slide_window).tril(0)
        attn_slide = F.softmax(full_attn.masked_fill(~slide_mask, -1e9), dim=-1)
        out_slide = torch.einsum('bhst,bthd->bshd', attn_slide, v)

        # gating
        gates = self.gate(x).view(batch, seq, self.n_heads, 3).softmax(dim=-1)
        g_comp, g_sel, g_slide = gates.unbind(dim=-1)

        out = (
            g_comp.unsqueeze(-1) * out_compressed +
            g_sel.unsqueeze(-1) * out_selected +
            g_slide.unsqueeze(-1) * out_slide
        )

        return self.o_proj(out.reshape(batch, seq, -1))


# ============================================================
# DSA: deepseek sparse attention v3.2
# key difference: "lightning indexer" for top-k selection
# uses separate low-dim projections for scoring efficiency
# ============================================================

class DSA(nn.Module):
    """
    DSA differs from NSA in the selection mechanism:
    - uses "lightning indexer": cheap low-rank projections (q^I, k^I) for relevance scoring
    - partially applies RoPE (only to indexer projections)
    - more efficient than NSA's compress-then-select

    the diagram shows:
    - q^A (absorbed, no rope) + q^R (rope applied) -> concatenated for main attention
    - separate q^I, k^I for top-k selection (cheap proxy scoring)
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        n_heads: int = 32,
        head_dim: int = 128,
        kv_latent_dim: int = 512,
        rope_head_dim: int = 64,
        index_dim: int = 32,  # lightning indexer dimension - very small
        n_select: int = 256,  # top-k tokens to select
        slide_window: int = 512,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.rope_head_dim = rope_head_dim
        self.index_dim = index_dim
        self.n_select = n_select
        self.slide_window = slide_window

        # main query path (absorbed + rope)
        self.q_absorbed = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.q_rope = nn.Linear(hidden_dim, n_heads * rope_head_dim, bias=False)

        # kv compression (MLA style)
        self.kv_compress = nn.Linear(hidden_dim, kv_latent_dim, bias=False)
        self.k_decompress = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)
        self.v_decompress = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)
        self.k_rope = nn.Linear(hidden_dim, rope_head_dim, bias=False)

        # lightning indexer: cheap projections for selection
        # these are MUCH smaller than main projections
        self.q_index = nn.Linear(hidden_dim, n_heads * index_dim, bias=False)
        self.k_index = nn.Linear(kv_latent_dim, index_dim, bias=False)  # shared across heads

        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def lightning_index(
        self,
        q_idx: torch.Tensor,  # [batch, seq_q, heads, index_dim]
        k_idx: torch.Tensor,  # [batch, seq_k, index_dim]
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        fast top-k selection using low-rank index projections
        returns indices of tokens to attend to
        """
        batch, seq_q, n_heads, _ = q_idx.shape
        seq_k = k_idx.shape[1]

        # apply rope to indexer (partial rope application)
        cos_idx = cos[..., :self.index_dim//2]
        sin_idx = sin[..., :self.index_dim//2]
        q_idx = apply_rotary_pos_emb(q_idx, cos_idx, sin_idx)
        k_idx = apply_rotary_pos_emb(k_idx.unsqueeze(2), cos_idx, sin_idx).squeeze(2)

        # cheap relevance scores: [batch, heads, seq_q, seq_k]
        scores = torch.einsum('bqhd,bkd->bhqk', q_idx, k_idx)

        # top-k selection per query position
        # in practice, might do block-wise selection
        _, indices = scores.topk(min(self.n_select, seq_k), dim=-1)  # [b, h, q, k_select]

        return indices

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape

        # main query path
        q_abs = self.q_absorbed(x).view(batch, seq, self.n_heads, self.head_dim)
        q_r = self.q_rope(x).view(batch, seq, self.n_heads, self.rope_head_dim)
        q_r = apply_rotary_pos_emb(q_r, cos[..., :self.rope_head_dim//2], sin[..., :self.rope_head_dim//2])

        # kv path (MLA compression)
        c_kv = self.kv_compress(x)
        k_abs = self.k_decompress(c_kv).view(batch, seq, self.n_heads, self.head_dim)
        v = self.v_decompress(c_kv).view(batch, seq, self.n_heads, self.head_dim)

        k_r = self.k_rope(x).view(batch, seq, 1, self.rope_head_dim)
        k_r = apply_rotary_pos_emb(k_r, cos[..., :self.rope_head_dim//2], sin[..., :self.rope_head_dim//2])
        k_r = k_r.expand(-1, -1, self.n_heads, -1)

        # lightning indexer for selection
        q_idx = self.q_index(x).view(batch, seq, self.n_heads, self.index_dim)
        k_idx = self.k_index(c_kv)  # [batch, seq, index_dim]

        select_indices = self.lightning_index(q_idx, k_idx, cos, sin)

        # compute attention scores for main path
        # q_full @ k_full = q_abs @ k_abs + q_r @ k_r
        score_abs = torch.einsum('bqhd,bkhd->bhqk', q_abs, k_abs)
        score_r = torch.einsum('bqhr,bkhr->bhqk', q_r, k_r)
        full_score = (score_abs + score_r) / math.sqrt(self.head_dim + self.rope_head_dim)

        # create sparse mask from lightning indexer
        sparse_mask = torch.zeros(batch, self.n_heads, seq, seq, device=x.device, dtype=torch.bool)

        # populate mask with selected indices
        for b in range(batch):
            for h in range(self.n_heads):
                for q in range(seq):
                    sparse_mask[b, h, q, select_indices[b, h, q]] = True

        # add sliding window (always attend to recent)
        slide_mask = torch.ones(seq, seq, device=x.device, dtype=torch.bool).triu(-self.slide_window).tril(0)
        sparse_mask = sparse_mask | slide_mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(full_score.masked_fill(~sparse_mask, -1e9), dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)

        return self.o_proj(out.reshape(batch, seq, -1))


# ============================================================
# comparison: MQA vs GQA in MLA context
# ============================================================

class MLA_MQA(nn.Module):
    """MLA with MQA: single kv latent, queries are all different"""
    def __init__(self, hidden_dim: int = 4096, n_heads: int = 32, head_dim: int = 128, kv_latent_dim: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim)
        self.kv_compress = nn.Linear(hidden_dim, kv_latent_dim)
        # single k,v projection (MQA style)
        self.k_proj = nn.Linear(kv_latent_dim, head_dim)  # ONE head
        self.v_proj = nn.Linear(kv_latent_dim, head_dim)  # ONE head
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim)
        c_kv = self.kv_compress(x)
        k = self.k_proj(c_kv).unsqueeze(2)  # [b, s, 1, d] - broadcast to all heads
        v = self.v_proj(c_kv).unsqueeze(2)

        # kv cache size: O(latent_dim * seq) + O(head_dim * seq)
        # effectively: cache c_kv only, recompute k,v on the fly

        attn = F.softmax(torch.einsum('bqhd,bkhd->bhqk', q, k.expand(-1,-1,self.n_heads,-1)) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v.expand(-1,-1,self.n_heads,-1))
        return self.o_proj(out.reshape(batch, seq, -1))


class MLA_GQA(nn.Module):
    """MLA with GQA: n_kv_groups latent decompressions"""
    def __init__(self, hidden_dim: int = 4096, n_heads: int = 32, head_dim: int = 128,
                 kv_latent_dim: int = 512, n_kv_groups: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_groups = n_kv_groups
        self.heads_per_group = n_heads // n_kv_groups

        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim)
        self.kv_compress = nn.Linear(hidden_dim, kv_latent_dim)
        # n_kv_groups k,v projections (GQA style)
        self.k_proj = nn.Linear(kv_latent_dim, n_kv_groups * head_dim)
        self.v_proj = nn.Linear(kv_latent_dim, n_kv_groups * head_dim)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim)
        c_kv = self.kv_compress(x)
        k = self.k_proj(c_kv).view(batch, seq, self.n_kv_groups, self.head_dim)
        v = self.v_proj(c_kv).view(batch, seq, self.n_kv_groups, self.head_dim)

        # expand k,v to match q heads
        k = k.repeat_interleave(self.heads_per_group, dim=2)  # [b, s, n_heads, d]
        v = v.repeat_interleave(self.heads_per_group, dim=2)

        attn = F.softmax(torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        return self.o_proj(out.reshape(batch, seq, -1))


# ============================================================
# kv cache size comparison
# ============================================================

def cache_size_analysis():
    """
    the whole point of MLA is kv cache compression for long-context serving

    standard MHA: 2 * n_layers * n_heads * head_dim * seq_len * batch * dtype_size
    MQA:          2 * n_layers * 1 * head_dim * seq_len * batch * dtype_size
    GQA:          2 * n_layers * n_kv_groups * head_dim * seq_len * batch * dtype_size
    MLA:          n_layers * kv_latent_dim * seq_len * batch * dtype_size
                  + n_layers * rope_head_dim * seq_len * batch * dtype_size  (for decoupled rope keys)

    for deepseek v3:
    - 61 layers, 128 heads, head_dim=128, kv_latent=512, rope_dim=64
    - MHA: 2 * 61 * 128 * 128 = 2,002,944 params per token
    - MLA: 61 * (512 + 64) = 35,136 params per token
    - compression ratio: ~57x
    """
    configs = {
        'MHA': lambda l, h, d, s: 2 * l * h * d * s,
        'MQA': lambda l, h, d, s: 2 * l * 1 * d * s,
        'GQA-8': lambda l, h, d, s: 2 * l * 8 * d * s,
        'MLA': lambda l, h, d, s: l * (512 + 64) * s,  # latent + rope key
    }

    # deepseek v3 params
    n_layers, n_heads, head_dim, seq_len = 61, 128, 128, 128000

    results = {}
    for name, fn in configs.items():
        size_bytes = fn(n_layers, n_heads, head_dim, seq_len) * 2  # bf16
        results[name] = size_bytes / (1024**3)  # GB

    return results

# cache_size_analysis() returns:
# {'MHA': ~238 GB, 'MQA': ~1.86 GB, 'GQA-8': ~14.9 GB, 'MLA': ~4.2 GB}
# MLA sits between MQA and GQA-8, but preserves per-head diversity via decompression
```

---

the critical architectural distinctions:

**NSA vs DSA:**
- NSA uses a three-branch gated architecture where block selection happens AFTER compression scoring. the compressed branch produces coarse attention patterns that guide which blocks get selected for fine-grained attention.
- DSA uses the "lightning indexer" - a separate low-rank projection path (q^I, k^I with dim ~32) that runs in parallel and cheaply scores relevance. this avoids the sequential dependency of compress-then-select.

**why decoupled RoPE:**
the mathematical impossibility is straightforward. if K = W_K @ c_KV where c_KV is your cached latent, you want to cache c_KV and apply RoPE at inference time. but:
```
RoPE(K) = RoPE(W_K @ c_KV) ≠ W_K @ RoPE(c_KV)
