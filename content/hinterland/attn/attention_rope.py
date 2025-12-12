import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """shared config for all attention variants"""
    d_model: int = 2048      # hidden dim
    n_heads: int = 32        # query heads
    d_head: int = 64         # per-head dim (d_model // n_heads typically)
    max_seq_len: int = 8192

    # MLA specific
    d_c: int = 512           # KV compression dim (latent)
    d_c_q: int = 1536        # Q compression dim
    d_rope: int = 64         # decoupled rope dim

    # GQA specific
    n_kv_heads: int = 8      # KV heads for GQA (n_heads // n_kv_heads = groups)

    # NSA specific
    block_size: int = 64     # compression block size
    n_selected: int = 16     # top-k blocks to select
    window_size: int = 512   # sliding window


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    apply rotary position embedding
    x: (batch, seq, heads, dim) or (batch, seq, dim)
    freqs: (seq, dim//2) precomputed frequencies

    the key insight: this is a position-dependent rotation R_θ(t)
    for dimension pairs (x_i, x_{i+1}), we apply:
    [cos(θt)  -sin(θt)] [x_i    ]
    [sin(θt)   cos(θt)] [x_{i+1}]
    """
    if x.dim() == 3:
        x = x.unsqueeze(2)
        squeeze_back = True
    else:
        squeeze_back = False

    # split into pairs
    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)

    # get sin/cos from freqs
    seq_len = x.shape[1]
    freqs = freqs[:seq_len]
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim//2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)

    # rotate pairs
    x0, x1 = x_reshape[..., 0], x_reshape[..., 1]
    out = torch.stack([
        x0 * cos - x1 * sin,
        x0 * sin + x1 * cos
    ], dim=-1)

    out = out.flatten(-2).type_as(x)
    if squeeze_back:
        out = out.squeeze(2)
    return out


def precompute_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """precompute RoPE frequencies"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return freqs


# vanilla MHA with RoPE
# KV cache per token: 2 * n_heads * d_head = 2 * 32 * 64 = 4096 floats
class MHA(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head

        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        self.register_buffer('freqs', precompute_freqs(cfg.d_head, cfg.max_seq_len))

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple] = None):
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head)

        # apply RoPE to q and k
        q = apply_rope(q, self.freqs)
        k = apply_rope(k, self.freqs)

        # concat with cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        # attention
        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out), (k.transpose(1, 2), v.transpose(1, 2))

    @staticmethod
    def kv_cache_size(cfg: AttentionConfig) -> int:
        """floats per token in KV cache"""
        return 2 * cfg.n_heads * cfg.d_head


# MQA: single KV head shared across all query heads
#
# KV cache per token: 2 * 1 * d_head = 2 * 64 = 128 floats
# 32x reduction from MHA
#
# tradeoff: less expressive KV representation
class MQA(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head

        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_head, bias=False)  # single head
        self.W_v = nn.Linear(cfg.d_model, cfg.d_head, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        self.register_buffer('freqs', precompute_freqs(cfg.d_head, cfg.max_seq_len))

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple] = None):
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.W_k(x).view(B, T, 1, self.d_head)
        v = self.W_v(x).view(B, T, 1, self.d_head)

        q = apply_rope(q, self.freqs)
        k = apply_rope(k, self.freqs)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        # broadcast k, v across heads
        k = k.expand(-1, -1, self.n_heads, -1)
        v = v.expand(-1, -1, self.n_heads, -1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out), (k[:, :, :1].transpose(1, 2), v[:, :, :1].transpose(1, 2))

    @staticmethod
    def kv_cache_size(cfg: AttentionConfig) -> int:
        return 2 * cfg.d_head  # single head


# GQA: groups of query heads share KV heads
#
# KV cache per token: 2 * n_kv_heads * d_head = 2 * 8 * 64 = 1024 floats
# 4x reduction from MHA (with 8 kv heads for 32 q heads)
#
# interpolates between MHA (n_kv_heads = n_heads) and MQA (n_kv_heads = 1)
class GQA(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # heads per kv head

        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        self.register_buffer('freqs', precompute_freqs(cfg.d_head, cfg.max_seq_len))

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple] = None):
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head)

        q = apply_rope(q, self.freqs)
        k = apply_rope(k, self.freqs)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        # repeat kv heads to match query heads
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, -1)
        # save only the original kv heads
        k_out = k[:, ::self.n_rep].transpose(1, 2)
        v_out = v[:, ::self.n_rep].transpose(1, 2)
        return self.W_o(out), (k_out, v_out)

    @staticmethod
    def kv_cache_size(cfg: AttentionConfig) -> int:
        return 2 * cfg.n_kv_heads * cfg.d_head


# MLA: compress KV into low-rank latent, use decoupled RoPE
#
# KV cache per token: d_c + d_rope = 512 + 64 = 576 floats
# ~7x reduction from MHA while maintaining full expressivity
#
# KEY INSIGHT: why decoupled rope?
#
# if we applied RoPE to the content keys:
#     k_t = R_θ(t) @ W^UK @ c_t^KV
#
# the problem is that during inference, we want to absorb W^UK into W^O:
#     out = softmax(Q @ K^T) @ V @ W^O
#         = softmax(Q @ (W^UK @ C)^T) @ W^UV @ C @ W^O
#
# but with RoPE applied to k:
#     K = R_θ @ W^UK @ C
#
# we can't factor out W^UK bc R_θ is position-dependent and doesn't commute.
#
# solution: separate the position encoding into its own small subspace:
#     k = [k_content; k_rope] = [W^UK @ c; R_θ @ W^KR @ h]
#     q = [q_content; q_rope] = [W^UQ @ c_q; R_θ @ W^QR @ c_q]
#
# now:
#     - content part (majority of dims): no RoPE, absorption works
#     - rope part (small d_rope dims): carries position info
#     - total KV cache = c (d_c) + k_rope (d_rope)
class MLA(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_c = cfg.d_c
        self.d_c_q = cfg.d_c_q
        self.d_rope = cfg.d_rope

        # query compression and up-projection
        self.W_dq = nn.Linear(cfg.d_model, cfg.d_c_q, bias=False)  # down-proj query
        self.W_uq = nn.Linear(cfg.d_c_q, cfg.n_heads * cfg.d_head, bias=False)  # up-proj query

        # KV compression (shared latent)
        self.W_dkv = nn.Linear(cfg.d_model, cfg.d_c, bias=False)  # down-proj kv
        self.W_uk = nn.Linear(cfg.d_c, cfg.n_heads * cfg.d_head, bias=False)  # up-proj key
        self.W_uv = nn.Linear(cfg.d_c, cfg.n_heads * cfg.d_head, bias=False)  # up-proj value

        # decoupled RoPE projections (separate small pathway)
        self.W_qr = nn.Linear(cfg.d_c_q, cfg.d_rope, bias=False)  # query rope
        self.W_kr = nn.Linear(cfg.d_model, cfg.d_rope, bias=False)  # key rope

        self.W_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        # rope freqs for the decoupled pathway
        self.register_buffer('freqs', precompute_freqs(cfg.d_rope, cfg.max_seq_len))

        # layernorm for latents (deepseek uses this for stability)
        self.kv_norm = nn.RMSNorm(cfg.d_c)
        self.q_norm = nn.RMSNorm(cfg.d_c_q)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple] = None):
        B, T, _ = x.shape

        # compress query
        c_q = self.q_norm(self.W_dq(x))  # (B, T, d_c_q)

        # content query (no RoPE)
        q_content = self.W_uq(c_q).view(B, T, self.n_heads, self.d_head)

        # rope query (decoupled)
        q_rope = self.W_qr(c_q)  # (B, T, d_rope)
        q_rope = apply_rope(q_rope, self.freqs)  # apply RoPE
        q_rope = q_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1)  # share across heads

        # compress KV into shared latent
        c_kv = self.kv_norm(self.W_dkv(x))  # (B, T, d_c)

        # content key (no RoPE) and value
        k_content = self.W_uk(c_kv).view(B, T, self.n_heads, self.d_head)
        v = self.W_uv(c_kv).view(B, T, self.n_heads, self.d_head)

        # rope key (decoupled, from raw input not latent)
        k_rope = self.W_kr(x)  # (B, T, d_rope)
        k_rope = apply_rope(k_rope, self.freqs)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1)

        # handle cache
        if kv_cache is not None:
            c_kv_cache, k_rope_cache = kv_cache
            c_kv = torch.cat([c_kv_cache, c_kv], dim=1)
            # recompute k_content from cached latent
            k_content = self.W_uk(c_kv).view(B, -1, self.n_heads, self.d_head)
            v = self.W_uv(c_kv).view(B, -1, self.n_heads, self.d_head)
            k_rope = torch.cat([k_rope_cache, k_rope], dim=1)

        # concatenate content and rope for attention
        # q = [q_content, q_rope], k = [k_content, k_rope]
        # but actually we compute attention over the combined representations
        # simpler: add the rope contribution to the attention scores

        q_content = q_content.transpose(1, 2)  # (B, n_heads, T, d_head)
        k_content = k_content.transpose(1, 2)
        v = v.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)  # (B, n_heads, T, d_rope)
        k_rope = k_rope.transpose(1, 2)

        # attention scores = content_scores + rope_scores
        content_scores = q_content @ k_content.transpose(-2, -1)
        rope_scores = q_rope @ k_rope.transpose(-2, -1)

        attn = (content_scores + rope_scores) / math.sqrt(self.d_head + self.d_rope)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, -1)

        # cache is just the latent + rope keys
        return self.W_o(out), (c_kv, k_rope[:, :, 0].transpose(1, 2))  # only need one head's rope

    @staticmethod
    def kv_cache_size(cfg: AttentionConfig) -> int:
        """cache = latent (d_c) + rope_key (d_rope)"""
        return cfg.d_c + cfg.d_rope


# NSA: hierarchical sparse attention with three branches
#
# 1. compression attention: coarse-grained, reduces blocks to single vectors
# 2. selection attention: fine-grained, picks top-k important blocks
# 3. sliding window: local context
#
# this is DIFFERENT from MLA:
# - MLA reduces KV cache SIZE via low-rank compression
# - NSA reduces attention COMPUTE via sparsity patterns
#
# they're orthogonal and can be combined (MLA for cache, NSA for compute)
#
# NSA uses GQA internally, not MLA-style compression
class NSA(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.block_size = cfg.block_size
        self.n_selected = cfg.n_selected
        self.window_size = cfg.window_size
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        # standard GQA projections
        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        # compression pathway: pool blocks into single vectors
        self.compress_k = nn.Linear(cfg.d_head * cfg.block_size, cfg.d_head, bias=False)
        self.compress_v = nn.Linear(cfg.d_head * cfg.block_size, cfg.d_head, bias=False)

        # block importance scoring for selection
        self.importance_score = nn.Linear(cfg.d_head, 1, bias=False)

        # gating to combine three branches
        self.gate = nn.Linear(cfg.d_model, 3, bias=False)

        self.register_buffer('freqs', precompute_freqs(cfg.d_head, cfg.max_seq_len))

    def _compress_attention(self, q, k, v, B, T, S):
        """coarse-grained: compress blocks then attend"""
        # reshape k, v into blocks
        n_blocks = S // self.block_size
        if n_blocks == 0:
            return torch.zeros(B, self.n_heads, T, self.d_head, device=q.device)

        # (B, n_kv_heads, S, d_head) -> (B, n_kv_heads, n_blocks, block_size * d_head)
        k_blocked = k[:, :, :n_blocks * self.block_size].reshape(
            B, self.n_kv_heads, n_blocks, self.block_size * self.d_head
        )
        v_blocked = v[:, :, :n_blocks * self.block_size].reshape(
            B, self.n_kv_heads, n_blocks, self.block_size * self.d_head
        )

        # compress each block to single vector
        k_comp = self.compress_k(k_blocked)  # (B, n_kv_heads, n_blocks, d_head)
        v_comp = self.compress_v(v_blocked)

        # expand for query heads
        k_comp = k_comp.repeat_interleave(self.n_rep, dim=1)
        v_comp = v_comp.repeat_interleave(self.n_rep, dim=1)

        # attend to compressed
        attn = (q @ k_comp.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        return attn @ v_comp

    def _selected_attention(self, q, k, v, B, T, S):
        """fine-grained: select top-k important blocks then full attention"""
        n_blocks = S // self.block_size
        if n_blocks == 0 or n_blocks <= self.n_selected:
            # just do full attention if not enough blocks
            k_exp = k.repeat_interleave(self.n_rep, dim=1)
            v_exp = v.repeat_interleave(self.n_rep, dim=1)
            attn = (q @ k_exp.transpose(-2, -1)) / math.sqrt(self.d_head)
            attn = F.softmax(attn, dim=-1)
            return attn @ v_exp

        # compute block importance scores
        k_blocked = k[:, :, :n_blocks * self.block_size].reshape(
            B, self.n_kv_heads, n_blocks, self.block_size, self.d_head
        )
        # mean pool for scoring
        k_mean = k_blocked.mean(dim=3)  # (B, n_kv_heads, n_blocks, d_head)
        scores = self.importance_score(k_mean).squeeze(-1)  # (B, n_kv_heads, n_blocks)

        # select top-k blocks
        _, top_idx = scores.topk(self.n_selected, dim=-1)  # (B, n_kv_heads, n_selected)

        # gather selected blocks (simplified - in practice this is more complex)
        # for demo, just use full attention with masking
        k_exp = k.repeat_interleave(self.n_rep, dim=1)
        v_exp = v.repeat_interleave(self.n_rep, dim=1)
        attn = (q @ k_exp.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        return attn @ v_exp

    def _sliding_attention(self, q, k, v, B, T, S):
        """local: sliding window attention"""
        k_exp = k.repeat_interleave(self.n_rep, dim=1)
        v_exp = v.repeat_interleave(self.n_rep, dim=1)

        attn = (q @ k_exp.transpose(-2, -1)) / math.sqrt(self.d_head)

        # create sliding window mask
        # each query attends to window_size keys before it
        mask = torch.ones(T, S, device=q.device, dtype=torch.bool)
        for i in range(T):
            start = max(0, S - T + i - self.window_size + 1)
            end = S - T + i + 1
            mask[i, start:end] = False

        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0)  # handle all-masked rows
        return attn @ v_exp

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple] = None):
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head)

        q = apply_rope(q, self.freqs)
        k = apply_rope(k, self.freqs)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)  # (B, n_kv_heads, S, d_head)
        v = v.transpose(1, 2)

        S = k.shape[2]  # total sequence length with cache

        # three branches
        out_compress = self._compress_attention(q, k, v, B, T, S)
        out_select = self._selected_attention(q, k, v, B, T, S)
        out_sliding = self._sliding_attention(q, k, v, B, T, S)

        # learned gating
        gates = F.softmax(self.gate(x), dim=-1)  # (B, T, 3)
        gates = gates.unsqueeze(2).unsqueeze(-1)  # (B, T, 1, 3, 1)

        # reshape outputs for gating
        out_compress = out_compress.transpose(1, 2)  # (B, T, n_heads, d_head)
        out_select = out_select.transpose(1, 2)
        out_sliding = out_sliding.transpose(1, 2)

        stacked = torch.stack([out_compress, out_select, out_sliding], dim=3)
        out = (stacked * gates).sum(dim=3)  # (B, T, n_heads, d_head)

        out = out.reshape(B, T, -1)
        return self.W_o(out), (k.transpose(1, 2), v.transpose(1, 2))

    @staticmethod
    def kv_cache_size(cfg: AttentionConfig) -> int:
        """NSA uses GQA cache: 2 * n_kv_heads * d_head"""
        return 2 * cfg.n_kv_heads * cfg.d_head


def print_kv_cache_comparison(cfg: AttentionConfig):
    """compare KV cache sizes across all mechanisms"""
    mechanisms = [
        ("MHA", MHA.kv_cache_size(cfg)),
        ("MQA", MQA.kv_cache_size(cfg)),
        ("GQA", GQA.kv_cache_size(cfg)),
        ("MLA", MLA.kv_cache_size(cfg)),
        ("NSA (GQA)", NSA.kv_cache_size(cfg)),
    ]

    mha_size = mechanisms[0][1]

    print("\n" + "="*60)
    print("KV cache size comparison (floats per token)")
    print("="*60)
    print(f"config: d_model={cfg.d_model}, n_heads={cfg.n_heads}, d_head={cfg.d_head}")
    print(f"        n_kv_heads={cfg.n_kv_heads}, d_c={cfg.d_c}, d_rope={cfg.d_rope}")
    print("-"*60)

    for name, size in mechanisms:
        ratio = mha_size / size
        print(f"{name:12s}: {size:6d} floats  ({ratio:5.1f}x reduction vs MHA)")

    print("-"*60)
    print("\nfor 128k context, batch=1, fp16:")
    for name, size in mechanisms:
        gb = (128 * 1024 * size * 2) / (1024**3)
        print(f"{name:12s}: {gb:6.2f} GB")


def demo_forward_pass(cfg: AttentionConfig):
    """run forward pass through all attention mechanisms"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*60)
    print(f"forward pass demo (device={device})")
    print("="*60)

    B, T = 2, 128
    x = torch.randn(B, T, cfg.d_model, device=device)

    mechanisms = [
        ("MHA", MHA(cfg).to(device)),
        ("MQA", MQA(cfg).to(device)),
        ("GQA", GQA(cfg).to(device)),
        ("MLA", MLA(cfg).to(device)),
        ("NSA", NSA(cfg).to(device)),
    ]

    for name, model in mechanisms:
        out, cache = model(x)
        if isinstance(cache[0], torch.Tensor):
            cache_shape = [c.shape for c in cache]
        else:
            cache_shape = "special"
        print(f"{name:4s}: output shape={tuple(out.shape)}, cache shapes={cache_shape}")


def explain_decoupled_rope():
    """explain why decoupled RoPE is necessary for MLA"""
    explanation = """
    ============================================================
    why decoupled RoPE in MLA?
    ============================================================

    the problem:
    ------------
    in standard attention with RoPE:
        q_t = R_θ(t) @ W_Q @ h_t
        k_t = R_θ(t) @ W_K @ h_t

    where R_θ(t) is the position-dependent rotation matrix.

    in MLA, we compress KV into latent c_t:
        c_t = W_down @ h_t        (compress to d_c dims)
        k_t = W_up @ c_t          (expand back)

    if we naively apply RoPE:
        k_t = R_θ(t) @ W_up @ c_t

    the absorption problem:
    -----------------------
    during inference, we want to precompute W_absorbed = W_up @ W_out
    and cache only c_t (the small latent).

    but: R_θ(t) @ W_up ≠ W_up @ R_θ(t)

    the rotation matrix doesn't commute with the up-projection!
    this breaks the ability to absorb matrices and defeats the
    purpose of caching the small latent.

    the solution:
    -------------
    decouple position encoding into a SEPARATE small subspace:

        k_content = W_up @ c_t                    (no RoPE, d_head dims)
        k_rope = R_θ(t) @ W_kr @ h_t              (with RoPE, d_rope dims)
        k = [k_content; k_rope]                   (concatenate)

    similarly for queries:
        q_content = W_uq @ c_q                    (no RoPE)
        q_rope = R_θ(t) @ W_qr @ c_q              (with RoPE)

    now:
    - content part: linear in latent, absorption works!
    - rope part: small (64 dims vs 2048 total), carries position info
    - cache = c_t (512 dims) + k_rope (64 dims) = 576 floats

    this is 7x smaller than MHA's 4096 floats while retaining full
    attention expressivity AND position awareness.

    ============================================================
    """
    print(explanation)


if __name__ == "__main__":
    cfg = AttentionConfig()

    explain_decoupled_rope()
    print_kv_cache_comparison(cfg)
    demo_forward_pass(cfg)

    print("\n" + "="*60)
    print("key differences summary")
    print("="*60)
    print("""
    MHA:  full KV per head, RoPE on everything
          ↓
    MQA:  single KV head, shared across queries (32x cache reduction)
          ↓
    GQA:  grouped KV heads, interpolates MHA↔MQA (4x reduction typically)
          ↓
    MLA:  compress KV to low-rank latent, decoupled RoPE (7x reduction)
          maintains full expressivity via up-projection at inference

    NSA:  orthogonal to above - reduces COMPUTE not CACHE
          hierarchical sparsity: compress + select + sliding window
          can be combined with GQA (as deepseek does) or potentially MLA
    """)
