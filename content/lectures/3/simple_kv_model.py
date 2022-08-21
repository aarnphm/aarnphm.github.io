import torch, torch.nn as nn, torch.nn.functional as F

def split_heads(x, n_heads):
    # x: [B, T, D_model] -> [B, H, T, D_head]
    B, T, D = x.shape
    assert D % n_heads == 0
    d_head = D // n_heads
    return x.view(B, T, n_heads, d_head).permute(0, 2, 1, 3).contiguous()

def merge_heads(x):
    # x: [B, H, T, D_head] -> [B, T, D_model]
    B, H, T, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)

def sample_from_logits(logits, temperature=1.0, top_k=None):
    # logits: [B, vocab]
    if temperature != 1.0:
        logits = logits / max(1e-6, temperature)
    if top_k is not None and top_k > 0:
        v, i = torch.topk(logits, k=top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        logits = mask.scatter(-1, i, v)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

class KVCache:
    """
    Per-stream, per-layer KV cache.
    Stores K and V as lists of tensors, one (K,V) per layer.
    Shapes: K,V: [B, H, T, D_head]
    """
    def __init__(self, n_layers: int):
        self.K = [None] * n_layers
        self.V = [None] * n_layers

    def get(self, layer_idx: int):
        return self.K[layer_idx], self.V[layer_idx]

    def set_full(self, layer_idx: int, K_full, V_full):
        # Used during prefill to stash entire prompt K/V
        self.K[layer_idx] = K_full
        self.V[layer_idx] = V_full

    def append_step(self, layer_idx: int, k_new, v_new):
        # k_new, v_new: [B, H, 1, D_head]
        K_prev, V_prev = self.get(layer_idx)
        if K_prev is None:
            self.K[layer_idx] = k_new
            self.V[layer_idx] = v_new
        else:
            self.K[layer_idx] = torch.cat([K_prev, k_new], dim=2)
            self.V[layer_idx] = torch.cat([V_prev, v_new], dim=2)

    # Optional: sliding-window eviction for local attention layers
    def trim_left(self, layer_idx: int, max_T: int):
        K, V = self.get(layer_idx)
        if K is None:
            return
        T = K.shape[2]
        if T > max_T:
            self.K[layer_idx] = K[:, :, T - max_T :, :]
            self.V[layer_idx] = V[:, :, T - max_T :, :]

class TinyMLP(nn.Module):
    # MLP block (define your own if you want something fancier)
    def __init__(self, d_model, hidden_mult=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_mult * d_model)
        self.fc2 = nn.Linear(hidden_mult * d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# 1-L attention decoder block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln_1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = TinyMLP(d_model)

    def attention(self, x, kv_cache: KVCache, layer_idx: int, is_prefill: bool):
        # x: [B, T, D_model]  (T can be full prompt during prefill or 1 during decode)
        # Returns: y: [B, T, D_model], updates kv_cache
        B, T, _ = x.shape

        x_norm = self.ln_1(x)
        Q = split_heads(self.q_proj(x_norm), self.n_heads)  # [B,H,T,Dh]
        K = split_heads(self.k_proj(x_norm), self.n_heads)  # [B,H,T,Dh]
        V = split_heads(self.v_proj(x_norm), self.n_heads)  # [B,H,T,Dh]

        if is_prefill:
            # Store entire prompt K/V once
            kv_cache.set_full(layer_idx, K, V)
            K_all, V_all = K, V
        else:
            # Append only the last step’s K/V (T==1)
            kv_cache.append_step(layer_idx, K[:, :, -1:, :], V[:, :, -1:, :])
            K_all, V_all = kv_cache.get(layer_idx)

        # SDPA does the scaling/softmax internally. Use causal mode.
        # Query can be the entire T (prefill) or T=1 (decode).
        # K_all/V_all are the accumulated caches: [B,H,S,Dh], S>=T
        y = F.scaled_dot_product_attention(
            Q, K_all, V_all,
            attn_mask=None,   # we can add attention mask for tasks such as structured outputs,
                              # but omitted for brevity
            dropout_p=0.0,    # inference
            is_causal=True    # causal self-attention
        )  # -> [B,H,T,Dh]

        y = merge_heads(y)       # [B,T,D_model]
        y = self.attn_out(y)
        y = self.attn_drop(y)
        y = x + y                # Residual
        return y

    def forward(self, x, kv_cache: KVCache, layer_idx: int, is_prefill: bool):
        x = self.attention(x, kv_cache, layer_idx, is_prefill)
        # MLP block (+ residual)
        x_norm = self.ln_2(x)
        x = x + self.mlp(x_norm)
        return x

# 1-L attention model
class OneLayerModelForCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=1, n_heads=12, max_len=8192):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)  # simple absolute pos-emb
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )  # in practice, this is mostly dependent on choices. There are rooflines for how much layers
           # you should include until you reach a certain diminishing returns threshold.
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_step(self, input_ids: torch.Tensor, kv_cache: KVCache, is_prefill: bool):
        # input_ids: [B, T]  (T can be prompt length during prefill, or 1 during decode)
        # T is also known as seq_len
        # returns: logits: [B, T, vocab]
        device = input_ids.device
        B, T = input_ids.shape

        # embeddings
        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # [B,T,D]

        # transformer stack
        for layer_idx, blk in enumerate(self.blocks):
            x = blk(x, kv_cache=kv_cache, layer_idx=layer_idx, is_prefill=is_prefill)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        return logits

@torch.no_grad()
def generate(model: OneLayerModelForCausalLM,
             prompt_ids: torch.Tensor,
             max_new_tokens=128,
             temperature=1.0, top_k=None, sliding_windows=None):
    # AD loop, forward pass
    # prompt_ids: [B, T0]
    # sliding_windows: optional dict {layer_idx: window_size} for local attention layers

    model.eval()
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)

    kv = KVCache(n_layers=model.n_layers)

    # 1) PREFILL: run full prompt once; caches K/V for all layers
    logits = model.forward_step(prompt_ids, kv_cache=kv, is_prefill=True)
    next_token = sample_from_logits(logits[:, -1, :], temperature=temperature, top_k=top_k)  # [B]
    out = [prompt_ids, next_token.unsqueeze(1)]

    # 2) DECODE: feed only last token, reuse kv cache
    cur = next_token.unsqueeze(1)  # [B,1]
    for _ in range(max_new_tokens - 1):
        logits = model.forward_step(cur, kv_cache=kv, is_prefill=False)
        cur = sample_from_logits(logits[:, -1, :], temperature=temperature, top_k=top_k).unsqueeze(1)
        out.append(cur)

        # Optional sliding window eviction per layer
        # ie: Mistral and Gemma of the world.
        if sliding_windows:
            for L, W in sliding_windows.items():
                kv.trim_left(L, max_T=W)

    return torch.cat(out, dim=1)  # token ids: [B, T0 + max_new_tokens]
