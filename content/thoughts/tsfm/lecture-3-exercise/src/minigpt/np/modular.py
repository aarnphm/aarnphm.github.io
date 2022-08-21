from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import typing as t

if t.TYPE_CHECKING:
  _ArrayFloat32_co = npt.NDArray[np.float32]


def build_causal_mask(Sq, Sk):
  return np.tril(np.ones((Sq, Sk), dtype=bool), 0)


def matmul_bwd(
  A,
  B,
  dY,
) -> tuple[_ArrayFloat32_co, _ArrayFloat32_co]:
  # dA, dB
  return dY @ B.T, A.T @ dY


def input_embedding(
  x,
  W_E,
) -> _ArrayFloat32_co:
  return x @ W_E


def input_embedding_bwd(
  x,
  W_E,
  d_out,
) -> tuple[_ArrayFloat32_co, _ArrayFloat32_co]:
  return matmul_bwd(x, W_E, d_out)


def relu(x) -> _ArrayFloat32_co:
  return np.maximum(0, x)


def gelu(x) -> _ArrayFloat32_co:
  x = x.astype(np.float32, copy=False)
  k = np.float32(np.sqrt(2.0 / np.pi))
  return 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * (x**3))))


def gelu_bwd(x, d_out) -> _ArrayFloat32_co:
  x = x.astype(np.float32, copy=False)
  k = np.float32(np.sqrt(2.0 / np.pi))
  u = k * (x + 0.044715 * (x**3))
  t = np.tanh(u)
  du_dx = k * (1.0 + 3.0 * 0.044715 * (x**2))
  dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
  return d_out * dgelu_dx


def layer_norm(
  x,
  gamma,
  beta,
  eps=1e-6,
) -> _ArrayFloat32_co:
  mu = x.mean(axis=-1, keepdims=True)
  var = x.var(axis=-1, keepdims=True)
  return gamma * (x - mu) / np.sqrt(var + 1e-6) + beta


def layer_norm_bwd(
  x,
  gamma,
  beta,
  d_out,
  eps=1e-6,
) -> tuple[_ArrayFloat32_co, _ArrayFloat32_co, _ArrayFloat32_co]:
  D = x.shape[-1]
  mu = x.mean(axis=-1, keepdims=True)
  x_mu = x - mu
  var = (x_mu**2).mean(axis=-1, keepdims=True)
  inv_std = 1.0 / np.sqrt(var + eps)
  x_hat = x_mu * inv_std

  d_gamma = np.sum(d_out * x_hat, axis=tuple(range(x.ndim - 1)))
  d_beta = np.sum(d_out, axis=tuple(range(x.ndim - 1)))

  d_X_hat = d_out * gamma
  d_var = np.sum(d_X_hat * x_mu * (-0.5) * inv_std**3, axis=-1, keepdims=True)
  d_mu = (
    np.sum(-d_X_hat * inv_std, axis=-1, keepdims=True)
    + d_var * np.sum(-2.0 * x_mu, axis=-1, keepdims=True) / D
  )
  d_X = d_X_hat * inv_std + d_var * 2.0 * x_mu / D + d_mu / D
  return d_X, d_gamma, d_beta


def softmax(
  x,
  axis=-1,
) -> _ArrayFloat32_co:
  ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return ex / np.sum(ex, axis=axis, keepdims=True)


def softmax_bwd_from_probs(
  p,
  grad_out,
  axis=-1,
) -> _ArrayFloat32_co:
  # dL/dz = p ⊙ (g - ⟨g, p⟩), where g = dL/dp
  dot = np.sum(grad_out * p, axis=axis, keepdims=True)
  return p * (grad_out - dot)


def softmax_bwd(
  scores,
  grad_out,
  axis: int = -1,
) -> _ArrayFloat32_co:
  return softmax_bwd_from_probs(softmax(scores, axis=axis), grad_out)


def qkv_proj(
  x,
  W_Q,
  W_K,
  W_V,
):
  q = x @ W_Q
  k = x @ W_K
  v = x @ W_V
  return q, k, v


def qkv_proj_fwd_cached(
  x,
  W_Q,
  W_K,
  W_V,
) -> tuple[
  tuple[_ArrayFloat32_co, _ArrayFloat32_co, _ArrayFloat32_co],
  tuple,
]:
  q, k, v = qkv_proj(x, W_Q, W_K, W_V)
  cache = (x, W_Q, W_K, W_V)
  return (q, k, v), cache


def qkv_proj_bwd_from_cache(
  d_Q,
  d_K,
  d_V,
  cache: tuple,
) -> tuple[
  _ArrayFloat32_co,
  _ArrayFloat32_co,
  _ArrayFloat32_co,
  _ArrayFloat32_co,
]:
  x, W_Q, W_K, W_V = cache
  D = x.shape[-1]
  D_q = W_Q.shape[1]
  x2d = x.reshape(-1, D)
  d_Q_2d = d_Q.reshape(-1, D_q)
  d_K_2d = d_K.reshape(-1, D_q)
  d_V_2d = d_V.reshape(-1, D_q)
  d_X_q, d_W_Q = matmul_bwd(x2d, W_Q, d_Q_2d)
  d_X_k, d_W_K = matmul_bwd(x2d, W_K, d_K_2d)
  d_X_v, d_W_V = matmul_bwd(x2d, W_V, d_V_2d)
  d_X = (d_X_q + d_X_k + d_X_v).reshape(x.shape)
  return d_X, d_W_Q, d_W_K, d_W_V


def qkv_proj_bwd(
  x,
  W_Q,
  W_K,
  W_V,
  d_out,
):
  orig_shape = x.shape
  D, D_q = W_Q.shape[0], W_Q.shape[1]

  x2d, d_out_2d = x.reshape(-1, D), d_out.reshape(-1, D_q)
  d_X_q, d_W_Q = matmul_bwd(x2d, W_Q, d_out_2d)
  d_X_k, d_W_K = matmul_bwd(x2d, W_K, d_out_2d)
  d_X_v, d_W_V = matmul_bwd(x2d, W_V, d_out_2d)
  d_X = (d_X_q + d_X_k + d_X_v).reshape(orig_shape)
  return d_X, d_W_Q, d_W_K, d_W_V


def mha_fwd(
  q,
  k,
  v,
  *,
  causal=False,
  attn_mask=None,
) -> _ArrayFloat32_co:
  scale = 1.0 / np.sqrt(q.shape[-1])
  scores = q @ np.swapaxes(k, -2, -1) * scale  # (..., S_q, S_k)

  # broadcastable masks where we need to enforce only causality generations
  # i.e: newly generated tokens must not affect previous tokens distribution.
  if attn_mask is None and causal:
    s = scores.shape[-1]
    attn_mask = build_causal_mask(s, s)

  if attn_mask is not None:
    if scores.ndim == 4:
      scores = np.where(attn_mask[None, None, :, :], scores, float('-inf'))
    else:
      scores = np.where(attn_mask, scores, float('-inf'))

  # (..., S_q, S_k) -> (..., S_q, D_h)
  attn = softmax(scores)
  return attn @ v


def mha_bwd(
  q,
  k,
  v,
  d_out,
  *,
  causal=False,
  attn_mask=None,
) -> tuple[_ArrayFloat32_co, _ArrayFloat32_co, _ArrayFloat32_co]:
  d_h = q.shape[-1]
  scale = 1.0 / np.sqrt(d_h)

  scores = q @ np.swapaxes(k, -2, -1) * scale

  # broadcastable masks where we need to enforce only causality generations
  # i.e: newly generated tokens must not affect previous tokens distribution.
  if attn_mask is None and causal:
    s = scores.shape[-1]
    attn_mask = build_causal_mask(s, s)

  if attn_mask is not None:
    if scores.ndim == 4:
      scores = np.where(attn_mask[None, None, :, :], scores, float('-inf'))
    else:
      scores = np.where(attn_mask, scores, float('-inf'))

  attn = softmax(scores)  # (..., S_q, S_k)
  d_v = np.swapaxes(attn, -2, -1) @ d_out  # (..., S_k, D_h)
  d_attn = d_out @ np.swapaxes(v, -2, -1)  # (..., S_q, S_k)
  d_scores = softmax_bwd_from_probs(attn, d_attn)

  if attn_mask is not None:
    if d_scores.ndim == 4:
      d_scores = np.where(attn_mask[None, None, :, :], d_scores, 0.0)
    else:
      d_scores = np.where(attn_mask, d_scores, 0.0)

  d_q = d_scores @ k * scale
  d_k = np.swapaxes(d_scores, -2, -1) @ q * scale
  return d_q, d_k, d_v


def mha_fwd_cached(
  q,
  k,
  v,
  *,
  causal=False,
  attn_mask=None,
) -> tuple[_ArrayFloat32_co, dict]:
  d_h = q.shape[-1]
  scale = 1.0 / np.sqrt(d_h)
  scores = q @ np.swapaxes(k, -2, -1) * scale
  # broadcastable masks
  if attn_mask is None and causal:
    s = scores.shape[-1]
    attn_mask = build_causal_mask(s, s)
  if attn_mask is not None:
    if scores.ndim == 4:
      scores = np.where(attn_mask[None, None, :, :], scores, float('-inf'))
    else:
      scores = np.where(attn_mask, scores, float('-inf'))
  attn = softmax(scores)
  out = attn @ v
  cache = {
    'q': q,
    'k': k,
    'v': v,
    'attn': attn,
    'attn_mask': attn_mask,
    'scale': scale,
  }
  return out, cache


def mha_bwd_from_cache(
  d_out,
  cache,
) -> tuple[_ArrayFloat32_co, _ArrayFloat32_co, _ArrayFloat32_co]:
  q, k, v = cache['q'], cache['k'], cache['v']
  attn, attn_mask = cache['attn'], cache['attn_mask']
  scale = cache['scale']

  d_v = np.swapaxes(attn, -2, -1) @ d_out
  d_attn = d_out @ np.swapaxes(v, -2, -1)
  d_scores = softmax_bwd_from_probs(attn, d_attn)
  if attn_mask is not None:
    if d_scores.ndim == 4:
      d_scores = np.where(attn_mask[None, None, :, :], d_scores, 0.0)
    else:
      d_scores = np.where(attn_mask, d_scores, 0.0)
  d_q = d_scores @ k * scale
  d_k = np.swapaxes(d_scores, -2, -1) @ q * scale
  return d_q, d_k, d_v


def ffn(
  x,
  w_1,
  w_2,
):
  return gelu(x @ w_1) @ w_2


def ffn_bwd(
  x,
  w_1,
  w_2,
  d_out,
) -> tuple[_ArrayFloat32_co, _ArrayFloat32_co, _ArrayFloat32_co]:
  b, s, d_model = x.shape
  x2d = x.reshape(-1, d_model)

  z1 = x2d @ w_1
  a1 = gelu(z1)
  d_a1, d_w_2 = matmul_bwd(a1, w_2, d_out.reshape(-1, d_model))
  d_z1 = gelu_bwd(z1, np.ones_like(z1)) * d_a1
  d_X2d, d_w_1 = matmul_bwd(x2d, w_1, d_z1)

  return d_X2d.reshape(b, s, d_model), d_w_1, d_w_2


def ffn_cached(
  x,
  w_1,
  w_2,
) -> tuple[_ArrayFloat32_co, tuple]:
  y = ffn(x, w_1, w_2)
  cache = (x, w_1, w_2)
  return y, cache


def ffn_bwd_from_cache(
  d_y, cache: tuple
) -> tuple[_ArrayFloat32_co, _ArrayFloat32_co, _ArrayFloat32_co]:
  x, w_1, w_2 = cache
  return ffn_bwd(x, w_1, w_2, d_y)


def layer_norm_cached(
  x,
  gamma,
  beta,
) -> tuple[_ArrayFloat32_co, tuple]:
  y = layer_norm(x, gamma, beta)
  return y, (x, gamma, beta)


def layer_norm_bwd_from_cache(
  d_y,
  cache,
) -> tuple[_ArrayFloat32_co, _ArrayFloat32_co, _ArrayFloat32_co]:
  x, gamma, beta = cache
  return layer_norm_bwd(x, gamma, beta, d_y)


def block_fwd(
  x,
  w_q,
  w_k,
  w_v,
  w_o,
  w_ff_expand,
  w_ff_contract,
  gamma,
  beta,
  *,
  causal=False,
  attn_mask=None,
):
  q, k, v = qkv_proj(x, w_q, w_k, w_v)
  attn_o = mha_fwd(q, k, v, causal=causal, attn_mask=attn_mask) @ w_o
  ln1 = layer_norm(x + attn_o, gamma, beta)
  ff = ffn(ln1, w_ff_expand, w_ff_contract)
  return layer_norm(ln1 + ff, gamma, beta)


def block_bwd(
  x,
  w_q,
  w_k,
  w_v,
  w_o,
  w_ff_expand,
  w_ff_contract,
  gamma,
  beta,
  d_out,
  *,
  causal=False,
  attn_mask=None,
):
  q, k, v = qkv_proj(x, w_q, w_k, w_v)
  attn_pre = mha_fwd(q, k, v, causal=causal, attn_mask=attn_mask)
  attn_proj = attn_pre @ w_o

  res1 = x + attn_proj
  ln1 = layer_norm(res1, gamma, beta)

  ff = ffn(ln1, w_ff_expand, w_ff_contract)
  res2 = ln1 + ff

  d_res2, d_g2, d_b2 = layer_norm_bwd(res2, gamma, beta, d_out)

  d_ln1_ff, d_w1, d_w2 = ffn_bwd(ln1, w_ff_expand, w_ff_contract, d_res2)
  d_res1, d_g1, d_b1 = layer_norm_bwd(res1, gamma, beta, d_ln1_ff + d_res2)

  b, s, d_model = x.shape
  d_qkv = attn_pre.shape[-1]

  d_attn_pre_2d, d_w_o = matmul_bwd(
    attn_pre.reshape(-1, d_qkv), w_o, d_res1.reshape(-1, d_model)
  )
  d_attn_pre = d_attn_pre_2d.reshape(b, s, d_qkv)

  # Support both 3D and 4D attention tensors
  if q.ndim == 3:
    d_q4, d_k4, d_v4 = mha_bwd(
      q[:, None],
      k[:, None],
      v[:, None],
      d_attn_pre[:, None],
      causal=causal,
      attn_mask=attn_mask,
    )
    d_q, d_k, d_v = d_q4[:, 0], d_k4[:, 0], d_v4[:, 0]
  else:
    d_q, d_k, d_v = mha_bwd(
      q, k, v, d_attn_pre, causal=causal, attn_mask=attn_mask
    )

  x2d = x.reshape(-1, d_model)
  d_X_q, d_W_Q = matmul_bwd(x2d, w_q, d_q.reshape(-1, d_qkv))
  d_X_k, d_W_K = matmul_bwd(x2d, w_k, d_k.reshape(-1, d_qkv))
  d_X_v, d_W_V = matmul_bwd(x2d, w_v, d_v.reshape(-1, d_qkv))

  d_X = d_res1 + (d_X_q + d_X_k + d_X_v).reshape(b, s, d_model)
  return (
    d_X,
    d_W_Q,
    d_W_K,
    d_W_V,
    d_w_o,
    d_w1,
    d_w2,
    (d_g1 + d_g2),
    (d_b1 + d_b2),
  )


# ---------------------------
# Torch version of components
# ---------------------------


def torch_matmul_bwd(a_np, b_np, d_out_np):
  a_t = torch.from_numpy(a_np).float().requires_grad_(True)
  b_t = torch.from_numpy(b_np).float().requires_grad_(True)
  d_out_t = torch.from_numpy(d_out_np).float()

  output = a_t @ b_t
  output.backward(d_out_t)

  return a_t.grad.detach().numpy(), b_t.grad.detach().numpy()


def torch_input_embedding_bwd(x_indices, w_e_np, d_out_np):
  w_e_t = torch.from_numpy(w_e_np).float().requires_grad_(True)
  x_idx_t = torch.from_numpy(x_indices).long()
  d_out_t = torch.from_numpy(d_out_np).float()

  output = F.embedding(x_idx_t, w_e_t)
  output.backward(d_out_t)

  return w_e_t.grad.detach().numpy()


def torch_softmax_bwd(x_np, grad_out_np):
  x_t = torch.from_numpy(x_np).float().requires_grad_(True)
  grad_out_t = torch.from_numpy(grad_out_np).float()

  softmax_output = F.softmax(x_t, dim=-1)
  softmax_output.backward(grad_out_t)

  return x_t.grad.detach().numpy()


def torch_mha_bwd(q_np, k_np, v_np, d_out_np, *, causal=False):
  q_t = torch.from_numpy(q_np).float().requires_grad_(True)
  k_t = torch.from_numpy(k_np).float().requires_grad_(True)
  v_t = torch.from_numpy(v_np).float().requires_grad_(True)
  d_out_t = torch.from_numpy(d_out_np).float()

  # Multi-head attention forward pass
  d_h = q_t.shape[-1]
  attn_scores = q_t @ k_t.transpose(-2, -1) / np.sqrt(d_h)

  if causal:
    s = attn_scores.shape[-1]
    mask = torch.tril(torch.ones((s, s), dtype=torch.bool))
    if attn_scores.dim() == 4:
      mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
    attn_scores = torch.where(
      mask, attn_scores, torch.tensor(1e-9, dtype=attn_scores.dtype)
    )
  attn_weights = F.softmax(attn_scores, dim=-1)
  output = attn_weights @ v_t

  output.backward(d_out_t)

  return (
    q_t.grad.detach().numpy(),
    k_t.grad.detach().numpy(),
    v_t.grad.detach().numpy(),
  )


def torch_qkv_proj_bwd(x_np, w_q_np, w_k_np, w_v_np, d_out_flat):
  batch_size, seq_len, d_model = x_np.shape
  _, d_qkv = w_q_np.shape

  x_t = torch.from_numpy(x_np).float().requires_grad_(True)
  w_q_t = torch.from_numpy(w_q_np).float().requires_grad_(True)
  w_k_t = torch.from_numpy(w_k_np).float().requires_grad_(True)
  w_v_t = torch.from_numpy(w_v_np).float().requires_grad_(True)
  d_out_t = torch.from_numpy(
    d_out_flat.reshape(batch_size, seq_len, d_qkv)
  ).float()

  # QKV projection forward pass
  q_t = x_t @ w_q_t
  k_t = x_t @ w_k_t
  v_t = x_t @ w_v_t

  # Sum outputs since they share the same input x
  d_X_t = q_t + k_t + v_t
  d_X_t.backward(d_out_t)

  return (
    x_t.grad.reshape(-1, d_model).detach().numpy(),
    w_q_t.grad.detach().numpy(),
    w_k_t.grad.detach().numpy(),
    w_v_t.grad.detach().numpy(),
  )


def torch_layer_norm_bwd(x_np, gamma_np, beta_np, d_out_np, eps=1e-6):
  x_t = torch.from_numpy(x_np).float().requires_grad_(True)
  gamma_t = torch.from_numpy(gamma_np).float().requires_grad_(True)
  beta_t = torch.from_numpy(beta_np).float().requires_grad_(True)
  d_out_t = torch.from_numpy(d_out_np).float()

  d_dim = x_np.shape[-1]
  y = F.layer_norm(
    x_t, normalized_shape=(d_dim,), weight=gamma_t, bias=beta_t, eps=eps
  )
  y.backward(d_out_t)

  return (
    x_t.grad.detach().numpy(),
    gamma_t.grad.detach().numpy(),
    beta_t.grad.detach().numpy(),
  )


def torch_ffn_bwd(x_np, w1_np, w2_np, d_out_np):
  x_t = torch.from_numpy(x_np).float().requires_grad_(True)
  w1_t = torch.from_numpy(w1_np).float().requires_grad_(True)
  w2_t = torch.from_numpy(w2_np).float().requires_grad_(True)
  d_out_t = torch.from_numpy(d_out_np).float()

  a1 = F.gelu(x_t @ w1_t, approximate='tanh')
  y = a1 @ w2_t
  y.backward(d_out_t)

  return (
    x_t.grad.detach().numpy(),
    w1_t.grad.detach().numpy(),
    w2_t.grad.detach().numpy(),
  )


def torch_block_bwd(
  x_np,
  w_q_np,
  w_k_np,
  w_v_np,
  w_o_np,
  w_ff_expand_np,
  w_ff_contract_np,
  gamma_np,
  beta_np,
  d_out_np,
  eps=1e-6,
  causal=False,
):
  x_t = torch.from_numpy(x_np).float().requires_grad_(True)
  w_q_t = torch.from_numpy(w_q_np).float().requires_grad_(True)
  w_k_t = torch.from_numpy(w_k_np).float().requires_grad_(True)
  w_v_t = torch.from_numpy(w_v_np).float().requires_grad_(True)
  w_o_t = torch.from_numpy(w_o_np).float().requires_grad_(True)
  w_ff_expand_t = torch.from_numpy(w_ff_expand_np).float().requires_grad_(True)
  w_ff_contract_t = (
    torch.from_numpy(w_ff_contract_np).float().requires_grad_(True)
  )
  gamma_t = torch.from_numpy(gamma_np).float().requires_grad_(True)
  beta_t = torch.from_numpy(beta_np).float().requires_grad_(True)
  d_out_t = torch.from_numpy(d_out_np).float()

  # QKV projections
  q_t = x_t @ w_q_t
  k_t = x_t @ w_k_t
  v_t = x_t @ w_v_t

  # Multi-head attention (batched matmul on 3D tensors)
  d_h = q_t.shape[-1]
  scores = (q_t @ k_t.transpose(-2, -1)) / np.sqrt(d_h)
  if causal:
    s = scores.shape[-1]
    mask = torch.tril(torch.ones((s, s), dtype=torch.bool))
    scores = torch.where(
      mask, scores, torch.tensor(float('-inf'), dtype=scores.dtype)
    )
  weights = torch.softmax(scores, dim=-1)
  attn_out = weights @ v_t

  # Output projection
  attn_proj = attn_out @ w_o_t

  # Residual + LN
  res1 = x_t + attn_proj
  ln1 = F.layer_norm(
    res1,
    normalized_shape=(x_np.shape[-1],),
    weight=gamma_t,
    bias=beta_t,
    eps=eps,
  )

  # FFN with GELU
  ff = F.gelu(ln1 @ w_ff_expand_t, approximate='tanh') @ w_ff_contract_t

  # Residual + LN
  res2 = ln1 + ff
  ln2 = F.layer_norm(
    res2,
    normalized_shape=(x_np.shape[-1],),
    weight=gamma_t,
    bias=beta_t,
    eps=eps,
  )

  # Backprop
  ln2.backward(d_out_t)

  return (
    x_t.grad.detach().numpy(),
    w_q_t.grad.detach().numpy(),
    w_k_t.grad.detach().numpy(),
    w_v_t.grad.detach().numpy(),
    w_o_t.grad.detach().numpy(),
    w_ff_expand_t.grad.detach().numpy(),
    w_ff_contract_t.grad.detach().numpy(),
    gamma_t.grad.detach().numpy(),
    beta_t.grad.detach().numpy(),
  )
