from __future__ import annotations

import numpy as np, numpy.typing as npt, torch, torch.nn.functional as F


def build_causal_mask(Sq: int, Sk: int) -> npt.NDArray[np.float32]:
  return np.tril(np.ones((Sq, Sk), dtype=bool), 0)


def matmul_bwd(
  A: npt.NDArray[np.float32], B: npt.NDArray[np.float32],
  dY: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  # dA, dB
  return dY @ B.T, A.T @ dY


def input_embedding(x: npt.NDArray[np.float32],
                    W_E: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
  return x @ W_E


def input_embedding_bwd(
  x: npt.NDArray[np.float32],
  W_E: npt.NDArray[np.float32], dOut: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  return matmul_bwd(x, W_E, dOut)


def relu(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
  return np.maximum(0, x)


def layer_norm(
  x: npt.NDArray[np.float32],
  gamma: npt.NDArray[np.float32], beta: npt.NDArray[np.float32], eps: float = 1e-6
) -> npt.NDArray[np.float32]:
  mu = x.mean(axis=-1, keepdims=True)
  var = x.var(axis=-1, keepdims=True)
  return gamma * (x - mu) / np.sqrt(var + 1e-6) + beta


def layer_norm_bwd(
  x: npt.NDArray[np.float32],
  gamma: npt.NDArray[np.float32], beta: npt.NDArray[np.float32],
  dOut: npt.NDArray[np.float32],
  eps: float = 1e-6,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  D = x.shape[-1]
  mu = x.mean(axis=-1, keepdims=True)
  x_mu = x - mu
  var = (x_mu**2).mean(axis=-1, keepdims=True)
  inv_std = 1.0 / np.sqrt(var + eps)
  x_hat = x_mu * inv_std

  dGamma = np.sum(dOut * x_hat, axis=tuple(range(x.ndim - 1)))
  dBeta = np.sum(dOut, axis=tuple(range(x.ndim - 1)))

  dX_hat = dOut * gamma
  dVar = np.sum(dX_hat * x_mu * (-0.5) * inv_std**3, axis=-1, keepdims=True)
  dMu = np.sum(-dX_hat * inv_std, axis=-1, keepdims=True) + dVar * np.sum(-2.0 * x_mu, axis=-1, keepdims=True) / D
  dX = dX_hat * inv_std + dVar * 2.0 * x_mu / D + dMu / D
  return dX, dGamma, dBeta


def softmax(x: npt.NDArray[np.float32],
            axis: int = -1) -> npt.NDArray[np.float32]:
  x = x - np.max(x, axis=axis, keepdims=True)
  ex = np.exp(x)
  return ex / np.sum(ex, axis=axis, keepdims=True)


def softmax_bwd_from_probs(
  p: npt.NDArray[np.float32],
  grad_out: npt.NDArray[np.float32],
  axis: int = -1
) -> npt.NDArray[np.float32]:
  # dL/dz = p ⊙ (g - ⟨g, p⟩), where g = dL/dp
  dot = np.sum(grad_out * p, axis=axis, keepdims=True)
  return p * (grad_out - dot)


def softmax_bwd(
  scores: npt.NDArray[np.float32],
  grad_out: npt.NDArray[np.float32], axis: int = -1
) -> npt.NDArray[np.float32]:
  return softmax_bwd_from_probs(softmax(scores, axis=axis), grad_out)


def qkv_proj(
  x: npt.NDArray[np.float32],
  W_Q: npt.NDArray[np.float32], W_K: npt.NDArray[np.float32], W_V: npt.NDArray[np.float32]
):
  q = x @ W_Q
  k = x @ W_K
  v = x @ W_V
  return q, k, v


def qkv_proj_fwd_cached(
  x: npt.NDArray[np.float32],
  W_Q: npt.NDArray[np.float32], W_K: npt.NDArray[np.float32], W_V: npt.NDArray[np.float32]
) -> tuple[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]], tuple]:
  q, k, v = qkv_proj(x, W_Q, W_K, W_V)
  cache = (x, W_Q, W_K, W_V)
  return (q, k, v), cache


def qkv_proj_bwd_from_cache(
  dQ: npt.NDArray[np.float32], dK: npt.NDArray[np.float32], dV: npt.NDArray[np.float32],
  cache: tuple
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  x, W_Q, W_K, W_V = cache
  D = x.shape[-1]
  Dq = W_Q.shape[1]
  X2D = x.reshape(-1, D)
  dQ2D = dQ.reshape(-1, Dq)
  dK2D = dK.reshape(-1, Dq)
  dV2D = dV.reshape(-1, Dq)
  dXq, dW_Q = matmul_bwd(X2D, W_Q, dQ2D)
  dXk, dW_K = matmul_bwd(X2D, W_K, dK2D)
  dXv, dW_V = matmul_bwd(X2D, W_V, dV2D)
  dX = (dXq + dXk + dXv).reshape(x.shape)
  return dX, dW_Q, dW_K, dW_V


def qkv_proj_bwd(
  x: npt.NDArray[np.float32],
  W_Q: npt.NDArray[np.float32], W_K: npt.NDArray[np.float32], W_V: npt.NDArray[np.float32],
  dOut: npt.NDArray[np.float32],
):
  orig_shape = x.shape
  D, Dq = W_Q.shape[0], W_Q.shape[1]

  X2D, dOut2D = x.reshape(-1, D), dOut.reshape(-1, Dq)
  dXq, dW_Q = matmul_bwd(X2D, W_Q, dOut2D)
  dXk, dW_K = matmul_bwd(X2D, W_K, dOut2D)
  dXv, dW_V = matmul_bwd(X2D, W_V, dOut2D)
  dX = (dXq + dXk + dXv).reshape(orig_shape)
  return dX, dW_Q, dW_K, dW_V


def mha_fwd(
  q: npt.NDArray[np.float32], k: npt.NDArray[np.float32], v: npt.NDArray[np.float32],
  *,
  causal: bool = False,
  attn_mask: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.float32]:
  scale = 1.0 / np.sqrt(Dh := q.shape[-1])
  scores = q @ np.swapaxes(k, -2, -1) * scale  # (..., S_q, S_k)

  # broadcastable masks where we need to enforce only causality generations
  # i.e: newly generated tokens must not affect previous tokens distribution.
  if attn_mask is None and causal:
    S = scores.shape[-1]
    attn_mask = build_causal_mask(S, S)

  if attn_mask is not None:
    if scores.ndim == 4:
      scores = np.where(attn_mask[None, None, :, :], scores, float('-inf'))
    else:
      scores = np.where(attn_mask, scores, float('-inf'))

  # (..., S_q, S_k) -> (..., S_q, D_h)
  attn = softmax(scores)
  return attn @ v


def mha_bwd(
  q: npt.NDArray[np.float32], k: npt.NDArray[np.float32], v: npt.NDArray[np.float32],
  dOut: npt.NDArray[np.float32],
  *,
  causal: bool = False,
  attn_mask: npt.NDArray[np.float32] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  Dh = q.shape[-1]
  scale = 1.0 / np.sqrt(Dh)

  scores = q @ np.swapaxes(k, -2, -1) * scale

  # broadcastable masks where we need to enforce only causality generations
  # i.e: newly generated tokens must not affect previous tokens distribution.
  if attn_mask is None and causal:
    S = scores.shape[-1]
    attn_mask = build_causal_mask(S, S)

  if attn_mask is not None:
    if scores.ndim == 4:
      scores = np.where(attn_mask[None, None, :, :], scores, float('-inf'))
    else:
      scores = np.where(attn_mask, scores, float('-inf'))

  attn = softmax(scores)  # (..., S_q, S_k)
  dV = np.swapaxes(attn, -2, -1) @ dOut  # (..., S_k, D_h)
  dAttn = dOut @ np.swapaxes(v, -2, -1)  # (..., S_q, S_k)
  dScores = softmax_bwd_from_probs(attn, dAttn)

  if attn_mask is not None:
    if dScores.ndim == 4:
      dScores = np.where(attn_mask[None, None, :, :], dScores, 0.0)
    else:
      dScores = np.where(attn_mask, dScores, 0.0)

  dQ = dScores @ k * scale
  dK = np.swapaxes(dScores, -2, -1) @ q * scale
  return dQ, dK, dV


def mha_fwd_cached(
  q: npt.NDArray[np.float32],
  k: npt.NDArray[np.float32],
  v: npt.NDArray[np.float32],
  *,
  causal: bool = False,
  attn_mask: npt.NDArray[np.float32] | None = None,
) -> tuple[npt.NDArray[np.float32], dict]:
  Dh = q.shape[-1]
  scale = 1.0 / np.sqrt(Dh)
  scores = q @ np.swapaxes(k, -2, -1) * scale
  # broadcastable masks
  if attn_mask is None and causal:
    S = scores.shape[-1]
    attn_mask = build_causal_mask(S, S)
  if attn_mask is not None:
    if scores.ndim == 4:
      scores = np.where(attn_mask[None, None, :, :], scores, float('-inf'))
    else:
      scores = np.where(attn_mask, scores, float('-inf'))
  attn = softmax(scores)
  out = attn @ v
  cache = {'q': q, 'k': k, 'v': v, 'attn': attn, 'attn_mask': attn_mask, 'scale': scale}
  return out, cache


def mha_bwd_from_cache(
  dOut: npt.NDArray[np.float32], cache: dict
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  q, k, v = cache['q'], cache['k'], cache['v']
  attn, attn_mask = cache['attn'], cache['attn_mask']
  scale = cache['scale']

  dV = np.swapaxes(attn, -2, -1) @ dOut
  dAttn = dOut @ np.swapaxes(v, -2, -1)
  dScores = softmax_bwd_from_probs(attn, dAttn)
  if attn_mask is not None:
    if dScores.ndim == 4:
      dScores = np.where(attn_mask[None, None, :, :], dScores, 0.0)
    else:
      dScores = np.where(attn_mask, dScores, 0.0)
  dQ = dScores @ k * scale
  dK = np.swapaxes(dScores, -2, -1) @ q * scale
  return dQ, dK, dV


def ffn(x: npt.NDArray[np.float32], W_1: npt.NDArray[np.float32], W_2: npt.NDArray[np.float32]):
  return relu(x @ W_1) @ W_2


def ffn_bwd(
  x: npt.NDArray[np.float32],
  W_1: npt.NDArray[np.float32], W_2: npt.NDArray[np.float32], dOut: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  B, S, d_model = x.shape
  X2D = x.reshape(-1, d_model)

  Z1 = X2D @ W_1
  A1 = relu(Z1)
  dA1, dW_2 = matmul_bwd(A1, W_2, dOut.reshape(-1, d_model))
  dZ1 = dA1 * (Z1 > 0)
  dX2D, dW_1 = matmul_bwd(X2D, W_1, dZ1)

  return dX2D.reshape(B, S, d_model), dW_1, dW_2


def ffn_cached(
  x: npt.NDArray[np.float32], W_1: npt.NDArray[np.float32], W_2: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], tuple]:
  y = ffn(x, W_1, W_2)
  cache = (x, W_1, W_2)
  return y, cache


def ffn_bwd_from_cache(
  dY: npt.NDArray[np.float32], cache: tuple
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  x, W_1, W_2 = cache
  return ffn_bwd(x, W_1, W_2, dY)


def layer_norm_cached(
  x: npt.NDArray[np.float32], gamma: npt.NDArray[np.float32], beta: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], tuple]:
  y = layer_norm(x, gamma, beta)
  return y, (x, gamma, beta)


def layer_norm_bwd_from_cache(
  dY: npt.NDArray[np.float32], cache: tuple
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
  x, gamma, beta = cache
  return layer_norm_bwd(x, gamma, beta, dY)


def block_fwd(
  x: npt.NDArray[np.float32],
  W_Q: npt.NDArray[np.float32],
  W_K: npt.NDArray[np.float32],
  W_V: npt.NDArray[np.float32],
  W_O: npt.NDArray[np.float32],
  W_FF_expand: npt.NDArray[np.float32],
  W_FF_contract: npt.NDArray[np.float32],
  gamma: npt.NDArray[np.float32],
  beta: npt.NDArray[np.float32],
  *,
  causal: bool = False,
  attn_mask: npt.NDArray[np.float32] | None = None,
):
  q, k, v = qkv_proj(x, W_Q, W_K, W_V)
  attn_o = mha_fwd(q, k, v, causal=causal, attn_mask=attn_mask) @ W_O
  ln1 = layer_norm(x + attn_o, gamma, beta)
  ff = ffn(ln1, W_FF_expand, W_FF_contract)
  return layer_norm(ln1 + ff, gamma, beta)


def block_bwd(
  x: npt.NDArray[np.float32],
  W_Q: npt.NDArray[np.float32],
  W_K: npt.NDArray[np.float32],
  W_V: npt.NDArray[np.float32],
  W_O: npt.NDArray[np.float32],
  W_FF_expand: npt.NDArray[np.float32],
  W_FF_contract: npt.NDArray[np.float32],
  gamma: npt.NDArray[np.float32],
  beta: npt.NDArray[np.float32],
  dOut: npt.NDArray[np.float32],
  *,
  causal: bool = False,
  attn_mask: npt.NDArray[np.float32] | None = None,
):
  q, k, v = qkv_proj(x, W_Q, W_K, W_V)
  attn_pre = mha_fwd(q, k, v, causal=causal, attn_mask=attn_mask)
  attn_proj = attn_pre @ W_O

  res1 = x + attn_proj
  ln1 = layer_norm(res1, gamma, beta)

  ff = ffn(ln1, W_FF_expand, W_FF_contract)
  res2 = ln1 + ff

  dRes2, dG2, dB2 = layer_norm_bwd(res2, gamma, beta, dOut)

  dLn1_ff, dW1, dW2 = ffn_bwd(ln1, W_FF_expand, W_FF_contract, dRes2)
  dRes1, dG1, dB1 = layer_norm_bwd(res1, gamma, beta, dLn1_ff + dRes2)

  B, S, d_model = x.shape
  d_qkv = attn_pre.shape[-1]

  dAttnPre_2d, dW_O = matmul_bwd(attn_pre.reshape(-1, d_qkv), W_O, dRes1.reshape(-1, d_model))
  dAttnPre = dAttnPre_2d.reshape(B, S, d_qkv)

  # Support both 3D and 4D attention tensors
  if q.ndim == 3:
    dQ4, dK4, dV4 = mha_bwd(
      q[:, None], k[:, None], v[:, None], dAttnPre[:, None], causal=causal, attn_mask=attn_mask
    )
    dQ, dK, dV = dQ4[:, 0], dK4[:, 0], dV4[:, 0]
  else:
    dQ, dK, dV = mha_bwd(q, k, v, dAttnPre, causal=causal, attn_mask=attn_mask)

  X2D = x.reshape(-1, d_model)
  dXq, dW_Q = matmul_bwd(X2D, W_Q, dQ.reshape(-1, d_qkv))
  dXk, dW_K = matmul_bwd(X2D, W_K, dK.reshape(-1, d_qkv))
  dXv, dW_V = matmul_bwd(X2D, W_V, dV.reshape(-1, d_qkv))

  dX = dRes1 + (dXq + dXk + dXv).reshape(B, S, d_model)
  return dX, dW_Q, dW_K, dW_V, dW_O, dW1, dW2, (dG1 + dG2), (dB1 + dB2)

# ---------------------------
# Torch version of components
# ---------------------------

def torch_matmul_bwd(A_np, B_np, dOut_np):
  A = torch.from_numpy(A_np).float().requires_grad_(True)
  B = torch.from_numpy(B_np).float().requires_grad_(True)
  dOut = torch.from_numpy(dOut_np).float()

  output = A @ B
  output.backward(dOut)

  return A.grad.detach().numpy(), B.grad.detach().numpy()


def torch_input_embedding_bwd(x_indices, W_E_np, dOut_np):
  W_E = torch.from_numpy(W_E_np).float().requires_grad_(True)
  x_indices = torch.from_numpy(x_indices).long()
  dOut = torch.from_numpy(dOut_np).float()

  output = F.embedding(x_indices, W_E)
  output.backward(dOut)

  return W_E.grad.detach().numpy()


def torch_softmax_bwd(x_np, grad_out_np):
  x = torch.from_numpy(x_np).float().requires_grad_(True)
  grad_out = torch.from_numpy(grad_out_np).float()

  softmax_output = F.softmax(x, dim=-1)
  softmax_output.backward(grad_out)

  return x.grad.detach().numpy()


def torch_mha_bwd(q_np, k_np, v_np, dOut_np, *, causal=False):
  q = torch.from_numpy(q_np).float().requires_grad_(True)
  k = torch.from_numpy(k_np).float().requires_grad_(True)
  v = torch.from_numpy(v_np).float().requires_grad_(True)
  dOut = torch.from_numpy(dOut_np).float()

  # Multi-head attention forward pass
  d_h = q.shape[-1]
  attn_scores = q @ k.transpose(-2, -1) / np.sqrt(d_h)

  if causal:
    S = attn_scores.shape[-1]
    mask = torch.tril(torch.ones((S, S), dtype=torch.bool))
    if attn_scores.dim() == 4:
      mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
    attn_scores = torch.where(mask, attn_scores, torch.tensor(1e-9, dtype=attn_scores.dtype))
  attn_weights = F.softmax(attn_scores, dim=-1)
  output = attn_weights @ v

  output.backward(dOut)

  return (q.grad.detach().numpy(), k.grad.detach().numpy(), v.grad.detach().numpy())


def torch_qkv_proj_bwd(x_np, W_Q_np, W_K_np, W_V_np, d_out_flat):
  batch_size, seq_len, d_model = x_np.shape
  _, d_qkv = W_Q_np.shape

  x = torch.from_numpy(x_np).float().requires_grad_(True)
  W_Q = torch.from_numpy(W_Q_np).float().requires_grad_(True)
  W_K = torch.from_numpy(W_K_np).float().requires_grad_(True)
  W_V = torch.from_numpy(W_V_np).float().requires_grad_(True)
  d_out = torch.from_numpy(d_out_flat.reshape(batch_size, seq_len, d_qkv)).float()

  # QKV projection forward pass
  q = x @ W_Q
  k = x @ W_K
  v = x @ W_V

  # Sum outputs since they share the same input x
  dX = q + k + v
  dX.backward(d_out)

  return (
    x.grad.reshape(-1, d_model).detach().numpy(),
    W_Q.grad.detach().numpy(),
    W_K.grad.detach().numpy(),
    W_V.grad.detach().numpy(),
  )


def torch_layer_norm_bwd(x_np, gamma_np, beta_np, dOut_np, eps=1e-6):
  x_t = torch.from_numpy(x_np).float().requires_grad_(True)
  gamma_t = torch.from_numpy(gamma_np).float().requires_grad_(True)
  beta_t = torch.from_numpy(beta_np).float().requires_grad_(True)
  dOut_t = torch.from_numpy(dOut_np).float()

  D = x_np.shape[-1]
  y = F.layer_norm(x_t, normalized_shape=(D,), weight=gamma_t, bias=beta_t, eps=eps)
  y.backward(dOut_t)

  return (x_t.grad.detach().numpy(), gamma_t.grad.detach().numpy(), beta_t.grad.detach().numpy())


def torch_ffn_bwd(x_np, W1_np, W2_np, dOut_np):
  x_t = torch.from_numpy(x_np).float().requires_grad_(True)
  W1_t = torch.from_numpy(W1_np).float().requires_grad_(True)
  W2_t = torch.from_numpy(W2_np).float().requires_grad_(True)
  dOut_t = torch.from_numpy(dOut_np).float()

  a1 = torch.relu(x_t @ W1_t)
  y = a1 @ W2_t
  y.backward(dOut_t)

  return (x_t.grad.detach().numpy(), W1_t.grad.detach().numpy(), W2_t.grad.detach().numpy())


def torch_block_bwd(
  x_np,
  W_Q_np,
  W_K_np,
  W_V_np,
  W_O_np,
  W_FF_expand_np,
  W_FF_contract_np,
  gamma_np,
  beta_np,
  dOut_np,
  eps=1e-6,
  causal=False,
):
  x_t = torch.from_numpy(x_np).float().requires_grad_(True)
  W_Q_t = torch.from_numpy(W_Q_np).float().requires_grad_(True)
  W_K_t = torch.from_numpy(W_K_np).float().requires_grad_(True)
  W_V_t = torch.from_numpy(W_V_np).float().requires_grad_(True)
  W_O_t = torch.from_numpy(W_O_np).float().requires_grad_(True)
  W_FF_expand_t = torch.from_numpy(W_FF_expand_np).float().requires_grad_(True)
  W_FF_contract_t = torch.from_numpy(W_FF_contract_np).float().requires_grad_(True)
  gamma_t = torch.from_numpy(gamma_np).float().requires_grad_(True)
  beta_t = torch.from_numpy(beta_np).float().requires_grad_(True)
  dOut_t = torch.from_numpy(dOut_np).float()

  # QKV projections
  q_t = x_t @ W_Q_t
  k_t = x_t @ W_K_t
  v_t = x_t @ W_V_t

  # Multi-head attention (batched matmul on 3D tensors)
  d_h = q_t.shape[-1]
  scores = (q_t @ k_t.transpose(-2, -1)) / np.sqrt(d_h)
  if causal:
    S = scores.shape[-1]
    mask = torch.tril(torch.ones((S, S), dtype=torch.bool))
    scores = torch.where(mask, scores, torch.tensor(float('-inf'), dtype=scores.dtype))
  weights = torch.softmax(scores, dim=-1)
  attn_out = weights @ v_t

  # Output projection
  attn_proj = attn_out @ W_O_t

  # Residual + LN
  res1 = x_t + attn_proj
  ln1 = F.layer_norm(res1, normalized_shape=(x_np.shape[-1],), weight=gamma_t, bias=beta_t, eps=eps)

  # FFN with ReLU
  ff = torch.relu(ln1 @ W_FF_expand_t) @ W_FF_contract_t

  # Residual + LN
  res2 = ln1 + ff
  ln2 = F.layer_norm(res2, normalized_shape=(x_np.shape[-1],), weight=gamma_t, bias=beta_t, eps=eps)

  # Backprop
  ln2.backward(dOut_t)

  return (
    x_t.grad.detach().numpy(),
    W_Q_t.grad.detach().numpy(),
    W_K_t.grad.detach().numpy(),
    W_V_t.grad.detach().numpy(),
    W_O_t.grad.detach().numpy(),
    W_FF_expand_t.grad.detach().numpy(),
    W_FF_contract_t.grad.detach().numpy(),
    gamma_t.grad.detach().numpy(),
    beta_t.grad.detach().numpy(),
  )
