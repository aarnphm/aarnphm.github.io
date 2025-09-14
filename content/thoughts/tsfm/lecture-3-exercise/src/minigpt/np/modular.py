from __future__ import annotations

import numpy as np, numpy.typing as npt


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
