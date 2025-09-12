import numpy as np
from typing import Tuple

def matmul_backward(A: np.ndarray, B: np.ndarray, dY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dB = A.T @ dY
    dA = dY @ B.T
    return dA, dB

def input_embedding(x: np.ndarray, W_E: np.ndarray) -> np.ndarray:
    return x @ W_E

def input_embedding_backward(x: np.ndarray, W_E: np.ndarray, dOut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return matmul_backward(x, W_E, dOut)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + 1e-6) + beta

def layer_norm_backward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, dOut: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-6
    D = x.shape[-1]
    mu = x.mean(axis=-1, keepdims=True)
    x_mu = x - mu
    var = (x_mu ** 2).mean(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = x_mu * inv_std

    dGamma = np.sum(dOut * x_hat, axis=tuple(range(x.ndim - 1)))
    dBeta  = np.sum(dOut, axis=tuple(range(x.ndim - 1)))

    dX_hat = dOut * gamma
    dVar = np.sum(dX_hat * x_mu * (-0.5) * inv_std**3, axis=-1, keepdims=True)
    dMu  = np.sum(-dX_hat * inv_std, axis=-1, keepdims=True) + dVar * np.sum(-2.0 * x_mu, axis=-1, keepdims=True) / D
    dX   = dX_hat * inv_std + dVar * 2.0 * x_mu / D + dMu / D
    return dX, dGamma, dBeta

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)

def softmax_backward_from_probs(p: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
    # dL/dz = p ⊙ (g - ⟨g, p⟩), where g = dL/dp
    dot = (grad_out * p).sum(axis=-1, keepdims=True)
    return p * (grad_out - dot)

def softmax_backward(scores: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
    return softmax_backward_from_probs(softmax(scores), grad_out)

def qkv_projection(x: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray):
    return x @ W_Q, x @ W_K, x @ W_V

def qkv_projection_backward(x: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, dOut: np.ndarray):
    dXq, dW_Q = matmul_backward(x, W_Q, dOut)
    dXk, dW_K = matmul_backward(x, W_K, dOut)
    dXv, dW_V = matmul_backward(x, W_V, dOut)
    return dXq + dXk + dXv, dW_Q, dW_K, dW_V

def multi_head_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = 1.0 / np.sqrt(q.shape[-1])
    attn = softmax(q @ np.swapaxes(k, -2, -1) * scale)
    return attn @ v

def multi_head_attention_backward(q: np.ndarray, k: np.ndarray, v: np.ndarray, dOut: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Dh = q.shape[-1]
    scale = 1.0 / np.sqrt(Dh)

    scores = q @ np.swapaxes(k, -2, -1) * scale
    attn = softmax(scores)

    dV = np.swapaxes(attn, -2, -1) @ dOut
    dAttn = dOut @ np.swapaxes(v, -2, -1)

    dScores = softmax_backward_from_probs(attn, dAttn)
    dQ = dScores @ k * scale
    dK = np.swapaxes(dScores, -2, -1) @ q * scale
    return dQ, dK, dV

def feed_forward_network(x: np.ndarray, W_1: np.ndarray, W_2: np.ndarray):
    return relu(x @ W_1) @ W_2

def feed_forward_network_backward(x: np.ndarray, W_1: np.ndarray, W_2: np.ndarray, dOut: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, S, d_model = x.shape
    X2D = x.reshape(-1, d_model)

    Z1 = X2D @ W_1
    A1 = relu(Z1)
    dA1, dW_2 = matmul_backward(A1, W_2, dOut.reshape(-1, d_model))
    dZ1 = dA1 * (Z1 > 0)
    dX2D, dW_1 = matmul_backward(X2D, W_1, dZ1)

    return dX2D.reshape(B, S, d_model), dW_1, dW_2

def block_forward(x: np.ndarray,
                  W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                  W_O: np.ndarray, W_FF_expand: np.ndarray, W_FF_contract: np.ndarray,
                  gamma: np.ndarray, beta: np.ndarray):
    q, k, v = qkv_projection(x, W_Q, W_K, W_V)
    attn_o = multi_head_attention(q, k, v) @ W_O
    ln1 = layer_norm(x + attn_o, gamma, beta)
    ff = feed_forward_network(ln1, W_FF_expand, W_FF_contract)
    return layer_norm(ln1 + ff, gamma, beta)

def block_backward(x: np.ndarray,
                   W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                   W_O: np.ndarray, W_FF_expand: np.ndarray, W_FF_contract: np.ndarray,
                   gamma: np.ndarray, beta: np.ndarray,
                   dOut: np.ndarray):
    q, k, v = qkv_projection(x, W_Q, W_K, W_V)
    attn_pre = multi_head_attention(q, k, v)
    attn_proj = attn_pre @ W_O

    res1 = x + attn_proj
    ln1  = layer_norm(res1, gamma, beta)

    ff   = feed_forward_network(ln1, W_FF_expand, W_FF_contract)
    res2 = ln1 + ff

    dRes2, dG2, dB2 = layer_norm_backward(res2, gamma, beta, dOut)

    dLn1_ff, dW1, dW2 = feed_forward_network_backward(ln1, W_FF_expand, W_FF_contract, dRes2)
    dRes1, dG1, dB1   = layer_norm_backward(res1, gamma, beta, dLn1_ff + dRes2)

    B, S, d_model = x.shape
    d_qkv = attn_pre.shape[-1]

    dAttnPre_2d, dW_O = matmul_backward(attn_pre.reshape(-1, d_qkv), W_O, dRes1.reshape(-1, d_model))
    dAttnPre = dAttnPre_2d.reshape(B, S, d_qkv)

    # Support both 3D and 4D attention tensors
    if q.ndim == 3:
        dQ4, dK4, dV4 = multi_head_attention_backward(q[:, None], k[:, None], v[:, None], dAttnPre[:, None])
        dQ, dK, dV = dQ4[:, 0], dK4[:, 0], dV4[:, 0]
    else:
        dQ, dK, dV = multi_head_attention_backward(q, k, v, dAttnPre)

    X2D = x.reshape(-1, d_model)
    dXq, dW_Q = matmul_backward(X2D, W_Q, dQ.reshape(-1, d_qkv))
    dXk, dW_K = matmul_backward(X2D, W_K, dK.reshape(-1, d_qkv))
    dXv, dW_V = matmul_backward(X2D, W_V, dV.reshape(-1, d_qkv))

    dX = dRes1 + (dXq + dXk + dXv).reshape(B, S, d_model)
    return dX, dW_Q, dW_K, dW_V, dW_O, dW1, dW2, (dG1 + dG2), (dB1 + dB2)
