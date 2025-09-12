import numpy as np
import torch
import torch.nn.functional as F


def torch_matmul_backward(A_np, B_np, dOut_np):
    """PyTorch implementation of matrix multiplication backward pass"""
    A_torch = torch.from_numpy(A_np).float().requires_grad_(True)
    B_torch = torch.from_numpy(B_np).float().requires_grad_(True)
    dOut_torch = torch.from_numpy(dOut_np).float()

    output_torch = A_torch @ B_torch
    output_torch.backward(dOut_torch)

    return A_torch.grad.detach().numpy(), B_torch.grad.detach().numpy()


def torch_input_embedding_backward(x_indices, W_E_np, dOut_np):
    """PyTorch implementation of input embedding backward pass"""
    W_E_torch = torch.from_numpy(W_E_np).float().requires_grad_(True)
    x_indices_torch = torch.from_numpy(x_indices).long()
    dOut_torch = torch.from_numpy(dOut_np).float()

    output_torch = F.embedding(x_indices_torch, W_E_torch)
    output_torch.backward(dOut_torch)

    return W_E_torch.grad.detach().numpy()


def torch_softmax_backward(x_np, grad_out_np):
    """PyTorch implementation of softmax backward pass"""
    x_torch = torch.from_numpy(x_np).float().requires_grad_(True)
    grad_out_torch = torch.from_numpy(grad_out_np).float()

    softmax_output = F.softmax(x_torch, dim=-1)
    softmax_output.backward(grad_out_torch)

    return x_torch.grad.detach().numpy()


def torch_multi_head_attention_backward(q_np, k_np, v_np, dOut_np):
    """PyTorch implementation of multi-head attention backward pass"""
    q_torch = torch.from_numpy(q_np).float().requires_grad_(True)
    k_torch = torch.from_numpy(k_np).float().requires_grad_(True)
    v_torch = torch.from_numpy(v_np).float().requires_grad_(True)
    dOut_torch = torch.from_numpy(dOut_np).float()

    # Multi-head attention forward pass
    d_h = q_torch.shape[-1]
    attn_scores = q_torch @ k_torch.transpose(-2, -1) / np.sqrt(d_h)
    attn_weights = F.softmax(attn_scores, dim=-1)
    output_torch = attn_weights @ v_torch

    output_torch.backward(dOut_torch)

    return (
        q_torch.grad.detach().numpy(),
        k_torch.grad.detach().numpy(),
        v_torch.grad.detach().numpy(),
    )


def torch_qkv_projection_backward(x_np, W_Q_np, W_K_np, W_V_np, dOut_flat):
    """PyTorch implementation of QKV projection backward pass"""
    batch_size, seq_len, d_model = x_np.shape
    _, d_qkv = W_Q_np.shape

    x_torch = torch.from_numpy(x_np).float().requires_grad_(True)
    W_Q_torch = torch.from_numpy(W_Q_np).float().requires_grad_(True)
    W_K_torch = torch.from_numpy(W_K_np).float().requires_grad_(True)
    W_V_torch = torch.from_numpy(W_V_np).float().requires_grad_(True)
    dOut_torch = torch.from_numpy(dOut_flat.reshape(batch_size, seq_len, d_qkv)).float()

    # QKV projection forward pass
    q_torch = x_torch @ W_Q_torch
    k_torch = x_torch @ W_K_torch
    v_torch = x_torch @ W_V_torch

    # Sum outputs since they share the same input x
    total_output = q_torch + k_torch + v_torch
    total_output.backward(dOut_torch)

    return (
        x_torch.grad.reshape(-1, d_model).detach().numpy(),
        W_Q_torch.grad.detach().numpy(),
        W_K_torch.grad.detach().numpy(),
        W_V_torch.grad.detach().numpy(),
    )


def torch_layer_norm_backward(x_np, gamma_np, beta_np, dOut_np, eps=1e-6):
    """PyTorch implementation of layer norm backward pass (last-dim)."""
    x_t = torch.from_numpy(x_np).float().requires_grad_(True)
    gamma_t = torch.from_numpy(gamma_np).float().requires_grad_(True)
    beta_t = torch.from_numpy(beta_np).float().requires_grad_(True)
    dOut_t = torch.from_numpy(dOut_np).float()

    D = x_np.shape[-1]
    y = F.layer_norm(x_t, normalized_shape=(D,), weight=gamma_t, bias=beta_t, eps=eps)
    y.backward(dOut_t)

    return (
        x_t.grad.detach().numpy(),
        gamma_t.grad.detach().numpy(),
        beta_t.grad.detach().numpy(),
    )


def torch_feed_forward_backward(x_np, W1_np, W2_np, dOut_np):
    """PyTorch implementation of FFN (ReLU) backward pass."""
    x_t = torch.from_numpy(x_np).float().requires_grad_(True)
    W1_t = torch.from_numpy(W1_np).float().requires_grad_(True)
    W2_t = torch.from_numpy(W2_np).float().requires_grad_(True)
    dOut_t = torch.from_numpy(dOut_np).float()

    a1 = torch.relu(x_t @ W1_t)
    y = a1 @ W2_t
    y.backward(dOut_t)

    return (
        x_t.grad.detach().numpy(),
        W1_t.grad.detach().numpy(),
        W2_t.grad.detach().numpy(),
    )


def torch_block_backward(
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
):
    """PyTorch implementation of the full block backward pass, matching block_forward."""
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
    weights = torch.softmax(scores, dim=-1)
    attn_out = weights @ v_t

    # Output projection
    attn_proj = attn_out @ W_O_t

    # Residual + LN
    res1 = x_t + attn_proj
    ln1 = F.layer_norm(
        res1, normalized_shape=(x_np.shape[-1],), weight=gamma_t, bias=beta_t, eps=eps
    )

    # FFN with ReLU
    ff = torch.relu(ln1 @ W_FF_expand_t) @ W_FF_contract_t

    # Residual + LN
    res2 = ln1 + ff
    ln2 = F.layer_norm(
        res2, normalized_shape=(x_np.shape[-1],), weight=gamma_t, bias=beta_t, eps=eps
    )

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


