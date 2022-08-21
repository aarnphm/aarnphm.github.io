from __future__ import annotations

import numpy as np, pytest, torch

from minigpt.np.modular import (
  matmul_bwd,
  input_embedding_bwd,
  qkv_proj_bwd,
  softmax_bwd,
  mha_bwd,
  qkv_proj,
  block_bwd,
  layer_norm_bwd,
  ffn_bwd,
  torch_matmul_bwd,
  torch_input_embedding_bwd,
  torch_softmax_bwd,
  torch_mha_bwd,
  torch_qkv_proj_bwd,
  torch_layer_norm_bwd,
  torch_ffn_bwd,
  torch_block_bwd,
)


@pytest.fixture(scope='module', autouse=True)
def auto_set_seed():
  np.random.seed(42)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True


def compare_gradients(grad_numpy, grad_torch, rtol=1e-5, atol=1e-6, name=''):
  """Compare numpy and torch gradients"""
  print(f'\n{name} Gradient Comparison:')
  print(f'  Numpy shape: {grad_numpy.shape}, Torch shape: {grad_torch.shape}')
  print(
    f'  Max absolute difference: {np.max(np.abs(grad_numpy - grad_torch)):.2e}'
  )
  print(
    f'  Relative error: {np.max(np.abs((grad_numpy - grad_torch) / (np.abs(grad_torch) + 1e-8))):.2e}'
  )

  if np.allclose(grad_numpy, grad_torch, rtol=rtol, atol=atol):
    print(f'  ✅ {name} gradients match!')
    return True
  else:
    print(f"  ❌ {name} gradients don't match!")
    return False


def test_matmul_bwd():
  """Test matmul backward pass against PyTorch"""
  # Create test data
  A_np = np.random.randn(4, 6).astype(np.float32)
  B_np = np.random.randn(6, 8).astype(np.float32)
  dOut_np = np.random.randn(4, 8).astype(np.float32)

  # NumPy implementation
  dA_numpy, dB_numpy = matmul_bwd(A_np, B_np, dOut_np)

  # PyTorch implementation
  dA_torch, dB_torch = torch_matmul_bwd(A_np, B_np, dOut_np)

  # Compare gradients
  match_A = compare_gradients(dA_numpy, dA_torch, name='dA')
  match_B = compare_gradients(dB_numpy, dB_torch, name='dB')

  assert match_A and match_B


def test_input_embedding_bwd():
  """Test input embedding backward pass"""
  # Create test data
  batch_size, seq_len, vocab_size, d_model = 2, 10, 1000, 512

  # Input indices (integer tokens)
  x_indices = np.random.randint(0, vocab_size, (batch_size, seq_len))
  x_onehot = np.eye(vocab_size)[
    x_indices
  ]  # Shape: (batch_size, seq_len, vocab_size)

  W_E_np = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.1
  dOut_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

  # Reshape for matmul: (batch_size * seq_len, vocab_size) @ (vocab_size, d_model)
  x_flat = x_onehot.reshape(-1, vocab_size)
  dOut_flat = dOut_np.reshape(-1, d_model)

  # NumPy implementation
  dX_flat, dW_E_numpy = input_embedding_bwd(x_flat, W_E_np, dOut_flat)

  # PyTorch implementation
  dW_E_torch = torch_input_embedding_bwd(x_indices, W_E_np, dOut_np)

  # Compare gradients (only W_E since x is discrete)
  match_W_E = compare_gradients(dW_E_numpy, dW_E_torch, name='dW_E')

  assert match_W_E


def test_softmax_bwd():
  """Test softmax backward pass"""
  # Create test data
  x_np = np.random.randn(3, 5).astype(np.float32)
  grad_out_np = np.random.randn(3, 5).astype(np.float32)

  # NumPy implementation
  grad_x_numpy = np.zeros_like(x_np)
  for i in range(x_np.shape[0]):
    grad_x_numpy[i] = softmax_bwd(x_np[i], grad_out_np[i])

  # PyTorch implementation
  grad_x_torch = torch_softmax_bwd(x_np, grad_out_np)

  # Compare gradients
  assert compare_gradients(grad_x_numpy, grad_x_torch, name='softmax')


@pytest.mark.parametrize('causal', [True, False])
def test_mha_bwd(causal):
  """Test multi-head attention backward pass"""
  # Create test data with smaller dimensions for debugging
  batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 8

  q_np = (
    np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    * 0.1
  )
  k_np = (
    np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    * 0.1
  )
  v_np = (
    np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    * 0.1
  )
  dOut_np = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(
    np.float32
  )

  # NumPy implementation
  dQ_numpy, dK_numpy, dV_numpy = mha_bwd(q_np, k_np, v_np, dOut_np)

  # PyTorch implementation
  dQ_torch, dK_torch, dV_torch = torch_mha_bwd(q_np, k_np, v_np, dOut_np)

  # Compare gradients
  match_q = compare_gradients(dQ_numpy, dQ_torch, name='dQ')
  match_k = compare_gradients(dK_numpy, dK_torch, name='dK')
  match_v = compare_gradients(dV_numpy, dV_torch, name='dV')

  assert match_q and match_k and match_v


def test_qkv_proj_bwd():
  """Test QKV projection backward pass"""

  # Create test data
  batch_size, seq_len, d_model, d_qkv = 2, 10, 512, 384

  x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.1
  W_Q_np = np.random.randn(d_model, d_qkv).astype(np.float32) * 0.1
  W_K_np = np.random.randn(d_model, d_qkv).astype(np.float32) * 0.1
  W_V_np = np.random.randn(d_model, d_qkv).astype(np.float32) * 0.1

  # Reshape for 2D matmul: (batch_size * seq_len, d_model) @ (d_model, d_qkv)
  x_flat = x_np.reshape(-1, d_model)

  # Forward pass
  q_flat, k_flat, v_flat = qkv_proj(x_flat, W_Q_np, W_K_np, W_V_np)

  # Create upstream gradients
  dOut_flat = np.random.randn(*q_flat.shape).astype(np.float32)

  # NumPy backward pass
  dX_numpy, dW_Q_numpy, dW_K_numpy, dW_V_numpy = qkv_proj_bwd(
    x_flat, W_Q_np, W_K_np, W_V_np, dOut_flat
  )

  # PyTorch implementation
  dX_torch, dW_Q_torch, dW_K_torch, dW_V_torch = torch_qkv_proj_bwd(
    x_np, W_Q_np, W_K_np, W_V_np, dOut_flat
  )

  # Compare gradients
  match_x = compare_gradients(dX_numpy, dX_torch, name='dX')
  match_wq = compare_gradients(dW_Q_numpy, dW_Q_torch, name='dW_Q')
  match_wk = compare_gradients(dW_K_numpy, dW_K_torch, name='dW_K')
  match_wv = compare_gradients(dW_V_numpy, dW_V_torch, name='dW_V')

  assert match_x and match_wq and match_wk and match_wv


def test_layer_norm_bwd():
  """Test layer norm backward pass"""
  B, S, D = 2, 5, 16
  x = np.random.randn(B, S, D).astype(np.float32)
  gamma = np.random.randn(D).astype(np.float32)
  beta = np.random.randn(D).astype(np.float32)
  dOut = np.random.randn(B, S, D).astype(np.float32)

  dX_np, dGamma_np, dBeta_np = layer_norm_bwd(x, gamma, beta, dOut)

  # PyTorch implementation
  dX_t, dGamma_t, dBeta_t = torch_layer_norm_bwd(x, gamma, beta, dOut)

  match_x = compare_gradients(dX_np, dX_t, name='LN dX')
  match_g = compare_gradients(dGamma_np, dGamma_t, name='LN dGamma')
  match_b = compare_gradients(dBeta_np, dBeta_t, name='LN dBeta')

  assert match_x and match_g and match_b


def test_ffn_bwd():
  """Test feed-forward network (ReLU) backward pass"""
  B, S, d_model, d_ff = 2, 7, 32, 64
  x = np.random.randn(B, S, d_model).astype(np.float32)
  W1 = (np.random.randn(d_model, d_ff) * 0.1).astype(np.float32)
  W2 = (np.random.randn(d_ff, d_model) * 0.1).astype(np.float32)
  dOut = np.random.randn(B, S, d_model).astype(np.float32)

  dX_np, dW1_np, dW2_np = ffn_bwd(x, W1, W2, dOut)

  # PyTorch implementation
  dX_t, dW1_t, dW2_t = torch_ffn_bwd(x, W1, W2, dOut)

  match_x = compare_gradients(dX_np, dX_t, name='FFN dX')
  match_w1 = compare_gradients(dW1_np, dW1_t, name='FFN dW1')
  match_w2 = compare_gradients(dW2_np, dW2_t, name='FFN dW2')

  assert match_x and match_w1 and match_w2


@pytest.mark.parametrize('causal', [True, False])
def test_block_bwd(causal):
  """Test full transformer block backward pass"""
  B, S, d_model = 2, 6, 32
  d_qkv = d_model  # keep same for simplicity
  d_ff = 64

  x = (np.random.randn(B, S, d_model) * 0.1).astype(np.float32)
  W_Q = (np.random.randn(d_model, d_qkv) * 0.1).astype(np.float32)
  W_K = (np.random.randn(d_model, d_qkv) * 0.1).astype(np.float32)
  W_V = (np.random.randn(d_model, d_qkv) * 0.1).astype(np.float32)
  W_O = (np.random.randn(d_model, d_model) * 0.1).astype(np.float32)
  W_FF_expand = (np.random.randn(d_model, d_ff) * 0.1).astype(np.float32)
  W_FF_contract = (np.random.randn(d_ff, d_model) * 0.1).astype(np.float32)
  gamma = (np.random.randn(d_model) * 0.1).astype(np.float32)
  beta = (np.random.randn(d_model) * 0.1).astype(np.float32)
  dOut = np.random.randn(B, S, d_model).astype(np.float32)

  grads_np = block_bwd(
    x,
    W_Q,
    W_K,
    W_V,
    W_O,
    W_FF_expand,
    W_FF_contract,
    gamma,
    beta,
    dOut,
    causal=causal,
  )

  # PyTorch implementation
  grads_t = torch_block_bwd(
    x,
    W_Q,
    W_K,
    W_V,
    W_O,
    W_FF_expand,
    W_FF_contract,
    gamma,
    beta,
    dOut,
    causal=causal,
  )

  names = [
    'dX',
    'dW_Q',
    'dW_K',
    'dW_V',
    'dW_O',
    'dW_FF_expand',
    'dW_FF_contract',
    'dGamma',
    'dBeta',
  ]
  matches = []
  for name, g_np, g_t in zip(names, grads_np, grads_t):
    matches.append(
      compare_gradients(g_np, g_t, name=f'{name} (causal={causal})')
    )

  assert all(matches)
