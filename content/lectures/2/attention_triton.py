# attention_triton.py
import triton
import triton.language as tl


@triton.jit
def attn_two_pass(
  Q,
  K,
  V,
  O,
  n,
  d,
  dv,
  stride_qm,
  stride_qd,
  stride_km,
  stride_kd,
  stride_vm,
  stride_vd,
  stride_om,
  stride_od,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_D: tl.constexpr,
):
  # Row block we compute
  row_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
  d_offs = tl.arange(0, BLOCK_D)
  n_offs = tl.arange(0, BLOCK_N)

  # ----------------------------
  # Pass 1: compute per-row max & denom for stable softmax
  # ----------------------------
  m_i = tl.full((BLOCK_M,), -1e9, dtype=tl.float32)
  l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

  for col_start in range(0, n, BLOCK_N):
    # Load Q-block [M, D]
    q = tl.load(
      Q + row_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd,
      mask=(row_offs[:, None] < n) & (d_offs[None, :] < d),
      other=0.0,
    )
    # Load K-block [N, D]
    k = tl.load(
      K + (col_start + n_offs)[:, None] * stride_km + d_offs[None, :] * stride_kd,
      mask=((col_start + n_offs)[:, None] < n) & (d_offs[None, :] < d),
      other=0.0,
    )
    # scores [M, N] = q @ k^T / sqrt(d)
    scores = tl.dot(q, tl.trans(k)) * (1.0 / tl.sqrt(tl.float32(d)))

    # Online softmax: track max and rescale accumulated sum
    m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
    # Rescale previous accumulator when max increases
    alpha = tl.exp(m_i - m_i_new)
    l_i = l_i * alpha
    # Add current tile's contribution
    p = tl.exp(scores - m_i_new[:, None])
    l_i += tl.sum(p, axis=1)
    # Update running max
    m_i = m_i_new

  # ----------------------------
  # Pass 2: accumulate O = softmax(S) @ V
  # ----------------------------
  inv_l_i = 1.0 / l_i
  Oacc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

  for col_start in range(0, n, BLOCK_N):
    q = tl.load(
      Q + row_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd,
      mask=(row_offs[:, None] < n) & (d_offs[None, :] < d),
      other=0.0,
    )
    k = tl.load(
      K + (col_start + n_offs)[:, None] * stride_km + d_offs[None, :] * stride_kd,
      mask=((col_start + n_offs)[:, None] < n) & (d_offs[None, :] < d),
      other=0.0,
    )
    scores = tl.dot(q, tl.trans(k)) * (1.0 / tl.sqrt(tl.float32(d)))

    # Recenter using m_i computed in pass 1
    scores = scores - m_i[:, None]
    p = tl.exp(scores) * inv_l_i[:, None]  # [M, N]

    # Load V-block [N, Dv]
    v = tl.load(
      V + (col_start + n_offs)[:, None] * stride_vm + d_offs[None, :] * stride_vd,
      mask=((col_start + n_offs)[:, None] < n) & (d_offs[None, :] < dv),
      other=0.0,
    )

    Oacc += tl.dot(p, v)  # [M, Dv]

  # Store result
  tl.store(
    O + row_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
    Oacc,
    mask=(row_offs[:, None] < n) & (d_offs[None, :] < dv),
  )
