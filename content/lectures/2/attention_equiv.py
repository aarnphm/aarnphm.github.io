# Attention ≈ Entropic kernel regression demo
# -------------------------------------------
#
# This script shows the mathematical equivalence between
# dot-product softmax attention and an RBF kernel smoother
# when Q and K are L2-normalized (same norm per row).
# It computes:
#   - standard attention weights and outputs
#   - RBF kernel weights (with sigma^2 = T) and outputs
# and reports the differences.
#
# You can tweak n, d, dv, and temperature T to experiment.

import numpy as np
import pandas as pd
from typing import Tuple

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    x_sum = np.sum(x, axis=axis, keepdims=True)
    return x / np.clip(x_sum, 1e-12, None)

def normalize_rows(X: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.clip(nrm, 1e-12, None)
    return X / nrm * target_norm

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, T: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    d = Q.shape[1]
    logits = (Q @ K.T) / np.sqrt(d) / T
    P = softmax(logits, axis=1)
    O = P @ V
    return P, O

def rbf_weights(Q: np.ndarray, K: np.ndarray, sigma2: float) -> np.ndarray:
    # Compute pairwise squared distances ||q_i - k_j||^2
    # Using (a - b)^2 = ||a||^2 + ||b||^2 - 2 a·b
    q2 = np.sum(Q**2, axis=1, keepdims=True)         # [n,1]
    k2 = np.sum(K**2, axis=1, keepdims=True).T       # [1,n]
    cross = Q @ K.T                                  # [n,n]
    d2 = q2 + k2 - 2.0 * cross
    W = np.exp(-d2 / (2.0 * sigma2))
    W = W / np.clip(np.sum(W, axis=1, keepdims=True), 1e-12, None)
    return W

def run_experiment(n=32, d=64, dv=16, T=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, d)).astype(np.float64)
    K = rng.standard_normal((n, d)).astype(np.float64)
    V = rng.standard_normal((n, dv)).astype(np.float64)

    Qn = normalize_rows(Q, target_norm=1.0)
    Kn = normalize_rows(K, target_norm=1.0)

    P_attn, O_attn = attention(Qn, Kn, V, T=T)

    sigma2 = T * np.sqrt(d)
    P_rbf = rbf_weights(Qn, Kn, sigma2=sigma2)
    O_rbf = P_rbf @ V

    w_max_abs = float(np.max(np.abs(P_attn - P_rbf)))
    w_fro = float(np.linalg.norm(P_attn - P_rbf, ord="fro"))
    o_max_abs = float(np.max(np.abs(O_attn - O_rbf)))
    o_fro = float(np.linalg.norm(O_attn - O_rbf, ord="fro"))

    summary = pd.DataFrame(
        [{
            "n": n, "d": d, "dv": dv, "T (sigma^2)": T,
            "sigma^2 used": sigma2,
            "max|ΔW|": w_max_abs, "||ΔW||_F": w_fro,
            "max|ΔO|": o_max_abs, "||ΔO||_F": o_fro
        }]
    )
    return summary, (P_attn, P_rbf), (O_attn, O_rbf)

# Run with defaults
summary, (P_attn, P_rbf), (O_attn, O_rbf) = run_experiment()

print(summary)
