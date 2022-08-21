from __future__ import annotations

import dataclasses, numpy as np


@dataclasses.dataclass
class LMConfig:
  name: str = 'hinterland-np'
  tokenizer: str = 'Qwen/Qwen3-Next-80B-A3B-Instruct'
  d_model: int = 128
  n_heads: int = 4
  n_layers: int = 2
  d_ff: int = 4 * 128
  vocab_size: int = (
    151936  # NOTE: this will be updated after loading the tokenizer
  )
  max_seq_len: int = 256
  weight_tying: bool = True
  seed: int = 42
  lr: float = 3e-4
  betas: tuple[float, float] = dataclasses.field(
    default_factory=lambda: (0.9, 0.95)
  )
  weight_decay: float = 0.01
  grad_clip: float | None = 1.0
  batch_size: int = 16
  steps: int = 200
  warmup_steps: int = 0
  eval_every: int = 50
  log_every: int = 10
  stride: int = 0  # sliding window stride; 0 => seq_len
  prefetch: int = 4
  # Learning-rate scheduling
  lr_min: float = 1e-6
  plateau_patience: int = 5
  plateau_factor: float = 0.5
  lr_cooldown: int = 0
  # Early stopping
  early_stop_patience: int = 10
  early_stop_min_delta: float = 1e-3
  target_loss: float | None = None


@dataclasses.dataclass
class LMParams:
  W_E: np.ndarray
  W_pos: np.ndarray
  blocks: list[BlockParams]
  gamma_f: np.ndarray
  beta_f: np.ndarray
  W_LM: np.ndarray | None


# transformers block
@dataclasses.dataclass
class BlockParams:
  W_Q: np.ndarray
  W_K: np.ndarray
  W_V: np.ndarray
  W_O: np.ndarray
  gamma1: np.ndarray
  beta1: np.ndarray
  W1: np.ndarray
  W2: np.ndarray
  gamma2: np.ndarray
  beta2: np.ndarray
