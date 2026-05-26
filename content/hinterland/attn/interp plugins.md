---
date: '2026-05-26'
description: proposal for efficient inference support for interp research
id: interp plugins
modified: 2026-05-26 17:32:24 GMT-04:00
tags:
  - ml
  - alignment
title: interp plugins
---

vLLM v1 plugin entry point:

```python title="setup.py"
setup(
  name='vllm-sae-plugin',
  entry_points={
    'vllm.general_plugins': ['register_sae = vllm_sae_plugin:register']
  },
)
```

Components:

1. `SAERegistry` - Manages multiple SAE/CLT checkpoints per layer
2. `ActivationInterceptor` - Hooks into attention/MLP outputs
3. `FeatureCache` - Persistent batch caching for sparse representations
4. `DriftMonitor` - Tracks feature activation distributions
5. `SteeringController` - Applies feature-level interventions

## activation interception

Hook into vLLM's model executor at attention output layer (pre-residual addition):

```python title="activations.py"
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class InterceptionConfig:
  sae_registry: dict[tuple[int, str], Any]
  target_modules: tuple[str, ...]
  intervention_cache: dict[int, dict[str, Any]]


def create_interception_config(
  sae_registry: dict[tuple[int, str], Any], target_modules: list[str]
) -> InterceptionConfig:
  return InterceptionConfig(
    sae_registry=sae_registry,
    target_modules=tuple(target_modules),
    intervention_cache={},
  )


def get_sae_for_layer(
  registry: dict[tuple[int, str], Any], layer_idx: int, module_type: str
) -> Any | None:
  return registry.get((layer_idx, module_type))


def apply_steering(
  sparse_features: torch.Tensor,
  interventions: dict[int, dict[str, Any]],
  layer_idx: int,
) -> torch.Tensor:
  if layer_idx not in interventions:
    return sparse_features

  intervention = interventions[layer_idx]
  return sparse_features + intervention.get('bias', 0) * intervention.get(
    'scale', 1.0
  )


def process_activations(
  activations: torch.Tensor,
  sae: Any | None,
  interventions: dict[int, dict[str, Any]],
  layer_idx: int,
  drift_monitor: Callable[[torch.Tensor, int], None],
) -> torch.Tensor:
  if sae is None:
    return activations

  sparse_features = sae.encode(activations)
  sparse_features = apply_steering(sparse_features, interventions, layer_idx)
  drift_monitor(sparse_features, layer_idx)

  return sae.decode(sparse_features)


def create_attention_hook(
  config: InterceptionConfig,
  layer_idx: int,
  drift_monitor: Callable[[torch.Tensor, int], None],
) -> Callable:
  def hook(module, input, output):
    activations = output[0]
    sae = get_sae_for_layer(config.sae_registry, layer_idx, 'attn')
    reconstructed = process_activations(
      activations, sae, config.intervention_cache, layer_idx, drift_monitor
    )
    return (reconstructed,) + output[1:]

  return hook


def register_hooks(
  config: InterceptionConfig,
  model,
  drift_monitor: Callable[[torch.Tensor, int], None],
):
  target_layers = [
    idx
    for idx in range(model.config.num_hidden_layers)
    if f'layer.{idx}.attn' in config.target_modules
  ]

  return [
    model.layers[idx].self_attn.register_forward_hook(
      create_attention_hook(config, idx, drift_monitor)
    )
    for idx in target_layers
  ]
```

## cross-layer transcoders

CLTs read from one layer and write to multiple downstream layers, requiring special handling:

```python title="transcoders.py"
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch


class CLTConfig(NamedTuple):
  clt: Any
  source_layer: int
  target_layers: tuple[int, ...]


def create_clt_config(clt_checkpoint) -> CLTConfig:
  clt = load_clt(clt_checkpoint)
  return CLTConfig(
    clt=clt,
    source_layer=clt_checkpoint.source_layer,
    target_layers=tuple(clt_checkpoint.target_layers),
  )


def encode_source_activations(
  clt: Any, activations: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
  sparse_features = clt.encode(activations)
  feature_cache = {
    pos.item(): sparse_features[idx] for idx, pos in enumerate(position_ids)
  }
  return sparse_features, feature_cache


def apply_cached_to_target(
  clt: Any,
  feature_cache: dict[int, torch.Tensor],
  target_layers: tuple[int, ...],
  layer_idx: int,
  position_ids: torch.Tensor,
) -> torch.Tensor | None:
  if layer_idx not in target_layers:
    return None

  cached = torch.stack([feature_cache[pos.item()] for pos in position_ids])
  return clt.decode_to_layer(cached, layer_idx)


def create_clt_pipeline(config: CLTConfig):
  def encode(activations: torch.Tensor, position_ids: torch.Tensor):
    return encode_source_activations(config.clt, activations, position_ids)

  def decode(
    feature_cache: dict[int, torch.Tensor],
    layer_idx: int,
    position_ids: torch.Tensor,
  ):
    return apply_cached_to_target(
      config.clt, feature_cache, config.target_layers, layer_idx, position_ids
    )

  return encode, decode
```

### matryoshka SAE support

$k$ varies per token. picks $k \in \{32, 64, 128, 256\}$ from complexity score:

```python title="matryoshka.py"
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch


class MatryoshkaConfig(NamedTuple):
  encoder: Any
  decoders: dict[int, Any]
  sparsity_levels: tuple[int, ...]
  complexity_estimator: Callable[[torch.Tensor], torch.Tensor]


def create_matryoshka_config(
  checkpoint, sparsity_levels: list[int] | None = None
) -> MatryoshkaConfig:
  if sparsity_levels is None:
    sparsity_levels = [32, 64, 128, 256]

  return MatryoshkaConfig(
    encoder=checkpoint.encoder,
    decoders={k: checkpoint.decoders[k] for k in sparsity_levels},
    sparsity_levels=tuple(sparsity_levels),
    complexity_estimator=ComplexityEstimator(),
  )


def select_sparsity_level(
  complexity_score: float, thresholds: tuple[float, ...] = (0.3, 0.6, 0.85)
) -> int:
  levels = [32, 64, 128, 256]
  for i, threshold in enumerate(thresholds):
    if complexity_score < threshold:
      return levels[i]
  return levels[-1]


def create_sparse_features(
  all_features: torch.Tensor, k: int, idx: int
) -> torch.Tensor:
  topk_vals, topk_idx = torch.topk(all_features[idx], k)
  sparse = torch.zeros_like(all_features[idx])
  sparse.scatter_(0, topk_idx, topk_vals)
  return sparse


def decode_with_sparsity(
  sparse_features: torch.Tensor, decoders: dict[int, Any], k: int
) -> torch.Tensor:
  return decoders[k](sparse_features)


def adaptive_encode_decode(
  config: MatryoshkaConfig, activations: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
  complexity = config.complexity_estimator(token_ids)
  all_features = config.encoder(activations)

  def process_token(idx: int, complexity_score: float) -> torch.Tensor:
    k = select_sparsity_level(complexity_score)
    sparse = create_sparse_features(all_features, k, idx)
    return decode_with_sparsity(sparse, config.decoders, k)

  reconstructed = [
    process_token(i, score) for i, score in enumerate(complexity)
  ]

  return torch.stack(reconstructed)
```

## feature drift monitoring

Track distributional shifts in feature activations over time:

```python title="drift.py"
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch


class DriftState(NamedTuple):
  n_features: int
  window_size: int
  activation_counts: torch.Tensor
  activation_means: torch.Tensor
  activation_vars: torch.Tensor
  baseline_counts: torch.Tensor | None
  baseline_means: torch.Tensor | None
  total_samples: int


def create_drift_state(n_features: int, window_size: int = 1000) -> DriftState:
  return DriftState(
    n_features=n_features,
    window_size=window_size,
    activation_counts=torch.zeros(n_features),
    activation_means=torch.zeros(n_features),
    activation_vars=torch.zeros(n_features),
    baseline_counts=None,
    baseline_means=None,
    total_samples=0,
  )


def update_welford_statistics(
  state: DriftState, sparse_features: torch.Tensor
) -> DriftState:
  active_mask = sparse_features != 0
  new_counts = state.activation_counts + active_mask.sum(dim=0)

  new_means = state.activation_means.clone()
  new_vars = state.activation_vars.clone()
  total = state.total_samples

  for i in range(state.n_features):
    active_values = sparse_features[:, i][active_mask[:, i]]
    if len(active_values) > 0:
      for val in active_values:
        total += 1
        delta = val - new_means[i]
        new_means[i] += delta / total
        delta2 = val - new_means[i]
        new_vars[i] += delta * delta2

  return state._replace(
    activation_counts=new_counts,
    activation_means=new_means,
    activation_vars=new_vars,
    total_samples=total,
  )


def compute_kl_divergence(
  freq_current: torch.Tensor,
  freq_baseline: torch.Tensor,
  epsilon: float = 1e-10,
) -> float:
  return torch.sum(
    freq_current
    * torch.log(freq_current / (freq_baseline + epsilon) + epsilon)
  ).item()


def detect_frequency_drift(
  state: DriftState, threshold: float = 0.1
) -> tuple[str, float] | None:
  if state.baseline_counts is None:
    return None

  freq_current = state.activation_counts / state.total_samples
  freq_baseline = state.baseline_counts / state.baseline_counts.sum()

  kl_div = compute_kl_divergence(freq_current, freq_baseline)

  return ('frequency_drift', kl_div) if kl_div > threshold else None


def detect_mean_shift(
  state: DriftState, threshold: float = 2.0
) -> tuple[str, list[int]] | None:
  if state.baseline_means is None:
    return None

  mean_shifts = torch.abs(state.activation_means - state.baseline_means)
  significant_shifts = mean_shifts > threshold

  if significant_shifts.any():
    drifted_features = torch.where(significant_shifts)[0].tolist()
    return ('feature_shift', drifted_features)

  return None


def check_drift(
  state: DriftState, alert_fn: Callable[[str, Any], None]
) -> DriftState:
  if state.baseline_counts is None:
    return state._replace(
      baseline_counts=state.activation_counts.clone(),
      baseline_means=state.activation_means.clone(),
    )

  for drift_result in [
    detect_frequency_drift(state),
    detect_mean_shift(state),
  ]:
    if drift_result is not None:
      drift_type, details = drift_result
      alert_fn(drift_type, details)

  return state


def update_drift_monitor(
  state: DriftState,
  sparse_features: torch.Tensor,
  alert_fn: Callable[[str, Any], None],
) -> DriftState:
  new_state = update_welford_statistics(state, sparse_features)

  if new_state.total_samples % new_state.window_size == 0:
    new_state = check_drift(new_state, alert_fn)

  return new_state


def emit_drift_alert(drift_type: str, details: Any) -> None:
  logger.warning(
    f'Feature drift detected: {drift_type}', extra={'details': details}
  )
```

## batched sparse ops

target: <5% decode overhead. fused TopK + decode:

```python title="kernels.py"
from __future__ import annotations

import torch


@torch.compile(mode='max-autotune')
def fused_topk_decode(
  encoded: torch.Tensor, decoder_weights: torch.Tensor, k: int
) -> torch.Tensor:
  batch, seq_len, n_features = encoded.shape
  topk_vals, topk_idx = torch.topk(encoded, k, dim=-1)
  selected_weights = decoder_weights[topk_idx]
  reconstructed = (topk_vals.unsqueeze(-1) * selected_weights).sum(dim=-2)
  return reconstructed


def register_sae_custom_op():
  from vllm.model_executor.custom_op import CustomOP

  @CustomOP.register_oot(name='sae_topk_decode')
  class SAETopKDecode:
    def forward(self, encoded, decoder, k):
      return fused_topk_decode(encoded, decoder, k)

  return SAETopKDecode
```

## persistent caching

vLLM v1 persistent-batch for SAE state:

```python title="cache.py"
from __future__ import annotations

from typing import Any, NamedTuple

import torch


class CacheState(NamedTuple):
  sparse_indices: torch.Tensor
  sparse_values: torch.Tensor
  position_map: dict[str, int]


class CacheConfig(NamedTuple):
  max_batch_size: int
  max_seq_len: int
  n_features: int
  k: int = 128
  dtype: torch.dtype = torch.float16
  device: str = 'cuda'


def create_cache_state(config: CacheConfig) -> CacheState:
  return CacheState(
    sparse_indices=torch.zeros(
      (config.max_batch_size, config.max_seq_len, config.k),
      dtype=torch.int32,
      device=config.device,
    ),
    sparse_values=torch.zeros(
      (config.max_batch_size, config.max_seq_len, config.k),
      dtype=config.dtype,
      device=config.device,
    ),
    position_map={},
  )


def extract_topk_features(
  sparse_features: torch.Tensor, k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
  return torch.topk(sparse_features, k, dim=-1)


def update_cache_diff(
  state: CacheState,
  request_id: str,
  new_positions: list[int],
  sparse_features: torch.Tensor,
) -> CacheState:
  base_idx = state.position_map.get(request_id, 0)
  topk_vals, topk_idx = extract_topk_features(sparse_features)

  new_indices = state.sparse_indices.clone()
  new_values = state.sparse_values.clone()

  n_new = len(new_positions)
  new_indices[base_idx : base_idx + n_new] = topk_idx
  new_values[base_idx : base_idx + n_new] = topk_vals

  new_position_map = {**state.position_map, request_id: base_idx + n_new}

  return CacheState(
    sparse_indices=new_indices,
    sparse_values=new_values,
    position_map=new_position_map,
  )


def reconstruct_from_cache(
  state: CacheState, request_id: str, decoder_weights: torch.Tensor
) -> torch.Tensor:
  positions = state.position_map.get(request_id, 0)
  indices = state.sparse_indices[:positions]
  values = state.sparse_values[:positions]

  weights = decoder_weights[indices]
  return (values.unsqueeze(-1) * weights).sum(dim=-2)


def create_cache_manager(config: CacheConfig):
  state_ref = [create_cache_state(config)]

  def update(
    request_id: str, new_positions: list[int], sparse_features: torch.Tensor
  ):
    state_ref[0] = update_cache_diff(
      state_ref[0], request_id, new_positions, sparse_features
    )

  def reconstruct(
    request_id: str, decoder_weights: torch.Tensor
  ) -> torch.Tensor:
    return reconstruct_from_cache(state_ref[0], request_id, decoder_weights)

  return update, reconstruct
```

## top-level API

`{feature_id: scalar}` steering map. `mode` $\to$ `(scale, bias)` at hook site. per-layer checkpoints in one dict.

```python
from vllm import LLM
from vllm_sae_plugin import SAEConfig, SteeringVector

llm = LLM(
  model='meta-llama/Llama-3.2-3B',
  sae_config=SAEConfig(
    checkpoints={
      'attn_15': 'path/to/clt_layer15.pt',
      'attn_20': 'path/to/matryoshka_layer20.pt',
      'mlp_23': 'path/to/standard_sae_layer23.pt',
    },
    target_modules=['attn_output'],
    enable_drift_monitoring=True,
    drift_window=1000,
    cache_config={'max_cached_features': 100_000, 'eviction_policy': 'lru'},
    optimization={
      'use_custom_kernels': True,
      'compile_mode': 'max-autotune',
      'enable_persistent_cache': True,
    },
  ),
)

steering = SteeringVector(
  layer=15, features={1337: 2.5, 4242: -1.0}, mode='additive'
)

outputs = llm.generate(
  'The weather in California is', steering_vectors=[steering], temperature=0.7
)

drift_report = llm.sae_plugin.get_drift_report()
# {"layer_15": {"kl_divergence": 0.03, "shifted_features": [128, 1337, 2048], "dead_features": [42, 99]}}
```

## monitoring

Prometheus / OTel off drift state:

```python title="metrics.py"
from __future__ import annotations

from typing import Any, NamedTuple

import torch


class MetricsConfig(NamedTuple):
  port: int = 9090
  dead_feature_threshold: float = 0.001


def create_metrics_registry() -> dict[str, Any]:
  return {
    'sae_feature_activations': Counter(),
    'sae_drift_kl_divergence': Gauge(),
    'sae_reconstruction_error': Histogram(),
    'sae_latency_overhead': Histogram(),
  }


def compute_activation_rate(drift_state: DriftState) -> list[float]:
  if drift_state.total_samples == 0:
    return torch.zeros(drift_state.n_features).tolist()
  return (drift_state.activation_counts / drift_state.total_samples).tolist()


def detect_dead_features(
  drift_state: DriftState, threshold: float = 0.001
) -> list[int]:
  activation_rate = drift_state.activation_counts / drift_state.total_samples
  dead_mask = activation_rate < threshold
  return torch.where(dead_mask)[0].tolist()


def compute_current_kl_divergence(drift_state: DriftState) -> float | None:
  if drift_state.baseline_counts is None:
    return None

  freq_current = drift_state.activation_counts / drift_state.total_samples
  freq_baseline = (
    drift_state.baseline_counts / drift_state.baseline_counts.sum()
  )

  return compute_kl_divergence(freq_current, freq_baseline)


def export_metrics(
  drift_state: DriftState, config: MetricsConfig
) -> dict[str, Any]:
  return {
    'feature_activation_rate': compute_activation_rate(drift_state),
    'drift_kl': compute_current_kl_divergence(drift_state),
    'dead_features': detect_dead_features(
      drift_state, config.dead_feature_threshold
    ),
  }


def create_metrics_exporter(config: MetricsConfig):
  registry = create_metrics_registry()

  def export(drift_state: DriftState) -> dict[str, Any]:
    metrics = export_metrics(drift_state, config)
    registry['sae_drift_kl_divergence'].set(metrics['drift_kl'] or 0.0)
    return metrics

  return export
```

## deferred

- multi-SAE ensembles per layer
- adaptive $k$ predictor beyond Matryoshka discrete levels
- pre-computed steering vectors keyed by feature-set hash
- cross-request feature aggregation (separate process)
- online SAE finetuning (out of scope)
