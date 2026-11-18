"""GardenGemma4ForCausalLM: vLLM's Gemma 4 with the garden body in every decoder layer.

Loaded by vLLM from the registry entry `register()` adds, when an engine is built with
`hf_overrides={'architectures': ['GardenGemma4ForCausalLM'], 'garden': {...GardenConfig...}}`
over the dequantised checkpoint (modal_interp.py `convert`). The weights load through the stock
Gemma4ForCausalLM path; only the layer forward changes.

Written against vLLM 0.33.0 (vllm/model_executor/models/gemma4.py). Not yet run on a GPU: the
spike in docs/latent-interp-vllm.md decides whether it runs under torch.compile + CUDA graphs
or needs enforce_eager.
"""

from __future__ import annotations

import types

from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

from .body import garden_layer
from .state import Garden, GardenConfig, Planner


class GardenGemma4ForCausalLM(Gemma4ForCausalLM):
  def __init__(self, *, vllm_config: VllmConfig, prefix: str = ''):
    super().__init__(vllm_config=vllm_config, prefix=prefix)
    hf = vllm_config.model_config.hf_config
    cfg = GardenConfig.from_dict(
      getattr(hf, 'garden', None) or getattr(self.config, 'garden', None)
    )
    if vllm_config.cache_config.kv_sharing_fast_prefill and (
      cfg.taps or cfg.readout
    ):
      # YOCO prefill runs layers 24..41 on the logits rows only: prompt positions there never exist
      raise ValueError(
        'garden: readout and taps need kv_sharing_fast_prefill off'
      )
    n_max = vllm_config.scheduler_config.max_num_batched_tokens
    conf = self.config
    dev = next(self.parameters()).device
    g = Garden(cfg, conf.num_hidden_layers, conf.hidden_size, n_max, dev)
    if g.dict and g.dict.engine not in (
      'torch/bf16-dequant',
      'vllm-0.33/bf16-dequant',
    ):
      print(
        f'[garden] {cfg.dict_path} was trained on {g.dict.engine} activations; serving it on the '
        'bf16 dequant (contract §7: check its fvu on this engine first)',
        flush=True,
      )
    self.model.garden = g
    self.garden_planner = Planner(g)

    def forward(
      layer, positions, hidden_states, residual, per_layer_input=None, **kw
    ):
      def attn(x):
        return layer.self_attn(positions=positions, hidden_states=x, **kw)

      return garden_layer(
        layer, layer.garden_l, hidden_states, per_layer_input, attn, g
      ), None

    for l, layer in enumerate(self.model.layers):
      if not hasattr(layer, 'self_attn'):  # PPMissingLayer
        continue
      layer.garden_l = l
      layer.forward = types.MethodType(forward, layer)
