"""garden_vllm: Gemma 4 E4B interpretability inside vLLM, as a general plugin.

Feature readout (transcoder codes per token), steering by transcoder feature, and activation
taps, at the sites of docs/latent-interp-contract.md, as graph-safe custom ops in an out-of-tree
Gemma 4 model class. Attribution graphs are not here: they need a backward pass, which vLLM
does not have (gemma_torch.py / attribute.py). Design: docs/latent-interp-vllm.md.

`pip install quartz/scripts/latent/garden_vllm` registers `plugin.register` under the
`vllm.general_plugins` entry point, which vLLM calls in every process before building a model.
The ops, the layer body and the host planner import without vLLM (tests, the torch engine).
"""

from __future__ import annotations

from .plugin import register
from .state import SITES, Garden, GardenConfig, Planner, Req, Sink, sink_for

__all__ = [
  'SITES',
  'Garden',
  'GardenConfig',
  'Planner',
  'Req',
  'Sink',
  'register',
  'sink_for',
]
