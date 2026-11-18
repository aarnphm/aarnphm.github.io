"""The `vllm.general_plugins` entry point: vLLM calls it in every process before it builds a
model, so the worker that runs the forward has the model class, the ops and the runner wraps."""

from __future__ import annotations

ARCH = 'GardenGemma4ForCausalLM'


def register() -> None:
  from vllm import ModelRegistry

  from . import ops, runner

  assert (
    ops.readout is not None
  )  # importing ops registers torch.ops.garden.* before any trace
  if ARCH not in ModelRegistry.get_supported_archs():
    ModelRegistry.register_model(
      ARCH, 'garden_vllm.model:GardenGemma4ForCausalLM'
    )
  runner.install()
