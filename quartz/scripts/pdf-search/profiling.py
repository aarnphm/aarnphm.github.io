"""Optional synchronized phase measurements for the pinned native GPU scheduler."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from threading import Lock
import time
from typing import ParamSpec, TypeVar

import mlx.core as mx

P = ParamSpec('P')
R = TypeVar('R')


class PhaseProfile:
  def __init__(self, enabled: bool) -> None:
    self.enabled = enabled
    self.lock = Lock()
    self.calls: dict[str, int] = {}
    self.seconds: dict[str, float] = {}

  def measure(self, name: str, function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def measured(*args: P.args, **kwargs: P.kwargs) -> R:
      mx.synchronize()
      started = time.perf_counter()
      result = function(*args, **kwargs)
      if name == 'vision':
        # The encoder constructs lazy embeddings. Evaluate them here so its
        # GPU cost is attributed to vision instead of the next prefill call.
        mx.eval(result)
      mx.synchronize()
      elapsed = time.perf_counter() - started
      with self.lock:
        self.calls[name] = self.calls.get(name, 0) + 1
        self.seconds[name] = self.seconds.get(name, 0) + elapsed
      return result

    return measured

  def snapshot(self) -> dict[str, object]:
    with self.lock:
      return {
        'synchronized': self.enabled,
        'calls': dict(self.calls),
        'seconds': dict(self.seconds),
        'memory': {
          'activeBytes': mx.get_active_memory(),
          'cacheBytes': mx.get_cache_memory(),
          'peakBytes': mx.get_peak_memory(),
        },
      }


def install(enabled: bool) -> PhaseProfile:
  profile = PhaseProfile(enabled)
  if enabled:
    from mlx_vlm.generate.ar import GenerationBatch, PromptProcessingBatch
    from mlx_vlm.server.generation import ResponseGenerator

    ResponseGenerator._gpu_embed = profile.measure(
      'vision', ResponseGenerator._gpu_embed
    )
    ResponseGenerator._step = profile.measure(
      'scheduler', ResponseGenerator._step
    )
    GenerationBatch.next = profile.measure('decode', GenerationBatch.next)
    PromptProcessingBatch.prompt_step = profile.measure(
      'prefill', PromptProcessingBatch.prompt_step
    )
    PromptProcessingBatch.generate = profile.measure(
      'prefill-final', PromptProcessingBatch.generate
    )
  return profile
