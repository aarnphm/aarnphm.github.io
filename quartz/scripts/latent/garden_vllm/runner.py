"""The per-step glue in vLLM's GPU model runner (vllm/v1/worker/gpu_model_runner.py, 0.33.0).

Two wraps, nothing else patched:

- `_prepare_inputs`: once vLLM has laid out the step's flat batch, read each scheduled request's
  `extra_args['garden']`, its first position and token count, and let the Planner write the slot
  and tap-row buffers (host-to-device copies on the forward's stream, so they land first).
- `_model_forward`: after the forward, the Planner copies the step's codes and taps to the host
  and hands them to the sink. Dummy runs and graph captures have no plan and are skipped.

Assumes the classic runner (VLLM_USE_V2_MODEL_RUNNER=0), no async scheduling (a step's buffers
are read before the next step writes them), no speculative decoding and TP 1.
"""

from __future__ import annotations

from .state import Req

_INSTALLED = False


def _garden_model(runner):
  m = runner.get_model() if hasattr(runner, 'get_model') else runner.model
  return m if hasattr(m, 'garden_planner') else None


def install() -> None:
  global _INSTALLED
  if _INSTALLED:
    return
  from vllm.v1.worker.gpu_model_runner import GPUModelRunner as R

  prepare, forward = R._prepare_inputs, R._model_forward

  def _prepare_inputs(
    self, scheduler_output, num_scheduled_tokens_np, *a, **kw
  ):
    out = prepare(self, scheduler_output, num_scheduled_tokens_np, *a, **kw)
    m = _garden_model(self)
    if m is None:
      return out
    batch = self.input_batch
    reqs = []
    for i, rid in enumerate(batch.req_ids[: batch.num_reqs]):
      st = self.requests[rid]
      sp = st.sampling_params
      spec = (
        (getattr(sp, 'extra_args', None) or {}).get('garden')
        if sp is not None
        else None
      )
      reqs.append(
        Req(
          id=rid,
          spec=spec,
          start=int(batch.num_computed_tokens_cpu[i]),
          n=int(num_scheduled_tokens_np[i]),
          prompt=len(st.prompt_token_ids or ()),
        )
      )
    self._garden_plan = m.garden_planner.plan(
      reqs, scheduler_output.finished_req_ids
    )
    return out

  def _model_forward(self, *a, **kw):
    out = forward(self, *a, **kw)
    plan = getattr(self, '_garden_plan', None)
    if plan is not None:
      self._garden_plan = None
      _garden_model(self).garden_planner.collect(plan)
    return out

  R._prepare_inputs, R._model_forward = _prepare_inputs, _model_forward
  _INSTALLED = True
