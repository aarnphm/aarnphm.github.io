"""Bounded drain/resume with successful stages retained across invocations."""

from __future__ import annotations

from time import monotonic

from files import fresh, fresh_job, save_state
from jobs import accept_review, submit
from plan import status
from providers import dispatch, prepare, response_json
from schema import Job, Provider, State


def review_job(
  state: State, job: Job, provider: Provider, timeout_ms: int | None = None
) -> Job:
  request = prepare(state, job, 'verify', provider, timeout_ms)
  state.calls += 1
  save_state(
    state,
    'dispatch',
    {
      'job': job.input.id,
      'invocation': request.invocation,
      'provider': provider,
      'stage': 'verify',
    },
  )
  result = dispatch(request, state)
  if not result.complete:
    raise ValueError(
      f'{provider}: {result.error}; invocation {request.invocation}'
    )
  accept_review(
    state, job, provider, request.invocation, response_json(result.response)
  )
  return job


def run(
  state: State,
  *,
  identity: str | None,
  max_jobs: int,
  until_done: bool,
  dry_run: bool,
) -> dict[str, object]:
  if dry_run:
    return {**status(state), 'dryRun': True}
  initial = state.calls
  deadline = monotonic() + state.config.limits.invocationMs / 1000
  visited: set[str] = set()
  reason = 'No runnable jobs remain'
  while (
    monotonic() < deadline - 15
    and state.calls - initial < state.config.limits.maxCalls
  ):
    ready = [
      job
      for job in state.jobs
      if job.status in ['queued', 'revise', 'awaiting_review']
      and (identity is None or job.input.id == identity)
    ]
    if not until_done:
      ready = [
        job
        for job in ready
        if job.input.id in visited or len(visited) < max_jobs
      ]
    if not ready:
      break
    job = ready[0]
    visited.add(job.input.id)
    try:
      fresh(state)
      fresh_job(job)
      if job.status != 'awaiting_review':
        if job.attempts >= state.config.limits.maxAttempts:
          job.status, job.reason = (
            'needs_attention',
            'Trace attempt budget reached',
          )
          save_state(state, 'attempt-limit', job.input.id)
          continue
        timeout = min(
          state.config.limits.workerMs,
          int((deadline - monotonic() - 10) * 1000),
        )
        request = prepare(state, job, 'trace', 'sol', timeout)
        job.attempts += 1
        state.calls += 1
        save_state(
          state,
          'dispatch',
          {
            'job': job.input.id,
            'invocation': request.invocation,
            'provider': 'sol',
            'stage': 'trace',
          },
        )
        result = dispatch(request, state)
        if not result.complete:
          raise ValueError(
            f'sol: {result.error}; invocation {request.invocation}'
          )
        submit(state, job, response_json(result.response), count_attempt=False)
      if job.status == 'awaiting_review':
        for provider in state.config.reviewers:
          if any(
            receipt.provider == provider and receipt.result.verdict == 'pass'
            for receipt in job.reviews
          ):
            continue
          if (
            monotonic() >= deadline - 15
            or state.calls - initial >= state.config.limits.maxCalls
          ):
            reason = 'Invocation budget reached'
            break
          timeout = min(
            state.config.limits.workerMs,
            int((deadline - monotonic() - 10) * 1000),
          )
          reviewed = review_job(state, job, provider, timeout)
          if reviewed.status != 'awaiting_review':
            break
    except (OSError, ValueError, RuntimeError) as error:
      job.status, job.reason = 'needs_attention', str(error)
      save_state(
        state, 'needs-attention', {'job': job.input.id, 'reason': job.reason}
      )
      reason = job.reason
      break
    if reason == 'Invocation budget reached':
      break
  if (
    monotonic() >= deadline - 15
    or state.calls - initial >= state.config.limits.maxCalls
  ):
    reason = 'Invocation budget reached'
  return {
    **status(state),
    'callsThisInvocation': state.calls - initial,
    'stopReason': reason,
  }
