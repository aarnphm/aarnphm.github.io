"""Candidate replacement, independent review receipts and verified bundle export."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
from uuid import UUID, uuid4

from files import (
  HARNESS,
  check_pin,
  dumps,
  fresh,
  fresh_job,
  pin,
  read_json,
  save_state,
  sha,
  write_json,
)
from providers import response_json
from schema import (
  Job,
  Provider,
  Receipt,
  Review,
  State,
  Trace,
  Transport,
  decode,
)
from svg import rasterize


def trace_svg(state: State, job: Job, result: Trace) -> str:
  artifact = result.svgFile
  if artifact is None:
    return result.svg
  if result.svg:
    raise ValueError('Supply inline SVG or an SVG file, never both')
  file = Path(artifact.file).resolve(strict=True)
  relative = file.relative_to((Path(state.run) / 'invocations').resolve())
  if len(relative.parts) < 2:
    raise ValueError('SVG file must belong to an invocation')
  UUID(relative.parts[0])
  request = read_json(
    Path(state.run) / 'invocations' / relative.parts[0] / 'request.json'
  )
  if (
    not isinstance(request, dict)
    or request.get('stage') != 'trace'
    or request.get('inputFingerprint') != job.inputFingerprint
    or request.get('provider') != state.config.trace.provider
  ):
    raise ValueError('SVG file belongs to a different trace request')
  if (
    not file.is_file() or file.stat().st_size > state.config.limits.maxSvgBytes
  ):
    raise ValueError(
      'SVG file exceeds the byte budget or is not a regular file'
    )
  data = file.read_bytes()
  if sha(data) != artifact.sha256:
    raise ValueError('SVG file hash differs')
  return data.decode('utf-8')


def submit(
  state: State, job: Job, response: object, count_attempt: bool = True
) -> None:
  fresh(state)
  fresh_job(job)
  result = decode(Trace, response)
  if (
    result.jobId != job.input.id
    or result.inputFingerprint != job.inputFingerprint
  ):
    raise ValueError('Trace belongs to a different job or input revision')
  if result.status == 'blocked_source':
    if (
      not result.unresolved
      or result.svg
      or result.svgFile
      or any(not item.strip() for item in result.unresolved)
    ):
      raise ValueError('Blocked trace needs evidence gaps and no SVG')
    job.status, job.reason = 'blocked_source', '\n'.join(result.unresolved)
  else:
    if not job.input.reference or job.input.blockedReason:
      raise ValueError('Unadmitted source')
    svg = trace_svg(state, job, result)
    if (
      not svg.strip()
      or not result.provenance
      or any(not item.strip() for item in result.provenance)
      or result.unresolved
    ):
      raise ValueError('Candidate lacks provenance or has unresolved regions')
    directory = Path(state.run) / 'attempts' / job.input.id / str(uuid4())
    candidate = rasterize(
      svg,
      Path(job.input.reference.file),
      directory,
      state.config,
      result.provenance,
    )
    fresh(state)
    fresh_job(job)
    job.candidate, job.reviews = candidate, []
    job.status, job.reason = 'awaiting_review', None
  if count_attempt:
    job.attempts += 1
  save_state(state, 'submit', {'id': job.input.id, 'status': job.status})


def validate_review(
  state: State, job: Job, provider: Provider, invocation: str, result: Review
) -> str:
  fresh(state)
  fresh_job(job)
  candidate = job.candidate
  if not candidate or not job.input.reference:
    raise ValueError('No candidate to review')
  for value in [candidate.master, candidate.shelf, candidate.detail]:
    check_pin(value)
  rubric = sha((HARNESS / 'prompts/verify.md').read_bytes())
  if (
    result.jobId,
    result.inputFingerprint,
    result.candidateSha256,
    result.rubricSha256,
  ) != (job.input.id, job.inputFingerprint, candidate.master.sha256, rubric):
    raise ValueError('Stale or mismatched review')
  expected = sorted([
    job.input.reference.sha256,
    candidate.shelf.sha256,
    candidate.detail.sha256,
  ])
  if sorted(result.inspectedImages) != expected:
    raise ValueError(
      'Review did not inspect exactly the supplied image hashes'
    )
  if result.verdict == 'pass' and (
    result.defects
    or not all(asdict(result.categories).values())
    or not all(
      value for tier in asdict(result.tiers).values() for value in tier.values()
    )
  ):
    raise ValueError('Passing review contains failed categories or defects')
  if result.verdict != 'pass' and not result.defects:
    raise ValueError('Nonpassing review needs concrete defects')
  if any(
    not value.strip()
    for defect in result.defects
    for value in asdict(defect).values()
  ):
    raise ValueError('Review defect is empty')
  UUID(invocation)
  directory = Path(state.run) / 'invocations' / invocation
  transport = decode(Transport, read_json(directory / 'transport.json'))
  if (
    not transport.complete
    or transport.provider != provider
    or transport.exitCode != 0
  ):
    raise ValueError('Review transport is incomplete')
  if dumps(response_json(transport.response)) != dumps(result):
    raise ValueError('Review differs from provider response')
  request = read_json(directory / 'request.json')
  if (
    not isinstance(request, dict)
    or request.get('invocation') != invocation
    or request.get('directory') != str(directory)
    or request.get('provider') != provider
    or request.get('stage') != 'verify'
    or request.get('inputFingerprint') != job.inputFingerprint
  ):
    raise ValueError('Review invocation does not match its request')
  bound = request.get('request')
  if (
    not isinstance(bound, dict)
    or bound.get('jobId') != job.input.id
    or bound.get('inputFingerprint') != job.inputFingerprint
    or bound.get('images')
    != [
      asdict(value)
      for value in [job.input.reference, candidate.shelf, candidate.detail]
    ]
    or bound.get('candidateSha256') != candidate.master.sha256
    or bound.get('rubricSha256') != rubric
  ):
    raise ValueError('Review invocation was bound to different artwork')
  return pin(directory / 'transport.json').sha256


def accept_review(
  state: State, job: Job, provider: Provider, invocation: str, response: object
) -> None:
  result = decode(Review, response)
  transport_sha = validate_review(state, job, provider, invocation, result)
  job.reviews = [
    receipt for receipt in job.reviews if receipt.provider != provider
  ]
  job.reviews.append(Receipt(provider, invocation, transport_sha, result))
  failure = next(
    (
      receipt.result
      for receipt in job.reviews
      if receipt.result.verdict != 'pass'
    ),
    None,
  )
  if failure and failure.verdict != 'pass':
    job.status = (
      'needs_attention'
      if failure.verdict == 'needs_review'
      else failure.verdict
    )
    job.reason = dumps([asdict(defect) for defect in failure.defects])
  else:
    job.status = (
      'accepted'
      if all(
        any(receipt.provider == required for receipt in job.reviews)
        for required in state.config.reviewers
      )
      else 'awaiting_review'
    )
    job.reason = None
  write_json(
    Path(state.run) / 'invocations' / invocation / 'review.json', result
  )
  save_state(
    state,
    'review',
    {
      'id': job.input.id,
      'provider': provider,
      'invocation': invocation,
      'status': job.status,
    },
  )


def verify(state: State, job: Job) -> None:
  fresh(state)
  fresh_job(job)
  candidate = job.candidate
  if not candidate or not job.input.reference:
    raise ValueError('No candidate available')
  for value in [candidate.master, candidate.shelf, candidate.detail]:
    check_pin(value)
  directory = Path(state.run) / 'verification' / job.input.id / str(uuid4())
  replay = rasterize(
    Path(candidate.master.file).read_text(),
    Path(job.input.reference.file),
    directory,
    state.config,
    candidate.provenance,
  )
  if (candidate.recipe, candidate.shelf.sha256, candidate.detail.sha256) != (
    replay.recipe,
    replay.shelf.sha256,
    replay.detail.sha256,
  ):
    raise ValueError('Candidate export replay differs')
  for receipt in job.reviews:
    if receipt.transportSha256 != validate_review(
      state, job, receipt.provider, receipt.invocation, receipt.result
    ):
      raise ValueError('Review receipt changed')
    bound = read_json(
      Path(state.run) / 'invocations' / receipt.invocation / 'review.json'
    )
    if dumps(bound) != dumps(receipt.result):
      raise ValueError('Saved review differs from state')
  write_json(
    directory / 'verification.json',
    {
      'jobId': job.input.id,
      'master': candidate.master.sha256,
      'shelf': replay.shelf.sha256,
      'detail': replay.detail.sha256,
      'deterministic': 'pass',
      'visual': job.status,
      'renderedPlacementVerified': False,
    },
  )


def export_bundle(state: State) -> dict[str, object]:
  accepted = [job for job in state.jobs if job.status == 'accepted']
  if not accepted:
    raise ValueError('No accepted jobs to export')
  for job in accepted:
    if any(
      receipt.result.verdict != 'pass' for receipt in job.reviews
    ) or not all(
      any(
        receipt.provider == provider and receipt.result.verdict == 'pass'
        for receipt in job.reviews
      )
      for provider in state.config.reviewers
    ):
      raise ValueError('Required independent reviews are missing')
    verify(state, job)
  directory = Path(state.run) / 'exports' / str(uuid4())
  directory.mkdir(parents=True)
  entries: list[dict[str, object]] = []
  for job in accepted:
    candidate = job.candidate
    if not candidate:
      raise ValueError('Accepted job has no candidate')
    output: list[dict[str, object]] = []
    for name, value in [
      ('master', candidate.master),
      ('shelf', candidate.shelf),
      ('detail', candidate.detail),
    ]:
      destination = (
        directory
        / f'{job.input.id}-{name}-{value.sha256[:16]}{Path(value.file).suffix}'
      )
      shutil.copyfile(value.file, destination)
      copied = pin(destination)
      if copied.sha256 != value.sha256:
        raise ValueError('Export copy differs')
      output.append(asdict(copied))
    entries.append({
      'jobId': job.input.id,
      'inputFingerprint': job.inputFingerprint,
      'source': asdict(job.input.source) if job.input.source else None,
      'files': output,
      'reviews': [asdict(receipt) for receipt in job.reviews],
    })
  fresh(state)
  receipt = {
    'version': 1,
    'directory': str(directory),
    'requested': len(state.jobs),
    'accepted': len(entries),
    'complete': len(entries) == len(state.jobs),
    'productionPublished': False,
    'renderedPlacementVerified': False,
    'entries': entries,
  }
  write_json(directory / 'manifest.json', receipt)
  return receipt
