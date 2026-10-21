"""Admit the whole catalogue and retain explicit unresolved edition jobs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

from files import (
  HARNESS,
  REPOSITORY,
  check_pin,
  dumps,
  fresh,
  fresh_job,
  implementation_hash,
  now,
  pin,
  read_json,
  save_state,
  sha,
)
from schema import Config, InputJob, Inputs, Job, Pin, Source, State, decode


def garden_input() -> tuple[list[InputJob], list[Pin]]:
  catalogue_file = REPOSITORY / 'quartz/util/base/bookcase/jackets.json'
  ledger_file = (
    REPOSITORY
    / 'docs/verification/bookcase-spine-art/research/round-22/open-ledger.json'
  )
  catalogue, ledger = read_json(catalogue_file), read_json(ledger_file)
  if not isinstance(catalogue, dict) or not isinstance(ledger, dict):
    raise ValueError('Invalid source handoff')
  entries, gaps = catalogue['entries'], ledger['entries']
  if (
    ledger['sourced'] != len(entries)
    or ledger['open'] != len(gaps)
    or ledger['books'] != len(entries) + len(gaps)
  ):
    raise ValueError(
      'Sourcing ledger and catalogue disagree; refresh the source handoff'
    )
  jobs = [
    InputJob(
      entry['id'],
      entry['title'],
      [entry['poster']],
      Source(entry['source'], entry['sourceSha256'], entry['editionEvidence']),
      Pin(
        str(
          REPOSITORY / 'quartz/static/bookcase-spines' / entry['detailArtwork']
        ),
        entry['detailArtworkSha256'],
      ),
      Pin(
        str(REPOSITORY / 'content' / entry['poster']), entry['posterSha256']
      ),
      None,
    )
    for entry in entries
  ]
  jobs.extend(
    InputJob(
      'gap-' + sha(entry['note'])[:16],
      entry['title'],
      [entry['note']],
      None,
      None,
      Pin(
        str(REPOSITORY / 'content' / entry['poster']), entry['posterSha256']
      ),
      entry['gap'],
    )
    for entry in gaps
  )
  return jobs, [pin(catalogue_file), pin(ledger_file)]


def plan(
  directory: Path, input_file: Path | None, config_file: Path | None
) -> State:
  config_pin = pin(config_file or HARNESS / 'config.json')
  config = decode(Config, read_json(Path(config_pin.file)))
  config.validate()
  implementation = implementation_hash()
  if input_file:
    entries = decode(Inputs, read_json(input_file)).jobs
    sources = [pin(input_file)]
  else:
    entries, sources = garden_input()
  if not entries:
    raise ValueError('Input has no jobs')
  ids: set[str] = set()
  notes: set[str] = set()
  jobs: list[Job] = []
  for entry in entries:
    if (
      not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,150}', entry.id)
      or entry.id in ids
    ):
      raise ValueError(f'Invalid or duplicate job ID: {entry.id}')
    ids.add(entry.id)
    if (
      not entry.title.strip()
      or not entry.notes
      or any(not note or note in notes for note in entry.notes)
      or len(set(entry.notes)) != len(entry.notes)
    ):
      raise ValueError('Missing title or duplicate note identity')
    notes.update(entry.notes)
    for value in [entry.reference, entry.front]:
      if value:
        value.file = str(
          (
            (input_file.resolve().parent if input_file else REPOSITORY)
            / value.file
          ).resolve()
        )
        check_pin(value)
    reason = entry.blockedReason
    if not entry.source or not entry.reference or not entry.front:
      reason = reason or 'Missing edition evidence, reference or front.'
    elif (
      not entry.source.evidence.strip()
      or not re.fullmatch(r'[a-f0-9]{64}', entry.source.sha256)
      or not entry.source.url.startswith(('https://', 'http://', 'file:///'))
    ):
      raise ValueError('Invalid source provenance')
    fingerprint = sha(
      dumps({
        'entry': asdict(entry),
        'config': asdict(config),
        'implementation': implementation,
      })
    )
    jobs.append(
      Job(
        entry,
        fingerprint,
        'blocked_source' if reason else 'queued',
        reason,
        0,
        None,
        [],
      )
    )
  state = State(
    1,
    now(),
    str(directory),
    config,
    config_pin,
    implementation,
    sources,
    jobs,
    0,
  )
  directory.parent.mkdir(parents=True, exist_ok=True)
  directory.mkdir()
  save_state(state, 'plan', {'jobs': len(jobs)})
  return state


def job_by_id(state: State, identity: str) -> Job:
  for job in state.jobs:
    if job.input.id == identity:
      return job
  raise ValueError(f'Unknown job: {identity}')


def status(state: State) -> dict[str, object]:
  stale_reason = None
  try:
    fresh(state)
  except (OSError, ValueError) as error:
    stale_reason = str(error)
  counts: dict[str, int] = {}
  rows: list[dict[str, object]] = []
  for job in state.jobs:
    reason = stale_reason
    try:
      fresh_job(job)
    except (OSError, ValueError) as error:
      reason = str(error)
    current = (
      'stale' if reason and job.status != 'blocked_source' else job.status
    )
    counts[current] = counts.get(current, 0) + 1
    rows.append({
      'id': job.input.id,
      'title': job.input.title,
      'status': current,
      'attempts': job.attempts,
      'reason': reason or job.reason,
    })
  return {
    'run': state.run,
    'requestedJobs': len(rows),
    'requestedNotes': len({
      note for job in state.jobs for note in job.input.notes
    }),
    'counts': counts,
    'complete': all(row['status'] == 'accepted' for row in rows),
    'modelCalls': state.calls,
    'jobs': rows,
  }
