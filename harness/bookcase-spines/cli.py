#!/usr/bin/env python3
"""Plan, trace, review and export pure SVG book spines using Python and native codecs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from files import (
  HARNESS,
  dumps,
  fresh,
  fresh_job,
  locked,
  read_json,
  read_state,
  recover,
  run_path,
  save_state,
)
from jobs import export_bundle, submit, verify
from plan import job_by_id, plan, status
from process import budget
from providers import prepare
from runner import review_job, run
from schema import Config, Provider, decode


def doctor() -> dict[str, object]:
  config = decode(Config, read_json(HARNESS / 'config.json'))
  config.validate()
  commands = [
    config.providers.sol.command,
    config.providers.argon.command,
    config.providers.caw.command,
    'rsvg-convert',
    'cwebp',
    'dwebp',
  ]
  return {
    'python': sys.version.split()[0],
    'availability': [
      {'command': command, 'path': shutil.which(command)}
      for command in commands
    ],
    'inspectors': {
      name: (
        Path.home() / '.codex/skills' / name / 'scripts/inspect_run.py'
      ).is_file()
      for name in ['argon', 'caw']
    },
    'trace': config.trace.__dict__,
    'reviewers': config.reviewers,
    'accountsVerified': False,
  }


def main() -> object:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    'command',
    choices=[
      'doctor',
      'plan',
      'status',
      'prepare',
      'submit',
      'run',
      'resume',
      'review',
      'verify',
      'export',
      'recover',
    ],
  )
  parser.add_argument('--run')
  parser.add_argument('--input', type=Path)
  parser.add_argument('--config', type=Path)
  parser.add_argument('--job')
  parser.add_argument('--stage', choices=['trace', 'verify'], default='trace')
  parser.add_argument(
    '--provider', choices=['sol', 'argon', 'caw'], default='sol'
  )
  parser.add_argument('--response', type=Path)
  parser.add_argument('--max-jobs', type=int, default=1)
  parser.add_argument('--until-done', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  args = parser.parse_args()
  if args.command == 'doctor':
    return doctor()
  if not args.run:
    raise ValueError('--run is required')
  if args.max_jobs < 1:
    raise ValueError('--max-jobs must be positive')
  directory = run_path(args.run)
  if args.command == 'plan':
    return status(plan(directory, args.input, args.config))
  if args.command == 'status':
    return status(read_state(directory))
  if args.command == 'recover':
    recover(directory)
    return {'recovered': str(directory)}
  with locked(directory):
    state = read_state(directory)
    fresh(state)
    if args.command in ['run', 'resume']:
      if args.job:
        job_by_id(state, args.job)
      if args.command == 'resume' and not args.dry_run:
        for job in state.jobs:
          if (
            (not args.job or args.job == job.input.id)
            and job.status == 'needs_attention'
            and not any(
              receipt.result.verdict in ['blocked_source', 'needs_review']
              for receipt in job.reviews
            )
          ):
            job.status = (
              (
                'revise'
                if any(
                  receipt.result.verdict == 'revise' for receipt in job.reviews
                )
                else 'awaiting_review'
              )
              if job.candidate
              else 'queued'
            )
            job.reason = None
        save_state(state, 'resume')
      with budget(state.config.limits.invocationMs / 1000):
        return run(
          state,
          identity=args.job,
          max_jobs=args.max_jobs,
          until_done=args.until_done,
          dry_run=args.dry_run,
        )
    if args.command == 'export':
      return export_bundle(state)
    if not args.job:
      raise ValueError('--job is required')
    job = job_by_id(state, args.job)
    fresh_job(job)
    provider: Provider = args.provider
    if args.command == 'prepare':
      return prepare(state, job, args.stage, provider)
    if args.command == 'submit':
      if not args.response:
        raise ValueError('--response is required')
      submit(state, job, read_json(args.response))
      return job
    if args.command == 'verify':
      verify(state, job)
      return {
        'id': job.input.id,
        'deterministic': 'pass',
        'status': job.status,
      }
    if args.command == 'review':
      try:
        return review_job(state, job, provider)
      except (OSError, ValueError, RuntimeError) as error:
        job.status, job.reason = 'needs_attention', str(error)
        save_state(
          state,
          'review-incomplete',
          {'job': job.input.id, 'reason': job.reason},
        )
        raise
  raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
  try:
    print(dumps(main()))
  except KeyboardInterrupt:
    print(
      dumps({
        'error': 'Interrupted; inspect saved provider logs before resuming'
      })
    )
    sys.exit(130)
  except (OSError, ValueError, RuntimeError) as error:
    print(dumps({'error': str(error)}))
    sys.exit(1)
