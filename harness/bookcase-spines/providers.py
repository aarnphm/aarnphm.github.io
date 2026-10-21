"""Fresh bounded implementation and review calls with native transport inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys
from uuid import uuid4

from files import (
  HARNESS,
  REPOSITORY,
  check_pin,
  dumps,
  fresh,
  fresh_job,
  read_json,
  sha,
  write_json,
)
from process import execute
import opus
from schema import (
  Job,
  Provider,
  Review,
  Stage,
  State,
  Trace,
  Transport,
  json_schema,
  decode,
  mapping,
  optional_text,
  strict_json,
)
from svg import dimensions


@dataclass
class Prepared:
  invocation: str
  directory: str
  provider: Provider
  stage: Stage
  executable: str
  argv: list[str]
  inputFingerprint: str
  request: dict[str, object]
  timeoutMs: int


def prepare(
  state: State,
  job: Job,
  stage: Stage,
  provider: Provider,
  timeout_ms: int | None = None,
) -> Prepared:
  fresh(state)
  fresh_job(job)
  if stage == 'trace' and provider != state.config.trace.provider:
    raise ValueError(
      'Trace provider must match the planned implementation worker'
    )
  if stage == 'verify' and provider == state.config.trace.provider:
    raise ValueError(
      'The implementation worker cannot review its own candidate'
    )
  if (
    job.input.blockedReason
    or not job.input.reference
    or not job.input.source
    or not job.input.front
  ):
    raise ValueError('Job lacks admitted references')
  if stage == 'verify' and not job.candidate:
    raise ValueError('No rasterized candidate to review')
  invocation = str(uuid4())
  directory = Path(state.run) / 'invocations' / invocation
  directory.mkdir(parents=True)
  backend = state.config.providers.get(provider)
  model = state.config.trace.model if stage == 'trace' else backend.model
  effort = state.config.trace.effort if stage == 'trace' else backend.effort
  schema = json_schema(Trace if stage == 'trace' else Review)
  schema_file = directory / 'schema.json'
  write_json(schema_file, schema)
  images = [job.input.reference]
  if stage == 'verify' and job.candidate:
    for value in [
      job.candidate.master,
      job.candidate.shelf,
      job.candidate.detail,
    ]:
      check_pin(value)
    images.extend([job.candidate.shelf, job.candidate.detail])
  width, height = dimensions(Path(job.input.reference.file))
  request: dict[str, object] = {
    'jobId': job.input.id,
    'title': job.input.title,
    'modelSelection': {'requestedModel': model, 'requestedEffort': effort},
    'inputFingerprint': job.inputFingerprint,
    'source': asdict(job.input.source),
    'images': [asdict(value) for value in images],
    'viewBox': f'0 0 {width} {height}',
    'candidateSha256': job.candidate.master.sha256 if job.candidate else None,
    'rubricSha256': sha((HARNESS / 'prompts/verify.md').read_bytes()),
    'limits': asdict(state.config.limits),
    'resultFile': str(directory / 'trace-result.json')
    if stage == 'trace'
    else None,
    'priorDefects': [
      asdict(defect)
      for receipt in job.reviews
      for defect in receipt.result.defects
    ]
    if stage == 'trace'
    else [],
  }
  prompt = (
    (HARNESS / 'prompts' / f'{stage}.md').read_text()
    + '\n\nRequest data:\n'
    + dumps(request)
    + '\n\nFinal response JSON schema:\n'
    + dumps(schema)
    + '\n'
  )
  (directory / 'task.txt').write_text(prompt)
  timeout_ms = timeout_ms or state.config.limits.workerMs
  if provider in ['sol', 'astra']:
    if not model or not effort:
      raise ValueError('Codex requires an explicit model and effort')
    argv = [
      'exec',
      '--cd',
      str(directory),
      '--skip-git-repo-check',
      '--ephemeral',
      '--ignore-user-config',
      '--sandbox',
      'workspace-write' if stage == 'trace' else 'read-only',
      '-c',
      'approval_policy="never"',
      '--model',
      model,
      '-c',
      f'model_reasoning_effort="{effort}"',
      '--json',
      '--output-schema',
      str(schema_file),
      '--output-last-message',
      str(directory / 'final.json'),
    ]
    for value in images:
      argv.extend(['--image', value.file])
    argv.append('-')
  elif provider == 'opus':
    if not model or not effort:
      raise ValueError('Opus requires an explicit model and effort')
    argv = opus.arguments(
      model, effort, schema, [value.file for value in images]
    )
  else:
    argv = [
      '--workspace',
      str(directory),
      '--browser',
      'none',
      '--stream',
      '--no-auto-title',
      '--timeout',
      str(max(1, timeout_ms // 1000 - 5)),
    ]
    if model:
      argv.extend(['--model', model])
    if provider == 'caw':
      argv.extend(['--force'] if stage == 'trace' else ['--mode', 'ask'])
  prepared = Prepared(
    invocation,
    str(directory),
    provider,
    stage,
    backend.command,
    argv,
    job.inputFingerprint,
    request,
    timeout_ms,
  )
  write_json(directory / 'request.json', prepared)
  return prepared


def inspect_codex(
  directory: Path, output: str, provider: Provider
) -> Transport:
  events = [
    mapping(strict_json(line)) for line in output.splitlines() if line.strip()
  ]
  completed = next(
    (
      event
      for event in reversed(events)
      if event.get('type') == 'turn.completed'
    ),
    None,
  )
  failed = any(
    event.get('type') in ['turn.failed', 'error'] for event in events
  )
  thread = next(
    (event for event in events if event.get('type') == 'thread.started'), {}
  )
  final = directory / 'final.json'
  response = final.read_text() if final.exists() else ''
  return Transport(
    provider,
    bool(
      completed and events[-1] is completed and not failed and response.strip()
    ),
    None,
    response,
    optional_text(thread.get('thread_id')),
    None,
    'Codex reported a failed turn' if failed else None,
    completed.get('usage') if completed else None,
  )


def dispatch(prepared: Prepared, state: State) -> Transport:
  directory = Path(prepared.directory)
  family = {
    'sol': 'codex',
    'astra': 'codex',
    'opus': 'claude',
    'argon': 'argon',
    'caw': 'cursor',
  }[prepared.provider]
  logs = REPOSITORY / 'work' / family / 'bookcase-spines' / prepared.invocation
  logs.mkdir(parents=True)
  prompt = (directory / 'task.txt').read_text()
  (logs / 'task.txt').write_text(prompt)
  result = execute(
    [prepared.executable, *prepared.argv],
    logs / 'process',
    timeout=prepared.timeoutMs / 1000,
    max_bytes=state.config.limits.maxOutputBytes,
    stdin=prompt,
    active=directory / 'active.json',
    env=opus.api_environment(REPOSITORY)
    if prepared.provider == 'opus'
    else None,
  )
  try:
    if prepared.provider in ['sol', 'astra']:
      report = inspect_codex(
        directory, Path(result.stdoutFile).read_text(), prepared.provider
      )
    elif prepared.provider == 'opus':
      images = prepared.request['images']
      if not isinstance(images, list):
        raise ValueError('Expected image list')
      paths = [optional_text(mapping(value).get('file')) for value in images]
      report = opus.inspect(
        Path(result.stdoutFile).read_text(),
        [file for file in paths if file is not None],
        state.config.providers.opus.model,
      )
    else:
      helper = (
        Path.home()
        / '.codex/skills'
        / prepared.provider
        / 'scripts/inspect_run.py'
      )
      inspection = execute(
        [
          sys.executable,
          str(helper),
          result.stdoutFile,
          '--exit-code',
          str(result.exitCode if result.exitCode is not None else 1),
          '--stderr',
          result.stderrFile,
        ],
        logs / 'inspection',
        timeout=10,
        max_bytes=state.config.limits.maxOutputBytes,
      )
      summary = mapping(read_json(Path(inspection.stdoutFile)))
      terminal = mapping(summary.get('terminal') or {})
      response = summary.get('response') or terminal.get('result') or ''
      errors = summary.get('tool_error_count', 0)
      models = summary.get('observed_models') or []
      model = summary.get('model')
      if model is None and isinstance(models, list) and models:
        if len(models) != 1:
          raise ValueError('Provider reported multiple models in one invocation')
        model = models[0]
      report = Transport(
        prepared.provider,
        inspection.exitCode == 0
        and summary.get('transport_complete') is True
        and errors == 0,
        result.exitCode,
        response if isinstance(response, str) else '',
        optional_text(summary.get('conversation_id')),
        optional_text(model),
        'Provider reported tool failures' if errors else None,
        summary.get('usage') or terminal.get('usage'),
      )
      if (
        prepared.provider == 'caw'
        and prepared.stage == 'trace'
        and report.complete
      ):
        file = directory / 'trace-result.json'
        if file.resolve().parent != directory.resolve() or not file.is_file():
          raise ValueError(
            'Grok trace needs its invocation-bound JSON result file'
          )
        if file.stat().st_size > state.config.limits.maxOutputBytes:
          raise ValueError('Trace JSON result exceeds the output budget')
        payload = file.read_text()
        trace = decode(Trace, response_json(payload))
        if (
          trace.inputFingerprint != prepared.inputFingerprint
          or trace.jobId != prepared.request['jobId']
        ):
          raise ValueError('Trace JSON result belongs to a different request')
        write_json(
          directory / 'response-source.json',
          {
            'kind': 'provider-file',
            'file': str(file),
            'sha256': sha(payload),
            'nativeResponseSha256': sha(report.response),
          },
        )
        report.response = payload
  except (OSError, ValueError, KeyError) as error:
    report = Transport(
      prepared.provider,
      False,
      result.exitCode,
      '',
      None,
      None,
      str(error),
      None,
    )
  report.exitCode = result.exitCode
  report.complete = (
    report.complete
    and result.exitCode == 0
    and result.error is None
    and bool(report.response.strip())
  )
  if prepared.provider == 'argon':
    requested = mapping(prepared.request['modelSelection']).get(
      'requestedModel'
    )
    if requested and report.model != requested:
      report.complete = False
      report.error = 'Argon did not confirm the requested model'
  if prepared.provider == 'caw' and prepared.stage == 'trace':
    observed = (report.model or '').lower()
    if not (
      'grok' in observed
      and '4.7' in observed
      and '500' in observed
      and ('extra high' in observed or 'xhigh' in observed)
      and 'fast' not in observed.replace('fast=false', '')
    ):
      report.complete = False
      report.error = 'Grok trace did not confirm Extra High, 500K and Fast disabled'
  report.error = (
    result.error
    or report.error
    or (
      None
      if report.complete
      else 'Incomplete provider transport; inspect raw logs'
    )
  )
  write_json(directory / 'process.json', result)
  write_json(
    directory / 'model-provenance.json',
    {
      **mapping(prepared.request['modelSelection']),
      'observedModel': report.model,
      'observationSource': 'native-provider-stream'
      if report.model
      else 'unreported',
      'rawOutput': result.stdoutFile,
      'rawOutputSha256': sha(Path(result.stdoutFile).read_bytes()),
    },
  )
  write_json(directory / 'transport.json', report)
  return report


def response_json(response: str) -> object:
  match = re.fullmatch(r'```(?:json)?\s*\n(.*)\n```', response.strip(), re.S)
  return strict_json(match[1] if match else response)
