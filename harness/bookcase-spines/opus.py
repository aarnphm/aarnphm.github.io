"""Claude API-credit authentication and native visual-review transport evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex

from schema import Transport, mapping, optional_text


def api_environment(repository: Path) -> dict[str, str]:
  names = {'CLAUDE_API_KEY', 'CLAUDE_WORKSPACE_ID', 'ANTHROPIC_WORKSPACE_ID'}
  values = {name: os.environ[name] for name in names if os.environ.get(name)}
  dotenv = repository / '.env'
  if dotenv.is_file():
    for line in dotenv.read_text().splitlines():
      name, separator, raw = line.removeprefix('export ').partition('=')
      name = name.strip()
      if separator and name in names and name not in values:
        # Parse only these assignments, without shell evaluation or interpolation.
        parts = shlex.split(raw, comments=True)
        if len(parts) != 1:
          raise ValueError(f'Expected one literal value for {name} in .env')
        values[name] = parts[0]
  key = values.get('CLAUDE_API_KEY')
  if not key:
    raise ValueError(
      'Opus requires CLAUDE_API_KEY in the environment or repository .env'
    )
  env = dict(os.environ)
  for name in [
    'ANTHROPIC_AUTH_TOKEN',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'CLAUDE_CODE_USE_BEDROCK',
    'CLAUDE_CODE_USE_VERTEX',
    'CLAUDE_CODE_USE_FOUNDRY',
  ]:
    env.pop(name, None)
  env['ANTHROPIC_API_KEY'] = key
  workspace = values.get('CLAUDE_WORKSPACE_ID') or values.get(
    'ANTHROPIC_WORKSPACE_ID'
  )
  if workspace:
    headers = [
      line
      for line in env.get('ANTHROPIC_CUSTOM_HEADERS', '').splitlines()
      if not line.lower().startswith('anthropic-workspace-id:')
    ]
    env['ANTHROPIC_CUSTOM_HEADERS'] = '\n'.join([
      *headers,
      f'anthropic-workspace-id: {workspace}',
    ])
  return env


def arguments(
  model: str, effort: str, schema: object, images: list[str]
) -> list[str]:
  argv = [
    '-p',
    '--bare',
    '--restricted',
    '--no-session-persistence',
    '--disable-slash-commands',
    '--strict-mcp-config',
    '--mcp-config',
    '{"mcpServers":{}}',
    '--tools',
    'Read',
    '--allowedTools',
    'Read',
    '--permission-mode',
    'dontAsk',
    '--permission-prompts',
    'none',
    '--model',
    model,
    '--effort',
    effort,
    '--output-format',
    'stream-json',
    '--verbose',
    '--json-schema',
    json.dumps(schema),
  ]
  for parent in sorted({str(Path(file).parent) for file in images}):
    argv.extend(['--add-dir', parent])
  return argv


def inspect(output: str, images: list[str], model: str | None) -> Transport:
  events = [
    mapping(json.loads(line)) for line in output.splitlines() if line.strip()
  ]
  terminal = next(
    (event for event in reversed(events) if event.get('type') == 'result'), {}
  )
  initial = next(
    (event for event in events if event.get('subtype') == 'init'), {}
  )
  reads: dict[str, str] = {}
  inspected: set[str] = set()
  errors: list[str] = []
  for event in events:
    message = event.get('message')
    if not isinstance(message, dict):
      continue
    content = mapping(message).get('content', [])
    if not isinstance(content, list):
      continue
    for value in content:
      block = mapping(value)
      if block.get('type') == 'tool_use' and block.get('name') == 'Read':
        file = mapping(block.get('input')).get('file_path')
        identity = block.get('id')
        if isinstance(file, str) and isinstance(identity, str):
          reads[identity] = file
      if block.get('type') == 'tool_result':
        identity = block.get('tool_use_id')
        if block.get('is_error') is True:
          errors.append('Claude reported a tool failure')
        elif isinstance(identity, str) and identity in reads:
          inspected.add(reads[identity])
  if not set(images).issubset(inspected):
    errors.append('Claude did not complete reads of every supplied image')
  if terminal.get('is_error') is True:
    details = terminal.get('errors') or terminal.get('result')
    errors.append(f'Claude reported an error: {details}')
  if terminal.get('permission_denials'):
    errors.append('Claude reported denied tool requests')
  models = mapping(terminal.get('modelUsage') or {})
  observed = optional_text(initial.get('model'))
  if models:
    observed = ', '.join(models)
  if observed != model:
    errors.append(f'Expected {model}, observed {observed}')
  structured = terminal.get('structured_output')
  response = (
    json.dumps(structured)
    if isinstance(structured, dict)
    else optional_text(terminal.get('result')) or ''
  )
  return Transport(
    'opus',
    terminal.get('subtype') == 'success' and not errors and bool(response),
    None,
    response,
    optional_text(terminal.get('session_id')),
    observed,
    '; '.join(errors) or None,
    terminal.get('usage'),
  )
