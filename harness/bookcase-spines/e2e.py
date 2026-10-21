"""Exercise the real CLI, image tools and process boundaries without model calls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

from schema import mapping

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
  ROOT
  / 'docs/verification/bookcase-spine-harness/artifacts/e2e'
  / str(time.time_ns())
)
SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 80"><rect width="20" height="80" fill="#eee9dc"/><path d="M4 12H16V16H4ZM4 24H12V28H4ZM4 64H16V68H4Z" fill="#242424"/></svg>'


def digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
  path.write_text(json.dumps(value, indent=2) + '\n')


def list_value(value: object) -> list[object]:
  if not isinstance(value, list):
    raise ValueError('Expected an array')
  return list(value)


class Pipeline(unittest.TestCase):
  def test_cli(self) -> None:
    EVIDENCE.mkdir(parents=True)
    (EVIDENCE / 'fixture.svg').write_text(SVG)
    reference = EVIDENCE / 'reference.png'
    subprocess.run(
      [
        'rsvg-convert',
        '--width',
        '80',
        '--height',
        '320',
        '--output',
        str(reference),
        str(EVIDENCE / 'fixture.svg'),
      ],
      check=True,
    )
    source = {
      'url': 'https://example.invalid/fixture',
      'sha256': digest(reference),
      'evidence': 'Synthetic geometric fixture, no physical book claim.',
    }
    pinned = {'file': str(reference), 'sha256': digest(reference)}
    input_file = EVIDENCE / 'input.json'
    write(
      input_file,
      {
        'version': 1,
        'jobs': [
          {
            'id': 'fixture',
            'title': 'Fixture',
            'notes': ['fixture.md', 'duplicate-placement.md'],
            'source': source,
            'reference': pinned,
            'front': pinned,
            'blockedReason': None,
          },
          {
            'id': 'gap',
            'title': 'Missing source',
            'notes': ['gap.md'],
            'source': None,
            'reference': None,
            'front': None,
            'blockedReason': 'No observed spine.',
          },
        ],
      },
    )
    run = EVIDENCE / 'run'
    calls: list[dict[str, object]] = []

    def cli(*args: str, expected: int = 0) -> dict[str, object]:
      command = ['bash', 'harness/bookcase-spines/scripts/run.sh', *args]
      result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=45
      )
      calls.append({
        'command': command,
        'exitCode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
      })
      self.assertEqual(
        result.returncode, expected, result.stderr + result.stdout
      )
      return mapping(json.loads(result.stdout))

    try:
      planned = cli('plan', '--input', str(input_file), '--run', str(run))
      self.assertEqual(planned['counts'], {'queued': 1, 'blocked_source': 1})
      self.assertEqual(planned['requestedNotes'], 3)
      self.assertFalse(planned['complete'])
      before = (run / 'state.json').read_bytes()
      cli('plan', '--input', str(input_file), '--run', str(run), expected=1)
      self.assertEqual((run / 'state.json').read_bytes(), before)
      prepared = cli(
        'prepare', '--run', str(run), '--job', 'fixture', '--stage', 'trace'
      )
      self.assertEqual(prepared['provider'], 'argon')
      self.assertIsInstance(prepared['argv'], list)
      self.assertIn('gemini-4-argon-high', list_value(prepared['argv']))
      self.assertIn('none', list_value(prepared['argv']))
      config = mapping(
        json.loads((ROOT / 'harness/bookcase-spines/config.json').read_text())
      )
      self.assertEqual(config['reviewers'], ['astra', 'opus', 'caw'])
      config_file = EVIDENCE / 'config.json'
      for reviewers in [['astra', 'opus'], ['astra', 'opus', 'caw', 'argon']]:
        write(config_file, {**config, 'reviewers': reviewers})
        cli(
          'plan',
          '--input',
          str(input_file),
          '--config',
          str(config_file),
          '--run',
          str(EVIDENCE / 'invalid-policy'),
          expected=1,
        )
      write(
        config_file,
        {
          **config,
          'trace': {
            'provider': 'sol',
            'model': 'gpt-6.1-sol',
            'effort': 'medium',
          },
        },
      )
      sol_run = EVIDENCE / 'sol-run'
      cli(
        'plan',
        '--input',
        str(input_file),
        '--config',
        str(config_file),
        '--run',
        str(sol_run),
      )
      sol = cli('prepare', '--run', str(sol_run), '--job', 'fixture')
      self.assertEqual(sol['provider'], 'sol')
      self.assertIn('read-only', list_value(sol['argv']))
      self.assertIn('gpt-6.1-sol', list_value(sol['argv']))
      cli(
        'prepare',
        '--run',
        str(run),
        '--job',
        'fixture',
        '--provider',
        'sol',
        expected=1,
      )
      response = EVIDENCE / 'response.json'
      candidate = {
        'jobId': 'fixture',
        'inputFingerprint': prepared['inputFingerprint'],
        'status': 'candidate',
        'svg': SVG,
        'provenance': ['Synthetic geometric regions.'],
        'unresolved': [],
      }
      write(response, candidate)
      submitted = cli(
        'submit',
        '--run',
        str(run),
        '--job',
        'fixture',
        '--response',
        str(response),
      )
      self.assertEqual(submitted['status'], 'awaiting_review')
      detail = mapping(mapping(submitted['candidate'])['detail'])
      self.assertEqual((detail['width'], detail['height']), (80, 320))
      cli('verify', '--run', str(run), '--job', 'fixture')
      before = (run / 'state.json').read_bytes()
      for bad in [
        SVG.replace(
          '<path', '<image href="data:image/png;base64,AAAA"/><path'
        ),
        SVG.replace('<path', '<text>invented font</text><path'),
        SVG.replace(
          'fill="#242424"', 'fill="url(https://example.invalid/payload)"'
        ),
        SVG.replace('20 80', '20 Infinity'),
        '<svg><g></svg>',
        SVG.replace('<path', '<g clip-path="url(#loop)" id="loop"/><path'),
        SVG.replace('20 80', '20 40'),
      ]:
        write(response, {**candidate, 'svg': bad})
        cli(
          'submit',
          '--run',
          str(run),
          '--job',
          'fixture',
          '--response',
          str(response),
          expected=1,
        )
        self.assertEqual((run / 'state.json').read_bytes(), before)
      for provider in ['astra', 'opus', 'caw']:
        request = cli(
          'prepare',
          '--run',
          str(run),
          '--job',
          'fixture',
          '--stage',
          'verify',
          '--provider',
          provider,
        )
        argv = list_value(request['argv'])
        if provider == 'astra':
          self.assertIn('gpt-6-astra', argv)
          self.assertIn('read-only', argv)
        if provider == 'opus':
          for flag in [
            '--bare',
            '--restricted',
            '--strict-mcp-config',
            '--no-session-persistence',
          ]:
            self.assertIn(flag, argv)
          self.assertEqual(argv[argv.index('--tools') + 1], 'Read')
          self.assertNotIn('--fallback-model', argv)
        if provider == 'caw':
          self.assertIn('none', argv)
          self.assertIn('ask', argv)
          self.assertNotIn('--force', argv)
      cli(
        'prepare',
        '--run',
        str(run),
        '--job',
        'fixture',
        '--stage',
        'verify',
        '--provider',
        'argon',
        expected=1,
      )
      self.assertEqual(
        cli('run', '--run', str(run), '--dry-run')['modelCalls'], 0
      )
      cli('export', '--run', str(run), expected=1)
      write(run / 'coordinator.lock', {'pid': os.getpid()})
      cli('verify', '--run', str(run), '--job', 'fixture', expected=1)
      cli('recover', '--run', str(run), expected=1)
      (run / 'coordinator.lock').unlink()
      reference.write_text('changed source')
      cli('verify', '--run', str(run), '--job', 'fixture', expected=1)
      self.assertEqual(
        mapping(cli('status', '--run', str(run))['counts'])['stale'], 1
      )
      self.assertEqual(reference.read_text(), 'changed source')
    finally:
      write(EVIDENCE / 'commands.json', calls)
      print(f'Evidence: {EVIDENCE}', file=sys.stderr)

  def test_process_limits(self) -> None:
    from process import execute

    directory = EVIDENCE / 'process-checks'
    directory.mkdir(parents=True, exist_ok=True)
    timeout = execute(
      [sys.executable, '-c', 'import time; time.sleep(20)'],
      directory / 'timeout',
      timeout=0.15,
      max_bytes=1024,
    )
    self.assertEqual(timeout.error, 'Process timed out')
    overflow = execute(
      [sys.executable, '-c', 'import sys; sys.stdout.write("x" * 1000000)'],
      directory / 'overflow',
      timeout=3,
      max_bytes=1024,
    )
    self.assertEqual(overflow.error, 'Process output limit exceeded')
    self.assertLessEqual(Path(overflow.stdoutFile).stat().st_size, 1024)


if __name__ == '__main__':
  unittest.main()
