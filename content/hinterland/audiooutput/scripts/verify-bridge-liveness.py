#!/usr/bin/env python3
import json
from pathlib import Path
import struct
import subprocess
import time

project = Path(__file__).resolve().parents[1]
app = project / 'dist/Audio Output.app/Contents/MacOS/AudioOutput'
artifacts = project.parents[2] / 'docs/verification/audiooutput/artifacts'
process = subprocess.Popen(
  [str(app), '--native-host'], stdin=subprocess.PIPE, stdout=subprocess.PIPE
)


def snapshot() -> dict:
  body = json.dumps({'type': 'snapshot'}).encode()
  process.stdin.write(struct.pack('<I', len(body)) + body)
  process.stdin.flush()
  size = struct.unpack('<I', process.stdout.read(4))[0]
  return json.loads(process.stdout.read(size))['configuration']


try:
  before = snapshot()
  assert before['enabled'], (
    'Run the desktop app with routing enabled before this check.'
  )
  time.sleep(12)
  after = snapshot()
  passed = after['enabled']
  report = f'Persistent native-host connection, desktop app running: enabled before={before["enabled"]}, after 12s={after["enabled"]}\n'
  (artifacts / 'bridge-liveness.txt').write_text(report)
  print(report)
  assert passed, 'Bridge marked a running desktop app offline.'
finally:
  process.stdin.close()
  process.wait(timeout=3)
