#!/usr/bin/env python3
import json
from pathlib import Path
import struct
import subprocess

project = Path(__file__).resolve().parents[1]
app = project / 'dist/Audio Output.app/Contents/MacOS/AudioOutput'
artifacts = project.parents[2] / 'docs/verification/audiooutput/artifacts'
artifacts.mkdir(parents=True, exist_ok=True)
requests = [
  {'type': 'snapshot'},
  {'type': 'unknown'},
  {'type': 'status', 'host': '../../outside'},
  {'type': 'snapshot'},
]
wire = b''
for request in requests:
  data = json.dumps(request).encode()
  wire += struct.pack('<I', len(data)) + data
result = subprocess.run(
  [str(app), '--native-host'],
  input=wire,
  capture_output=True,
  timeout=5,
  check=True,
)
responses = []
remaining = result.stdout
while remaining:
  size = struct.unpack('<I', remaining[:4])[0]
  responses.append(json.loads(remaining[4 : 4 + size]))
  remaining = remaining[4 + size :]
assert len(responses) == len(requests)
assert 'sites' in responses[0]['configuration']
assert responses[1]['error'] == 'Unknown native-message type'
assert responses[2]['error'] == 'Invalid site status'
assert responses[3]['configuration'] == responses[0]['configuration']
for invalid in (struct.pack('<I', 65537), struct.pack('<I', 10) + b'{'):
  rejected = subprocess.run(
    [str(app), '--native-host'], input=invalid, capture_output=True, timeout=5
  )
  assert rejected.returncode == 1
report = 'PASS: four framed requests, two snapshots, unknown-command rejection, invalid-status rejection, oversized and truncated-message rejection.\n'
(artifacts / 'bridge.txt').write_text(report)
print(report)
