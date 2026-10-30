#!/usr/bin/env python3
"""Exercise the real process tap and render callback with two BlackHole devices."""

import argparse
import math
from pathlib import Path
import struct
import subprocess
import wave
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-uid', default='BlackHole64ch_UID')
parser.add_argument('--report', default='probe.txt')
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
root = project.parents[2]
artifacts = root / 'docs/verification/audiooutput/artifacts'
artifacts.mkdir(parents=True, exist_ok=True)
app = project / 'dist/Audio Output.app/Contents/MacOS/AudioOutput'
fixture = artifacts / 'tone-fixture'
subprocess.run(
  [
    'xcrun',
    'swiftc',
    str(project / 'scripts/tone-fixture.swift'),
    '-o',
    str(fixture),
  ],
  check=True,
)
inventory = subprocess.check_output([str(app), '--inventory'], text=True)
(artifacts / 'inventory.txt').write_text(inventory)
for uid in ('BlackHole16ch_UID', args.output_uid):
  if uid not in inventory:
    raise SystemExit(f'Connect {uid} to run this virtual-device check.')
wave_path = artifacts / 'stereo-tone.wav'
with wave.open(str(wave_path), 'wb') as audio:
  audio.setnchannels(2)
  audio.setsampwidth(2)
  audio.setframerate(48000)
  audio.writeframes(
    b''.join(
      struct.pack(
        '<hh',
        int(327 * math.sin(frame * 2 * math.pi * 440 / 48000)),
        int(327 * math.sin(frame * 2 * math.pi * 660 / 48000)),
      )
      for frame in range(48000)
    )
  )
with (artifacts / 'fixture.log').open('w') as log:
  source = subprocess.Popen(
    [str(fixture), str(wave_path), 'BlackHole16ch_UID'], stdout=log, stderr=log
  )
  try:
    time.sleep(2)
    command = [str(app), '--probe', str(source.pid), args.output_uid, '4']
    try:
      result = subprocess.run(
        command, capture_output=True, text=True, timeout=35
      )
    except subprocess.TimeoutExpired as error:
      details = (error.stderr or b'').decode()
      (artifacts / args.report).write_text(
        f'Command: {command}\nTimed out after 35 seconds\n{details}'
      )
      raise SystemExit(f'Route check timed out. Last stages:\n{details}')
    after = subprocess.check_output([str(app), '--inventory'], text=True)
    default_before = next(
      line
      for line in inventory.splitlines()
      if line.startswith('DEFAULT_OUTPUT')
    )
    default_after = next(
      line for line in after.splitlines() if line.startswith('DEFAULT_OUTPUT')
    )
    if default_before != default_after:
      raise SystemExit('Default output changed during the probe.')
    report = f'Default unchanged: {default_after}\nCommand: {command}\nExit: {result.returncode}\n{result.stdout}{result.stderr}'
    (artifacts / args.report).write_text(report)
    print(report)
    if result.returncode:
      raise SystemExit(result.returncode)
  finally:
    source.terminate()
    source.wait(timeout=5)
