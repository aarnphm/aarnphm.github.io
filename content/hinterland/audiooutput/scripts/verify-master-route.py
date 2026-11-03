#!/usr/bin/env python3
"""Exercise device-scoped capture with a -60 dBFS stereo tone on a named output."""
import argparse
import math
from pathlib import Path
import struct
import subprocess
import time
import wave

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output_uid')
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
artifacts = project.parents[2] / 'docs/verification/audiooutput-master-volume/artifacts'
artifacts.mkdir(parents=True, exist_ok=True)
app = project / 'dist/Audio Output.app/Contents/MacOS/AudioOutput'
fixture = artifacts / 'tone-fixture'
subprocess.run(['xcrun', 'swiftc', str(project / 'scripts/tone-fixture.swift'), '-o', str(fixture)], check=True)
wave_path = artifacts / 'quiet-tone.wav'
with wave.open(str(wave_path), 'wb') as audio:
    audio.setnchannels(2)
    audio.setsampwidth(2)
    audio.setframerate(48000)
    audio.writeframes(b''.join(struct.pack('<hh', int(32 * math.sin(frame * 2 * math.pi * 440 / 48000)), int(32 * math.sin(frame * 2 * math.pi * 660 / 48000))) for frame in range(48000)))
before = subprocess.check_output([str(app), '--inventory'], text=True).splitlines()[0]
with (artifacts / 'tone.log').open('w') as log:
    source = subprocess.Popen([str(fixture), str(wave_path), args.output_uid], stdout=log, stderr=log)
    try:
        time.sleep(2)
        command = [str(app), '--probe-device', args.output_uid, '0.5', '4']
        result = subprocess.run(command, text=True, capture_output=True, timeout=35)
        after = subprocess.check_output([str(app), '--inventory'], text=True).splitlines()[0]
        report = f'Command: {command}\nGain: 0.5\nSource: stereo PCM16, peak 32/32768\nDefault unchanged: {before == after}\nExit: {result.returncode}\n{result.stdout}{result.stderr}'
        (artifacts / 'device-probe.txt').write_text(report)
        print(report)
        assert before == after, 'System default changed'
        assert result.returncode == 0, 'Master device probe failed'
    finally:
        source.terminate()
        source.wait(timeout=5)
