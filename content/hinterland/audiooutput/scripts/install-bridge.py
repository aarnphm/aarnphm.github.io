#!/usr/bin/env python3
import base64
import hashlib
import json
from pathlib import Path
import shlex

project = Path(__file__).resolve().parents[1]
manifest = json.loads((project / 'Extension/manifest.json').read_text())
key = base64.b64decode(manifest['key'])
extension_id = ''.join(
  chr(ord('a') + int(char, 16))
  for char in hashlib.sha256(key).hexdigest()[:32]
)
executable = project / 'dist/Audio Output.app/Contents/MacOS/AudioOutput'
if not executable.is_file():
  raise SystemExit('Run just build first.')
bridge = project / 'dist/native-host'
bridge.write_text(
  f'#!/bin/sh\nexec {shlex.quote(str(executable))} --native-host "$@"\n'
)
bridge.chmod(0o755)
host = {
  'name': 'xyz.aarnphm.audiooutput',
  'description': 'Audio Output site routing',
  'path': str(bridge),
  'type': 'stdio',
  'allowed_origins': [f'chrome-extension://{extension_id}/'],
}
support = Path.home() / 'Library/Application Support'
for browser in ('net.imput.helium', 'Google/Chrome', 'Chromium'):
  root = support / browser
  if not root.is_dir():
    continue
  hosts = root / 'NativeMessagingHosts'
  hosts.mkdir(exist_ok=True)
  target = hosts / 'xyz.aarnphm.audiooutput.json'
  target.write_text(json.dumps(host, indent=2) + '\n')
  print(f'Registered {target}')
print(f'Extension ID: {extension_id}')
print(f'Load unpacked: {project / "Extension"}')
