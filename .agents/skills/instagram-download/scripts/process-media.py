#!/usr/bin/env python3
"""Process browser-observed Instagram media into verified local downloads."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import zipfile


@dataclass(frozen=True)
class Item:
  position: int
  kind: Literal['image', 'video', 'audio']
  source: str
  audio_source: str | None = None
  expected_duration: float | None = None
  start_seconds: float = 0
  duration_seconds: float | None = None


def record(value: object) -> dict[str, object]:
  if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
    raise ValueError('Expected a JSON object')
  return value


def source_field(data: dict[str, object], prefix: str = '') -> str | None:
  values = [data.get(prefix + name) for name in ('path', 'url')]
  sources = [value for value in values if isinstance(value, str) and value]
  if len(sources) > 1:
    raise ValueError('Supply a path or a URL, not both')
  return sources[0] if sources else None


def number(value: object, name: str, optional: bool = True) -> float | None:
  if value is None and optional:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f'{name} must be a number')
  result = float(value)
  if not math.isfinite(result) or result < 0:
    raise ValueError(f'{name} must be finite and nonnegative')
  return result


def parse_item(value: object) -> Item:
  data = record(value)
  position = data.get('position')
  if (
    isinstance(position, bool) or not isinstance(position, int) or position < 1
  ):
    raise ValueError('Positions must be positive integers')
  kind = data.get('kind')
  if kind not in ('image', 'video', 'audio'):
    raise ValueError('Kind must be image, video, or audio')
  source = source_field(data)
  if source is None:
    raise ValueError('Each item needs its observed URL or local path')
  start = number(data.get('start_seconds', 0), 'start_seconds', False)
  duration = number(data.get('duration_seconds'), 'duration_seconds')
  if duration == 0:
    raise ValueError('duration_seconds must be positive')
  return Item(
    position,
    kind,
    source,
    source_field(data, 'audio_'),
    number(data.get('expected_duration'), 'expected_duration'),
    start if start is not None else 0,
    duration,
  )


def run(command: list[str], log: list[list[str]], timeout: int = 240) -> str:
  log.append(command)
  result = subprocess.run(
    command, check=True, capture_output=True, text=True, timeout=timeout
  )
  return result.stdout


def probe(
  path: Path, log: list[list[str]], frames: bool = False
) -> dict[str, object]:
  command = ['ffprobe', '-v', 'error']
  if frames:
    command.append('-count_frames')
  command += ['-show_streams', '-show_format', '-of', 'json', str(path)]
  return record(json.loads(run(command, log)))


def stream_list(info: dict[str, object]) -> list[dict[str, object]]:
  streams = info.get('streams')
  if not isinstance(streams, list):
    raise ValueError('ffprobe returned no streams')
  return [record(s) for s in streams]


def media_duration(info: dict[str, object]) -> float:
  return float(str(record(info['format'])['duration']))


def acquire(source: str, target: Path, log: list[list[str]]) -> Path:
  url = urlsplit(source)
  if not url.scheme:
    path = Path(source).expanduser().resolve(strict=True)
    if not path.is_file():
      raise ValueError('Local source must be a file')
    return path
  host = url.hostname or ''
  domains = ('cdninstagram.com', 'fbcdn.net', 'fbsbx.com')
  if (
    url.scheme != 'https'
    or url.username
    or url.password
    or url.port not in (None, 443)
    or not any(
      host == domain or host.endswith('.' + domain) for domain in domains
    )
  ):
    raise ValueError(
      'Source URL must be an observed HTTPS Meta media CDN file'
    )
  query = [
    (k, v)
    for k, v in parse_qsl(url.query, keep_blank_values=True)
    if k not in {'bytestart', 'byteend'}
  ]
  complete = urlunsplit((
    url.scheme,
    url.netloc,
    url.path,
    urlencode(query),
    '',
  ))
  run(
    [
      'curl',
      '--fail',
      '--location',
      '--silent',
      '--show-error',
      '--max-time',
      '60',
      '--proto',
      '=https',
      '--proto-redir',
      '=https',
      '--output',
      str(target),
      complete,
    ],
    log,
    65,
  )
  if target.stat().st_size == 0:
    raise ValueError('Downloaded file is empty')
  return target


def image_extension(path: Path) -> str:
  with path.open('rb') as handle:
    head = handle.read(32)
  if head.startswith(b'\xff\xd8\xff'):
    return '.jpg'
  if head.startswith(b'\x89PNG\r\n\x1a\n'):
    return '.png'
  if head.startswith((b'GIF87a', b'GIF89a')):
    return '.gif'
  if head.startswith(b'RIFF') and head[8:12] == b'WEBP':
    return '.webp'
  if head[4:8] == b'ftyp':
    return '.avif' if b'avif' in head else '.heic'
  raise ValueError('Unrecognized image file signature')


def output_record(
  path: Path, log: list[list[str]], expected: float | None = None
) -> dict[str, object]:
  info = probe(path, log, frames=True)
  if expected is not None and abs(media_duration(info) - expected) > 0.15:
    raise ValueError(f'Duration mismatch in {path.name}')
  run(
    [
      'ffmpeg',
      '-hide_banner',
      '-loglevel',
      'error',
      '-xerror',
      '-i',
      str(path),
      '-f',
      'null',
      '-',
    ],
    log,
  )
  with path.open('rb') as handle:
    checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
  return {
    'file': path.name,
    'bytes': path.stat().st_size,
    'sha256': checksum,
    'probe': info,
    'full_decode': 'passed',
  }


def export_audio(
  source: Path, stem: str, destination: Path, item: Item, log: list[list[str]]
) -> list[dict[str, object]]:
  info = probe(source, log)
  audio = next(
    (s for s in stream_list(info) if s.get('codec_type') == 'audio'), None
  )
  if audio is None:
    raise ValueError('The supplied audio source has no audio stream')
  expected = item.duration_seconds or (
    media_duration(info) - item.start_seconds
  )
  if (
    item.expected_duration is not None
    and abs(expected - item.expected_duration) > 0.15
  ):
    raise ValueError('Observed audio duration differs from downloaded audio')
  if (
    expected <= 0
    or item.start_seconds + expected > media_duration(info) + 0.15
  ):
    raise ValueError('Audio excerpt is outside the source duration')
  trimming = item.start_seconds != 0 or item.duration_seconds is not None
  codec = (
    'copy' if audio.get('codec_name') == 'aac' and not trimming else 'aac'
  )
  m4a, mp3 = destination / (stem + '.m4a'), destination / (stem + '.mp3')
  command = [
    'ffmpeg',
    '-hide_banner',
    '-loglevel',
    'error',
    '-n',
    '-i',
    str(source),
  ]
  for path, encoder in [(m4a, codec), (mp3, 'libmp3lame')]:
    command += ['-map', '0:a:0', '-vn']
    if trimming:
      command += ['-ss', str(item.start_seconds), '-t', str(expected)]
    command += ['-c:a', encoder]
    if encoder != 'copy':
      command += ['-b:a', '128k']
    if path == m4a:
      command += ['-movflags', '+faststart']
    command.append(str(path))
  run(command, log)
  return [output_record(path, log, expected) for path in (m4a, mp3)]


def process(
  item: Item, destination: Path, work: Path, log: list[list[str]]
) -> list[dict[str, object]]:
  source = acquire(
    item.source, work / f'{item.position:02d}-{item.kind}.bin', log
  )
  stem = f'{item.position:02d}'
  if item.kind == 'image':
    target = destination / (stem + image_extension(source))
    shutil.copyfile(source, target)
    return [output_record(target, log)]
  if item.kind == 'audio':
    return export_audio(source, stem + '-audio', destination, item, log)
  info = probe(source, log)
  video = next(
    (s for s in stream_list(info) if s.get('codec_type') == 'video'), None
  )
  if video is None:
    raise ValueError('Video source has no video stream')
  expected = item.expected_duration or media_duration(info)
  if abs(media_duration(info) - expected) > 0.15:
    raise ValueError('Observed player duration differs from downloaded video')
  width, height = int(str(video['width'])), int(str(video['height']))
  factor = min(1, 1920 / max(width, height), 1080 / min(width, height))
  scaled_width = max(2, int(width * factor) // 2 * 2)
  scaled_height = max(2, int(height * factor) // 2 * 2)
  audio = (
    acquire(item.audio_source, work / (stem + '-audio.bin'), log)
    if item.audio_source
    else source
  )
  audio_stream = next(
    (
      s
      for s in stream_list(probe(audio, log))
      if s.get('codec_type') == 'audio'
    ),
    None,
  )
  if item.audio_source and audio_stream is None:
    raise ValueError('The separate audio source has no audio stream')
  target = destination / (stem + '.mp4')
  command = [
    'ffmpeg',
    '-hide_banner',
    '-loglevel',
    'error',
    '-n',
    '-i',
    str(source),
  ]
  if item.audio_source:
    command += ['-i', str(audio)]
  command += ['-map', '0:v:0']
  if audio_stream:
    command += [
      '-map',
      '1:a:0' if item.audio_source else '0:a:0',
      '-c:a',
      'copy' if audio_stream.get('codec_name') == 'aac' else 'aac',
    ]
    if abs(media_duration(probe(audio, log)) - expected) > 0.15:
      raise ValueError('Separate audio and video durations differ')
  command += [
    '-vf',
    f'scale={scaled_width}:{scaled_height}:flags=lanczos',
    '-c:v',
    'libx264',
    '-threads',
    '4',
    '-preset',
    'medium',
    '-crf',
    '23',
    '-maxrate',
    '5M',
    '-bufsize',
    '10M',
    '-pix_fmt',
    'yuv420p',
    '-movflags',
    '+faststart',
    str(target),
  ]
  run(command, log)
  result = output_record(target, log, expected)
  output_streams = stream_list(record(result['probe']))
  if sum(s.get('codec_type') == 'audio' for s in output_streams) != int(
    audio_stream is not None
  ):
    raise ValueError('Exported video lost or gained an audio stream')
  outputs = [result]
  if audio_stream:
    outputs += export_audio(audio, stem + '-audio', destination, item, log)
  return outputs


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()
  data = record(json.loads(args.manifest.read_text()))
  raw_items = data.get('items')
  if not isinstance(raw_items, list) or not raw_items:
    raise ValueError('Capture manifest must contain media items')
  items = sorted(
    (parse_item(item) for item in raw_items), key=lambda item: item.position
  )
  if len({item.position for item in items}) != len(items):
    raise ValueError('Duplicate media positions')
  if data.get('audio_status') not in {'present', 'none', 'not_exposed'}:
    raise ValueError('Specify audio_status as present, none, or not_exposed')
  for tool in ('curl', 'ffmpeg', 'ffprobe'):
    if shutil.which(tool) is None:
      raise ValueError(f'Required existing tool is missing: {tool}')
  destination = args.output_dir.expanduser().resolve()
  destination.mkdir(parents=True, exist_ok=False)
  work = Path(tempfile.mkdtemp(prefix='instagram-download-'))
  print(f'Source working directory: {work}', flush=True)
  report: dict[str, object] = {
    'source_url': data.get('source_url'),
    'account': data.get('account'),
    'audio_status': data.get('audio_status'),
    'status': 'running',
    'invocation': [sys.executable, *sys.argv],
    'source_directory': str(work),
    'commands': [],
    'outputs': [],
  }
  commands: list[list[str]] = []
  outputs: list[dict[str, object]] = []
  try:
    for item in items:
      produced = process(item, destination, work, commands)
      outputs.extend(produced)
      print(
        json.dumps({
          'position': item.position,
          'files': [p['file'] for p in produced],
        }),
        flush=True,
      )
    if len(items) > 1:
      archive = destination / 'media.zip'
      with zipfile.ZipFile(
        archive, 'x', compression=zipfile.ZIP_DEFLATED
      ) as bundle:
        for path in sorted(destination.iterdir()):
          if path != archive:
            bundle.write(path, path.name)
      with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
          raise ValueError('ZIP CRC verification failed')
      report['archive'] = {
        'file': archive.name,
        'bytes': archive.stat().st_size,
        'crc_check': 'passed',
      }
    report['status'] = 'passed'
  except (ValueError, OSError, subprocess.SubprocessError) as error:
    report['status'] = 'failed'
    report['error'] = str(error)
    raise
  finally:
    report['commands'], report['outputs'] = commands, outputs
    (destination / 'verification.json').write_text(
      json.dumps(report, indent=2) + '\n'
    )
  print(
    json.dumps(
      {
        'output_directory': str(destination),
        'status': 'passed',
        'files': [p['file'] for p in outputs],
      },
      indent=2,
    )
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
