import argparse
import array
import hashlib
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


TORONTO = ZoneInfo('America/Toronto')
RECORDINGS = (
  Path.home()
  / 'Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings'
)
REPOSITORY = Path(__file__).resolve().parents[2]
MEMO_ATTRIBUTE_RULES = (
  'content/triathlon/memos/**/*.m4a filter=lfs diff=lfs merge=lfs -text',
  'content/triathlon/memos/**/*.peaks.json -filter diff merge text',
)


@dataclass(frozen=True)
class Recording:
  path: Path
  recorded_at: datetime
  duration: float
  aac_index: int
  digest: str


def sha256(path: Path) -> str:
  with path.open('rb') as source:
    return hashlib.file_digest(source, 'sha256').hexdigest()


def active_recording_names(directory: Path) -> set[str] | None:
  database = directory / 'CloudRecordings.db'
  if not database.exists():
    if directory.resolve() == RECORDINGS.resolve():
      raise ValueError(
        'Voice Memos metadata is unavailable; cannot exclude Recently Deleted recordings'
      )
    return None
  try:
    with closing(
      sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True)
    ) as connection:
      # Voice Memos can leave Recently Deleted audio on disk.
      return {
        path
        for (path,) in connection.execute(
          'SELECT ZPATH FROM ZCLOUDRECORDING WHERE ZEVICTIONDATE IS NULL'
        )
      }
  except sqlite3.Error as error:
    raise ValueError(f'Cannot read Voice Memos metadata: {error}') from error


def update_memo_attributes(text: str) -> str:
  lines = []
  for line in text.splitlines(keepends=True):
    fields = line.split()
    if line.strip() in MEMO_ATTRIBUTE_RULES:
      continue
    if (
      fields
      and fields[0].lstrip('/').startswith('content/triathlon/memos/')
      and set(fields[1:]) == {'filter=lfs', 'diff=lfs', 'merge=lfs', '-text'}
    ):
      continue
    lines.append(line)
  preserved = ''.join(lines)
  if preserved and not preserved.endswith('\n'):
    preserved += '\n'
  return preserved + '\n'.join(MEMO_ATTRIBUTE_RULES) + '\n'


def ensure_memo_attributes(repository: Path) -> None:
  path = repository / '.gitattributes'
  current = path.read_text() if path.exists() else ''
  updated = update_memo_attributes(current)
  if updated != current:
    path.write_text(updated)


def probe(path: Path) -> Recording:
  result = subprocess.run(
    [
      'ffprobe',
      '-v',
      'error',
      '-show_format',
      '-show_streams',
      '-of',
      'json',
      str(path),
    ],
    check=True,
    capture_output=True,
    text=True,
    timeout=60,
  )
  data = json.loads(result.stdout)
  audio = next(
    (
      stream for stream in data['streams'] if stream.get('codec_name') == 'aac'
    ),
    None,
  )
  if audio is None:
    raise ValueError(
      f'{path.name} has no AAC track to copy without re-encoding'
    )
  created = data['format'].get('tags', {}).get('creation_time')
  if not created:
    raise ValueError(f'{path.name} has no recording creation time')
  recorded_at = datetime.fromisoformat(created.replace('Z', '+00:00'))
  if recorded_at.tzinfo is None:
    raise ValueError(f'{path.name} has an ambiguous recording timezone')
  duration = float(audio['duration'])
  if not math.isfinite(duration) or duration <= 0:
    raise ValueError(f'{path.name} has an invalid duration')
  return Recording(
    path,
    recorded_at.astimezone(TORONTO),
    duration,
    int(audio['index']),
    sha256(path),
  )


def waveform(path: Path) -> list[float]:
  result = subprocess.run(
    [
      'ffmpeg',
      '-v',
      'error',
      '-i',
      str(path),
      '-map',
      '0:a:0',
      '-ac',
      '1',
      '-ar',
      '8000',
      '-f',
      's16le',
      '-',
    ],
    check=True,
    capture_output=True,
    timeout=240,
  )
  samples = array.array('h', result.stdout)
  if sys.byteorder != 'little':
    samples.byteswap()
  if not samples:
    raise ValueError('The recording has no decoded audio')
  count = 512
  peaks = []
  for index in range(count):
    segment = samples[
      index * len(samples) // count : (index + 1) * len(samples) // count
    ]
    peaks.append(
      math.sqrt(sum(value * value for value in segment) / max(1, len(segment)))
    )
  maximum = max(peaks) or 1
  return [round(value / maximum, 4) for value in peaks]


def memo_name(recording: Recording, destination: Path) -> str:
  for existing in sorted(destination.glob('*.peaks.json')):
    if (
      json.loads(existing.read_text()).get('sourceSha256') == recording.digest
    ):
      return existing.name.removesuffix('.peaks.json')
  if recording.path.parent.resolve() == destination.resolve():
    return recording.path.stem
  return f'{recording.recorded_at:%Y%m%d-%H%M%S}-{recording.digest[:12]}'


def timestamp(value: datetime) -> str:
  offset = value.strftime('%z')
  return f'{value:%Y-%m-%d %H:%M:%S} GMT{offset[:3]}:{offset[3:]}'


def update_stream(
  text: str, day: date, memos: list[str], now: datetime
) -> str:
  embeds = [f'![[triathlon/memos/{name}.m4a]]' for name in memos]
  day_text = day.isoformat()
  sections = list(re.finditer(r'^## .+$', text, re.MULTILINE))
  matches = []
  for index, heading in enumerate(sections):
    end = (
      sections[index + 1].start() if index + 1 < len(sections) else len(text)
    )
    section = text[heading.start() : end]
    if heading.group() == f'## {day_text}' and re.search(
      r'  - description: training log \d+', section
    ):
      matches.append((heading.start(), end, section))
  if len(matches) > 1:
    raise ValueError(
      f'Multiple training entries exist for {day_text}; select the intended entry'
    )
  if matches:
    start, end, section = matches[0]
    for name in memos:
      section = section.replace(
        f'![[triathlon/memos/{name}.qta]]', f'![[triathlon/memos/{name}.m4a]]'
      )
    missing = [embed for embed in embeds if embed not in section]
    if missing:
      separator = re.search(r'^---\s*$', section, re.MULTILINE)
      if not separator:
        raise ValueError(
          f'The training entry for {day_text} has no closing separator'
        )
      section = (
        section[: separator.start()].rstrip()
        + '\n\n'
        + '\n\n'.join(missing)
        + '\n\n'
        + section[separator.start() :]
      )
    return text[:start] + section + text[end:]
  numbers = [
    int(value)
    for value in re.findall(
      r'^  - description: training log (\d+)', text, re.MULTILINE
    )
  ]
  number = max(numbers, default=0) + 1
  entry = (
    f'## {day_text}\n\n- [meta]:\n  - date: {timestamp(now)}\n  - tags:\n'
    f'    - life\n    - o/training\n  - description: training log {number:03d}\n\n'
    f'![[triathlon#{day_text}#analytics]]\n\n'
    + '\n\n'.join(embeds)
    + '\n\n---\n\n'
  )
  frontmatter = re.match(r'\A---\n.*?\n---\n', text, re.DOTALL)
  if not frontmatter:
    raise ValueError('stream.md has no complete frontmatter')
  insertion = frontmatter.end()
  return text[:insertion] + '\n' + entry + text[insertion:].lstrip('\n')


def import_recording(recording: Recording, destination: Path) -> str:
  name = memo_name(recording, destination)
  if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
    raise ValueError(f'Unsupported memo filename: {name}')
  destination.mkdir(parents=True, exist_ok=True)
  audio = destination / f'{name}.m4a'
  peaks = destination / f'{name}.peaks.json'
  if peaks.exists():
    metadata = json.loads(peaks.read_text())
    if metadata.get('sourceSha256') != recording.digest:
      raise ValueError(f'{name} already refers to different source bytes')
    if audio.exists() and metadata.get('audioSha256') == sha256(audio):
      return name
    raise ValueError(f'{name} has an incomplete or changed import')
  if audio.exists() and audio.resolve() != recording.path.resolve():
    raise ValueError(f'{audio} exists without import metadata')
  with tempfile.TemporaryDirectory(prefix='voice-memo-') as temporary:
    converted = Path(temporary) / f'{name}.m4a'
    subprocess.run(
      [
        'ffmpeg',
        '-v',
        'error',
        '-nostdin',
        '-i',
        str(recording.path),
        '-map',
        f'0:{recording.aac_index}',
        '-c:a',
        'copy',
        '-map_metadata',
        '-1',
        '-movflags',
        '+faststart',
        str(converted),
      ],
      check=True,
      timeout=240,
    )
    metadata = {
      'source': recording.path.name,
      'sourceSha256': recording.digest,
      'audioSha256': sha256(converted),
      'recordedAt': recording.recorded_at.isoformat(),
      'duration': recording.duration,
      'peaks': waveform(converted),
    }
    if sha256(recording.path) != recording.digest:
      raise ValueError(
        f'{recording.path.name} changed during import; wait until recording finishes'
      )
    if audio.resolve() == recording.path.resolve():
      raise ValueError(
        'Move an original .m4a outside the destination before importing it'
      )
    shutil.copyfile(converted, audio)
    peaks.write_text(json.dumps(metadata, indent=2) + '\n')
  return name


def main() -> None:
  parser = argparse.ArgumentParser(
    description='Import Apple Voice Memos into triathlon training logs'
  )
  dates = parser.add_mutually_exclusive_group()
  dates.add_argument(
    '--date',
    type=date.fromisoformat,
    metavar='YYYY-MM-DD',
    help='Import one recording date in America/Toronto (default: today)',
  )
  dates.add_argument(
    '--since',
    type=date.fromisoformat,
    metavar='YYYY-MM-DD',
    help='Import all recording dates on or after this date in America/Toronto',
  )
  parser.add_argument('--source', type=Path, default=RECORDINGS)
  parser.add_argument('--file', type=Path, action='append')
  parser.add_argument('--repo', type=Path, default=REPOSITORY)
  parser.add_argument('--list', action='store_true')
  args = parser.parse_args()
  day = args.since or args.date or datetime.now(TORONTO).date()
  if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
    raise ValueError('ffmpeg and ffprobe must be available on PATH')
  if args.file:
    candidates = args.file
  else:
    try:
      prefixes = ()
      if args.since is None:
        nearby_days = [day + timedelta(days=offset) for offset in (-1, 0, 1)]
        prefixes = tuple(
          nearby.strftime(pattern)
          for nearby in nearby_days
          for pattern in ('%Y%m%d', '%Y-%m-%d')
        )
      candidates = sorted(
        path
        for path in args.source.iterdir()
        if path.suffix.lower() in {'.qta', '.m4a'}
        and (args.since is not None or path.name.startswith(prefixes))
        and not (
          path.suffix.lower() == '.m4a'
          and path.with_suffix('.peaks.json').exists()
        )
      )
    except PermissionError as error:
      raise ValueError(
        'macOS denied access to Voice Memos. Grant the running app Full Disk Access, or copy selected recordings to an accessible folder and pass --file.'
      ) from error
  active_by_directory: dict[Path, set[str] | None] = {}
  recordings = []
  for path in candidates:
    directory = path.parent.resolve()
    if directory not in active_by_directory:
      active_by_directory[directory] = active_recording_names(directory)
    active = active_by_directory[directory]
    if active is not None and path.name not in active:
      if args.file:
        raise ValueError(f'{path.name} is not an active Voice Memo')
      continue
    recording = probe(path)
    recorded_day = recording.recorded_at.date()
    selected = recorded_day >= day if args.since else recorded_day == day
    if selected:
      recordings.append(recording)
    elif args.file:
      if args.since:
        raise ValueError(
          f'{path.name} was recorded on {recorded_day}, before {day}'
        )
      raise ValueError(
        f'{path.name} was recorded on {recorded_day}, not {day}'
      )
  recordings.sort(key=lambda item: (item.recorded_at, item.digest))
  if not recordings:
    selection = f'since {day}' if args.since else f'for {day}'
    print(f'No recordings {selection}; nothing changed')
    return
  if args.list:
    for recording in recordings:
      print(
        f'{recording.recorded_at.isoformat()}  {recording.duration:.1f}s  {recording.path.name}'
      )
    return
  destination = args.repo / 'content/triathlon/memos'
  stream = args.repo / 'content/stream.md'
  current = stream.read_text()
  names_by_day: dict[date, list[str]] = {}
  for recording in recordings:
    names = names_by_day.setdefault(recording.recorded_at.date(), [])
    name = memo_name(recording, destination)
    if name not in names:
      names.append(name)
  updated = current
  now = datetime.now(TORONTO)
  for recorded_day, names in names_by_day.items():
    updated = update_stream(updated, recorded_day, names, now)
  for recording in recordings:
    name = import_recording(recording, destination)
    print(f'{name}.m4a  {recording.duration:.1f}s')
  if stream.read_text() != current:
    raise ValueError(
      'stream.md changed during import; rerun to apply embeds to the latest text'
    )
  ensure_memo_attributes(args.repo)
  if updated != current:
    stream.write_text(updated)


if __name__ == '__main__':
  try:
    main()
  except (ValueError, OSError, subprocess.SubprocessError) as error:
    print(f'Voice memo import failed: {error}', file=sys.stderr)
    sys.exit(1)
