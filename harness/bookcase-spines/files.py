"""Atomic run state, freshness pins and explicit coordinator ownership."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from schema import Job, Pin, State, decode

HARNESS = Path(__file__).resolve().parent
REPOSITORY = HARNESS.parents[1]


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def data(value: object) -> object:
  return (
    asdict(value)
    if is_dataclass(value) and not isinstance(value, type)
    else value
  )


def dumps(value: object) -> str:
  return json.dumps(
    data(value), sort_keys=True, ensure_ascii=False, allow_nan=False
  )


def sha(value: bytes | str) -> str:
  return hashlib.sha256(
    value.encode() if isinstance(value, str) else value
  ).hexdigest()


def read_json(file: Path) -> object:
  return json.loads(file.read_text())


def pin(file: Path) -> Pin:
  return Pin(str(file.resolve(strict=True)), sha(file.read_bytes()))


def check_pin(value: Pin) -> None:
  if sha(Path(value.file).read_bytes()) != value.sha256:
    raise ValueError(f'Changed input: {value.file}')


def atomic(file: Path, value: bytes | str) -> None:
  file.parent.mkdir(parents=True, exist_ok=True)
  temp = file.with_name(f'{file.name}.{uuid4()}.tmp')
  with temp.open('xb') as stream:
    stream.write(value.encode() if isinstance(value, str) else value)
    stream.flush()
    os.fsync(stream.fileno())
  temp.replace(file)


def write_json(file: Path, value: object) -> None:
  atomic(
    file,
    json.dumps(data(value), indent=2, ensure_ascii=False, allow_nan=False)
    + '\n',
  )


def run_path(value: str) -> Path:
  path = Path(value)
  return (
    path.resolve()
    if path.is_absolute() or '/' in value
    else REPOSITORY / '.quartz-cache/bookcase-spines' / value
  )


def implementation_hash() -> str:
  paths = sorted([
    *HARNESS.glob('*.py'),
    *HARNESS.glob('prompts/*.md'),
    HARNESS / 'scripts/run.sh',
  ])
  return sha(
    '\n'.join(
      f'{file.relative_to(HARNESS)}:{sha(file.read_bytes())}'
      for file in paths
      if file.name != 'e2e.py'
    )
  )


def fresh(state: State) -> None:
  check_pin(state.configPin)
  if state.implementation != implementation_hash():
    raise ValueError('Harness implementation changed; create a fresh plan')
  for value in state.sourcePins:
    check_pin(value)


def fresh_job(job: Job) -> None:
  for value in [job.input.reference, job.input.front]:
    if value:
      check_pin(value)


def read_state(directory: Path) -> State:
  state = decode(State, read_json(directory / 'state.json'))
  state.config.validate()
  if Path(state.run).resolve() != directory.resolve():
    raise ValueError('Run differs from its recorded root')
  return state


def save_state(state: State, event: str, detail: object = None) -> None:
  write_json(Path(state.run) / 'state.json', state)
  with (Path(state.run) / 'events.jsonl').open('a') as stream:
    stream.write(dumps({'at': now(), 'event': event, 'detail': detail}) + '\n')


@contextmanager
def guard(directory: Path) -> Iterator[None]:
  with (directory / 'coordinator.guard').open('a') as handle:
    try:
      fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise ValueError('Another coordinator owns the run') from error
    try:
      yield
    finally:
      fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def locked(directory: Path) -> Iterator[None]:
  with guard(directory):
    marker = directory / 'coordinator.lock'
    with marker.open('x') as stream:
      stream.write(dumps({'pid': os.getpid(), 'at': now()}))
    try:
      yield
    finally:
      marker.unlink()


def recover(directory: Path) -> None:
  with guard(directory):
    marker = directory / 'coordinator.lock'
    owner = read_json(marker)
    if not isinstance(owner, dict) or type(owner.get('pid')) is not int:
      raise ValueError('Malformed ownership marker; inspect manually')
    if any(directory.glob('invocations/*/active.json')):
      raise ValueError(
        'Unfinished provider markers need process inspection before recovery'
      )
    try:
      os.kill(owner['pid'], 0)
    except ProcessLookupError:
      marker.unlink()
      save_state(read_state(directory), 'recover', owner)
      return
    raise ValueError('Coordinator process is alive; recovery refused')
