"""Bound only owned subprocess groups; retain capped raw output and exit evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from typing import Iterator, Mapping

DEADLINE: ContextVar[float | None] = ContextVar(
  'invocation_deadline', default=None
)


@contextmanager
def budget(seconds: float) -> Iterator[None]:
  token = DEADLINE.set(time.monotonic() + seconds)
  try:
    yield
  finally:
    DEADLINE.reset(token)


@dataclass
class Result:
  exitCode: int | None
  stdoutFile: str
  stderrFile: str
  error: str | None
  durationSeconds: float


def execute(
  argv: list[str],
  directory: Path,
  *,
  timeout: float,
  max_bytes: int,
  stdin: str = '',
  active: Path | None = None,
  env: Mapping[str, str] | None = None,
) -> Result:
  directory.mkdir(parents=True, exist_ok=True)
  stdout_file, stderr_file = (
    directory / 'stdout.ndjson',
    directory / 'stderr.txt',
  )
  started = time.monotonic()
  deadline = DEADLINE.get()
  if deadline is not None:
    timeout = min(timeout, deadline - started)
  if timeout <= 0:
    raise ValueError('Invocation budget reached')
  error: str | None = None
  code: int | None = None
  with (
    (directory / 'stdin.txt').open('w+') as source,
    stdout_file.open('wb') as stdout,
    stderr_file.open('wb') as stderr,
  ):
    source.write(stdin)
    source.seek(0)
    process = subprocess.Popen(
      argv,
      cwd=directory,
      stdin=source,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      start_new_session=True,
      env=env,
    )
    if active:
      active.write_text(json.dumps({'pid': process.pid, 'argv': argv}))
    handlers = {
      sig: signal.getsignal(sig) for sig in [signal.SIGTERM, signal.SIGHUP]
    }

    def interrupted(signum: int, frame: object) -> None:
      raise KeyboardInterrupt(f'Signal {signum}')

    for sig in handlers:
      signal.signal(sig, interrupted)
    try:
      with selectors.DefaultSelector() as selector:
        if process.stdout is None or process.stderr is None:
          raise RuntimeError('Missing process pipes')
        selector.register(process.stdout, selectors.EVENT_READ, stdout)
        selector.register(process.stderr, selectors.EVENT_READ, stderr)
        total = 0
        while selector.get_map():
          if time.monotonic() - started >= timeout:
            error = 'Process timed out'
            break
          for key, _ in selector.select(min(0.1, timeout)):
            data = os.read(key.fd, 65536)
            if not data:
              selector.unregister(key.fileobj)
              continue
            remaining = max(0, max_bytes - total)
            key.data.write(data[:remaining])
            total += len(data)
            if total > max_bytes:
              error = 'Process output limit exceeded'
              break
          if error:
            break
        if error is None:
          try:
            code = process.wait(
              timeout=max(0.01, timeout - (time.monotonic() - started))
            )
          except subprocess.TimeoutExpired:
            error = 'Process timed out'
    except KeyboardInterrupt:
      error = 'Process interrupted'
      raise
    finally:
      # A successful parent can still leave descendants holding files or pipes.
      try:
        os.killpg(process.pid, signal.SIGTERM)
      except ProcessLookupError:
        pass
      try:
        process.wait(timeout=1)
      except subprocess.TimeoutExpired:
        pass
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      code = process.wait()
      for stream in [process.stdout, process.stderr]:
        if stream:
          stream.close()
      for sig, handler in handlers.items():
        signal.signal(sig, handler)
      if active:
        active.unlink(missing_ok=True)
      result = Result(
        code,
        str(stdout_file),
        str(stderr_file),
        error,
        time.monotonic() - started,
      )
      (directory / 'process.json').write_text(
        json.dumps({'argv': argv, **asdict(result)}, indent=2) + '\n'
      )
  return result
