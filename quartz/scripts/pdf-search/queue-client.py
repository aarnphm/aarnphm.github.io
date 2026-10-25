"""Enqueue public PDFs and wake one detached worker using the existing model clients."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import portalocker

from pending import PendingPdfs, source_hash
from run import stop

REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent


def enqueue(root: Path, slugs: list[str]) -> dict[str, object]:
  queue = PendingPdfs(root)
  report = queue.enqueue(slugs)
  if report['pending']:
    probe = portalocker.Lock(queue.cache / '.queue-worker.lock', timeout=0)
    try:
      probe.acquire()
    except portalocker.exceptions.LockException:
      return {**report, 'workerRunning': True}
    probe.release()
    with (queue.cache / 'queue-worker.log').open('ab') as log:
      process = subprocess.Popen(
        [
          sys.executable,
          str(Path(__file__).resolve()),
          'work',
          '--root',
          str(root),
        ],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
    report['dispatcherPid'] = process.pid
  return report


def work(root: Path) -> None:
  queue = PendingPdfs(root)
  worker_lock = portalocker.Lock(queue.cache / '.queue-worker.lock', timeout=0)
  try:
    worker_lock.acquire()
  except portalocker.exceptions.LockException:
    return
  child: subprocess.Popen[bytes] | None = None

  def interrupted(signum: int, _frame: object) -> None:
    raise InterruptedError(f'Queue worker received signal {signum}')

  signal.signal(signal.SIGTERM, interrupted)
  signal.signal(signal.SIGINT, interrupted)
  (queue.cache / 'queue-worker.pid').write_text(str(os.getpid()))
  claimed = []
  try:
    queue.recover()
    while claimed := queue.claim(worker_lock):
      inventory = queue.inventory()
      valid = []
      for job in claimed:
        source = inventory.get(job.slug)
        if source is None or source_hash(source.source) != job.sourceSha256:
          queue.finish(
            [job],
            'Source changed or left the public inventory; enqueue it again',
          )
        else:
          valid.append(job)
      if not valid:
        continue
      command = [
        sys.executable,
        str(SCRIPTS / 'run.py'),
        '--root',
        str(root),
        '--wait-for-lock',
      ]
      for job in valid:
        command.extend(['--slug', job.slug])
      child = subprocess.Popen(
        command, cwd=root, stdin=subprocess.DEVNULL, start_new_session=True
      )
      code = child.wait()
      child = None
      queue.finish(
        valid,
        f'PDF runner exited {code}; inspect background-papers/status.json'
        if code
        else None,
      )
  except BaseException as error:
    if child is not None:
      stop(child)
    queue.finish(claimed, str(error))
    raise
  finally:
    worker_lock.release()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('enqueue', 'work'))
  parser.add_argument('--root', type=Path, default=REPOSITORY)
  parser.add_argument('--slug', action='append', default=[])
  args = parser.parse_args()
  root = args.root.resolve()
  os.chdir(root)
  if args.action == 'enqueue':
    print(json.dumps(enqueue(root, args.slug)), flush=True)
  else:
    work(root)


if __name__ == '__main__':
  try:
    main()
  except (ValueError, FileNotFoundError) as error:
    print(json.dumps({'error': str(error)}), flush=True)
    sys.exit(1)
