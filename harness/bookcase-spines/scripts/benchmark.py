#!/usr/bin/env python3
"""Measure local validation, codecs, and coordinator startup without model calls."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from files import HARNESS, REPOSITORY, pin, read_json, write_json
from schema import Config, decode
from svg import inspect_svg, rasterize


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--samples', type=int, default=5)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  if args.samples < 1 or args.samples > 100:
    parser.error('samples must be between 1 and 100')
  directory = (
    args.output
    or REPOSITORY
    / 'docs/verification/bookcase-spine-harness/artifacts/benchmark'
    / str(time.time_ns())
  ).resolve()
  directory.mkdir(parents=True)
  svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 1280"><rect width="80" height="1280" fill="#eee9dc"/>'
    + ''.join(
      f'<path d="M8 {8 + i * 12}h64v5H8Z" fill="#242424"/>' for i in range(100)
    )
    + '</svg>'
  )
  source = directory / 'reference.svg'
  source.write_text(svg)
  reference = directory / 'reference.png'
  subprocess.run(
    ['rsvg-convert', '--output', str(reference), str(source)], check=True
  )
  config = decode(Config, read_json(HARNESS / 'config.json'))
  timings: dict[str, list[float]] = {
    'validationMs': [],
    'rasterPairMs': [],
    'cliDoctorMs': [],
  }
  for index in range(args.samples):
    start = time.perf_counter()
    for _ in range(100):
      inspect_svg(svg, config.limits)
    timings['validationMs'].append((time.perf_counter() - start) * 10)
    start = time.perf_counter()
    rasterize(
      svg,
      reference,
      directory / f'sample-{index}',
      config,
      ['Synthetic benchmark with 100 geometric marks.'],
    )
    timings['rasterPairMs'].append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    subprocess.run(
      ['bash', str(HARNESS / 'scripts/run.sh'), 'doctor'],
      check=True,
      capture_output=True,
    )
    timings['cliDoctorMs'].append((time.perf_counter() - start) * 1000)
  summary = {
    name: {
      'samples': values,
      'median': statistics.median(values),
      'min': min(values),
      'max': max(values),
    }
    for name, values in timings.items()
  }
  result = {
    'command': sys.argv,
    'python': sys.version,
    'source': pin(source).__dict__,
    'reference': pin(reference).__dict__,
    'dimensions': [80, 1280],
    'pathCount': 100,
    'summaryMs': summary,
    'limitation': 'Local warm measurements on a geometric fixture, without model latency or a runtime FPS claim.',
  }
  write_json(directory / 'benchmark.json', result)
  print(directory / 'benchmark.json')


if __name__ == '__main__':
  main()
