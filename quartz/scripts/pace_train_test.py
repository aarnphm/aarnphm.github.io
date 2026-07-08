from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_pace_train() -> ModuleType:
  spec = importlib.util.spec_from_file_location(
    'pace_train', Path(__file__).with_name('pace_train.py')
  )
  if spec is None or spec.loader is None:
    raise RuntimeError('could not load pace_train.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


pace_train = load_pace_train()


def activity(iso: str, avg_hr: float) -> dict[str, object]:
  return {
    'date': iso,
    'sport': 'run',
    'distanceKm': 10.0,
    'elevationM': 0.0,
    'vGap': 3.0,
    'avgHr': avg_hr,
  }


@pytest.mark.parametrize('target', ('pace', 'hr'))
def test_build_dataset_includes_feed_today(target: str) -> None:
  meta = {
    'today': '2099-01-02',
    'thresholds': [{'sport': 'run', 'vThr': 3.0}],
    'athlete': {'hrMaxEst': 190.0},
  }
  days = {'2099-01-01': {'date': '2099-01-01'}}
  acts = [
    activity('2099-01-01', 150.0),
    activity('2099-01-02', 151.0),
    activity('2099-01-03', 152.0),
  ]

  _, _, y, dates, _, _, _ = pace_train.build_dataset(meta, days, acts, target)

  assert len(y) == 2
  assert dates == ['2099-01-01', '2099-01-02']
