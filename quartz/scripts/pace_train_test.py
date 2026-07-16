from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
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


def test_build_dataset_includes_swim_pace_residual() -> None:
  meta = {
    'today': '2099-01-01',
    'thresholds': [{'sport': 'swim', 'vThr': 0.8}],
    'athlete': {'hrMaxEst': 190.0},
  }
  acts = [
    {
      'date': '2099-01-01',
      'sport': 'swim',
      'distanceKm': 0.75,
      'elevationM': 0.0,
      'vGap': 0.7,
      'avgHr': 145.0,
    }
  ]

  raw, _, y, _, sports, backbone, actual = pace_train.build_dataset(
    meta, {}, acts, 'pace'
  )

  expected_backbone = 0.8 * (0.75 / (0.8 * 3600 / 1000)) ** (1 - 1.03)
  assert sports == ['swim']
  assert raw[0, 0:3].tolist() == [1.0, 0.0, 0.0]
  assert backbone[0] == pytest.approx(expected_backbone)
  assert actual[0] == pytest.approx(0.7)
  assert y[0] == pytest.approx(math.log(0.7 / expected_backbone))


def test_temporal_split_keeps_recent_validation_rows_for_each_sport() -> None:
  dates = ['2099-01-01', '2099-01-02', '2099-01-03'] * 3
  sports = ['swim'] * 3 + ['bike'] * 3 + ['run'] * 3

  split = pace_train.temporal_split(dates, sports, 0.34)

  assert split.tolist() == [False, False, True] * 3


def test_sport_balanced_weights_give_each_sport_equal_mass() -> None:
  sports = np.array(['swim', 'bike', 'bike', 'run', 'run', 'run'])

  weights = pace_train.sport_balanced_weights(sports)

  totals = [
    float(weights[sports == sport].sum()) for sport in pace_train.SPORTS
  ]
  assert totals == pytest.approx([2.0, 2.0, 2.0])
  assert float(weights.mean()) == pytest.approx(1.0)


def test_stratified_bootstrap_preserves_each_sport_count() -> None:
  sports = np.array(['swim', 'bike', 'bike', 'run', 'run', 'run'])

  boot = pace_train.stratified_bootstrap(sports, np.random.default_rng(7))
  sampled = sports[boot]

  assert [int(np.sum(sampled == sport)) for sport in pace_train.SPORTS] == [
    1,
    2,
    3,
  ]


def test_swim_validation_metric_is_seconds_per_100_metres() -> None:
  metrics = pace_train.validation_by_sport(
    'pace',
    np.array(['swim', 'swim', 'swim']),
    np.array(['swim', 'swim']),
    np.array([0.5, 0.4]),
    np.array([0.5, 0.4]),
    np.array([0.4, 0.4]),
  )

  assert metrics['swim'] == {
    'nTrain': 3,
    'nVal': 2,
    'mae': 0.0,
    'baselineMae': 25.0,
    'beatsBaseline': True,
    'unit': 's/100m',
  }


def test_pace_promotion_requires_swim_to_beat_its_baseline() -> None:
  by_sport = {
    'swim': {'beatsBaseline': False},
    'bike': {'beatsBaseline': True},
    'run': {'beatsBaseline': True},
  }

  assert not pace_train.passes_baseline_gate('pace', 0.1, 0.2, by_sport)
  assert pace_train.passes_baseline_gate('hr', 0.1, 0.2, by_sport)
