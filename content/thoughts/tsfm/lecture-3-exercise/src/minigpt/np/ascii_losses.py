from __future__ import annotations

import argparse, json, math, pathlib
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Series:
  name: str
  xs: list[float]
  ys: list[float]
  ch: str


def _read_json(p: pathlib.Path) -> dict:
  with open(p, 'r') as f:
    return json.load(f)


def _find_latest_meta(ckpt_dir: pathlib.Path) -> tuple[pathlib.Path, dict, dict | None]:
  candidates = list(ckpt_dir.rglob('steps_*/meta.json'))
  if not candidates:
    raise FileNotFoundError(f'no meta.json found under {ckpt_dir}')

  best_meta: dict | None = None
  best_path: pathlib.Path | None = None
  best_step = -1
  for p in candidates:
    try:
      meta = _read_json(p)
      step = int(meta.get('step', -1))
      if step > best_step:
        best_step = step
        best_meta = meta
        best_path = p
    except Exception:
      continue

  assert best_meta is not None and best_path is not None
  # config is sibling file
  cfg_path = best_path.with_name('config.json')
  cfg = _read_json(cfg_path) if cfg_path.is_file() else None
  return best_path, best_meta, cfg


def _build_series(meta: dict, cfg: dict | None) -> tuple[Series, Series | None, dict]:
  name = str(meta.get('name', 'run'))
  step = int(meta.get('step', 0))

  # Train losses: assume logged every `log_every` steps if available; otherwise use 1..N
  train_losses = [float(x) for x in meta.get('train_losses', [])]
  if not train_losses:
    raise ValueError('meta.json missing train_losses')

  log_every = int((cfg or {}).get('log_every', 1))
  # Prefer explicit step points if present; otherwise assume 1..N (per-step logging)
  train_points = meta.get('train_points')
  if isinstance(train_points, list) and train_points and all(isinstance(p, (int, float)) for p in train_points):
    xs_train = [float(p) for p in train_points]
  else:
    xs_train = [float(i + 1) for i in range(len(train_losses))]
  train = Series('train', xs_train, train_losses, '.')

  # Validation points: prefer explicit points [[step, loss], ...]
  val_series: Series | None = None
  val_points = meta.get('val_points')
  if isinstance(val_points, list) and val_points:
    xs_val = [float(p[0]) for p in val_points]
    ys_val = [float(p[1]) for p in val_points]
    val_series = Series('val', xs_val, ys_val, 'o')
  else:
    # fallback: maybe only losses; infer steps via eval_every if present
    val_losses = meta.get('val_losses') or meta.get('validation_losses')
    if isinstance(val_losses, list) and val_losses:
      eval_every = int((cfg or {}).get('eval_every', 1))
      xs_val = [float((i + 1) * max(1, eval_every)) for i in range(len(val_losses))]
      ys_val = [float(x) for x in val_losses]
      val_series = Series('val', xs_val, ys_val, 'o')

  info = {
    'name': name,
    'step': step,
    'log_every': log_every,
    'eval_every': int((cfg or {}).get('eval_every', 0) or 0),
  }
  return train, val_series, info


def _build_accuracy_series(meta: dict, cfg: dict | None) -> tuple[Series | None, Series | None]:
  # train accuracies: list[float]
  train_accs = meta.get('train_accuracies')
  train_series: Series | None = None
  if isinstance(train_accs, list) and train_accs:
    train_points = meta.get('train_points')
    if isinstance(train_points, list) and train_points and all(isinstance(p, (int, float)) for p in train_points):
      xs_train = [float(p) for p in train_points]
    else:
      xs_train = [float(i + 1) for i in range(len(train_accs))]
    ys_train = [float(a) for a in train_accs]
    train_series = Series('acc-train', xs_train, ys_train, '.')

  # validation accuracy points: [[step, acc], ...]
  val_acc_points = meta.get('val_acc_points')
  val_series: Series | None = None
  if isinstance(val_acc_points, list) and val_acc_points:
    try:
      xs_val = [float(p[0]) for p in val_acc_points]
      ys_val = [float(p[1]) for p in val_acc_points]
      val_series = Series('acc-val', xs_val, ys_val, 'o')
    except Exception:
      val_series = None
  return train_series, val_series


def _minmax(vals: Iterable[float]) -> tuple[float, float]:
  it = iter(vals)
  try:
    v0 = float(next(it))
  except StopIteration:
    return 0.0, 1.0
  mn = mx = v0
  for v in it:
    v = float(v)
    if v < mn:
      mn = v
    if v > mx:
      mx = v
  if not math.isfinite(mn) or not math.isfinite(mx) or mn == mx:
    # Avoid divide-by-zero; widen slightly
    return float(mn), float(mn + 1.0)
  return float(mn), float(mx)


def _render_ascii(series_list: list[Series], width: int = 80, height: int = 16) -> str:
  # Domains
  x_min = min(min(s.xs) for s in series_list)
  x_max = max(max(s.xs) for s in series_list)
  y_min = min(min(s.ys) for s in series_list)
  y_max = max(max(s.ys) for s in series_list)

  # add small margins
  y_pad = 0.02 * (y_max - y_min) if y_max > y_min else 1.0
  y_min -= y_pad
  y_max += y_pad

  # Grid
  grid = [[' ' for _ in range(width)] for _ in range(height)]

  def to_col(x: float) -> int:
    if x_max == x_min:
      return 0
    t = (x - x_min) / (x_max - x_min)
    return int(round(t * (width - 1)))

  def to_row(y: float) -> int:
    if y_max == y_min:
      return height // 2
    t = (y - y_min) / (y_max - y_min)
    # invert y so larger is higher on plot
    r = int(round((1.0 - t) * (height - 1)))
    return max(0, min(height - 1, r))

  # Plot points
  for s in series_list:
    prev_c, prev_r = None, None
    for x, y in zip(s.xs, s.ys):
      c = to_col(float(x))
      r = to_row(float(y))
      # place char; merge if needed
      ch = s.ch
      cur = grid[r][c]
      grid[r][c] = '#' if cur not in (' ', ch) else ch

      # optional: draw simple vertical line to the previous point for readability
      if prev_c is not None and prev_r is not None:
        if c == prev_c:
          # fill vertical segment between points
          ra, rb = sorted((prev_r, r))
          for rr in range(ra, rb + 1):
            cur2 = grid[rr][c]
            grid[rr][c] = '#' if cur2 not in (' ', ch) else ch
        else:
          # draw a simple stepped line to approximate diagonal
          cc = prev_c
          rr = prev_r
          step = 1 if c > prev_c else -1
          while cc != c:
            cc += step
            cur2 = grid[rr][cc]
            grid[rr][cc] = '#' if cur2 not in (' ', ch) else ch
          # then vertical to r
          ra, rb = sorted((rr, r))
          for vv in range(ra, rb + 1):
            cur2 = grid[vv][c]
            grid[vv][c] = '#' if cur2 not in (' ', ch) else ch
      prev_c, prev_r = c, r

  # Compose output
  lines: list[str] = []
  for i, row in enumerate(grid):
    # left y tick every 4 rows
    if i % 4 == 0:
      # map row back to y for label
      frac = 1.0 - (i / max(1, (height - 1)))
      y_val = y_min + frac * (y_max - y_min)
      label = f"{y_val:8.3f}"
    else:
      label = ' ' * 8
    lines.append(label + ' | ' + ''.join(row))

  # x-axis label line
  x_axis = ' ' * 8 + ' + ' + '-' * (width - 2)
  x_ticks = f"{x_min:.0f}".ljust(width // 2 - 2) + f"{x_max:.0f}".rjust(width // 2)
  lines.append(x_axis)
  lines.append(' ' * 11 + x_ticks)
  return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description='ASCII train/val loss plotter')
  ap.add_argument('--ckpt-dir', type=pathlib.Path, default=pathlib.Path('checkpoints'), help='checkpoints root')
  ap.add_argument('--width', type=int, default=80)
  ap.add_argument('--height', type=int, default=16)
  ns = ap.parse_args(argv)

  meta_path, meta, cfg = _find_latest_meta(ns.ckpt_dir)
  train, val, info = _build_series(meta, cfg)

  # Loss panel
  series_loss = [train]
  if val is not None:
    series_loss.append(val)
  header_loss = (
    f"run: {info['name']}  step: {info['step']}\n"
    f"log_every: {info['log_every']}  eval_every: {info['eval_every']}\n"
    f"loss: train='.'  val='o'  overlap='#'\n"
  )
  body_loss = _render_ascii(series_loss, width=max(20, ns.width), height=max(8, ns.height))

  # Accuracy panel (if available)
  acc_train, acc_val = _build_accuracy_series(meta, cfg)
  output = header_loss + body_loss
  if acc_train is not None or acc_val is not None:
    series_acc: list[Series] = []
    if acc_train is not None:
      series_acc.append(acc_train)
    if acc_val is not None:
      series_acc.append(acc_val)
    header_acc = (
      "\n\n"  # spacer between panels
      f"accuracy: train='.'  val='o'  overlap='#'\n"
    )
    body_acc = _render_ascii(series_acc, width=max(20, ns.width), height=max(8, ns.height))
    output = output + header_acc + body_acc

  print(output)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
