"""
Beer sales forecasting: naive, moving-average, and exponential smoothing.

Supports parsing multiple provided sheet formats in this folder:
- BeerSales-Naive.csv
- BeerSales-MA(3)-MA(4).csv
- BeerSales-ES(0.85)-ES(0.2)-2023.csv
- BeerSales-Double-Exponential-Holt-(0.85)(0.95)-2023.csv

The script extracts the demand series and any provided forecast columns
(e.g., F[t](3), F[t](4), F[t](0.85), F[t](0.2), Naïve), evaluates them, and
optionally computes its own baseline forecasts (naive/MA/ES/Holt) if requested.

It reports MAD, MSE, RMSE, and direction accuracy for each method.

Usage
-----
python beer_sales_analysis.py \
  [--windows 3 4] \
  [--alphas 0.85 0.2] \
  [--holt 0.85 0.95] \
  [--es-init first|mean|<number>] \
  [--optimize-es] [--loss mse|mae] \
  [--milp-es] [--grid-points 101] \
  [--csv BeerSales-ES(0.85)-ES(0.2)-2023.csv]
"""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

try:
  from .timeseries import direction_accuracy
except Exception:
  sys.path.append(os.path.dirname(__file__))
  from timeseries import direction_accuracy

class _NoopBar:
  def update(self, n: int = 1) -> None:
    return

  def close(self) -> None:
    return


def _make_progress(total: int, desc: str):
  """Return a tqdm progress bar when available; otherwise a no-op object.

  Keeps the script dependency-free while providing nice UX when tqdm is
  installed. Use via pbar.update(1) and pbar.close().
  """
  try:
    from tqdm import tqdm  # type: ignore

    return tqdm(total=total, desc=desc, leave=False, unit="it")
  except Exception:
    return _NoopBar()

def _clean_numeric(s: pd.Series) -> pd.Series:
  s = (
    s.astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("\u00A0", "", regex=False)
    .str.strip()
  )
  return pd.to_numeric(s, errors="coerce")


def _read_sheet_with_header(csv_path: Path) -> pd.DataFrame:
  """Read CSV with unknown header row; detect the header line dynamically.

  Robust to prefaces before the true header. Detect a header row if:
  - First cell is one of {Month, t, time, period}, or
  - The row contains tokens like D(t)/D[t] or F(t)/F[t].
  Falls back to the first row otherwise.
  """
  df_raw = pd.read_csv(csv_path, header=None, engine="python", on_bad_lines="skip", dtype=str)
  header_row = None
  header_first = {"month", "t", "time", "period"}
  import re
  patt_dt = re.compile(r"^\s*d\s*([\[\(])t([\]\)])\s*$", re.IGNORECASE)
  patt_ft = re.compile(r"^\s*f\s*([\[\(])t([\]\)])", re.IGNORECASE)
  for i in range(len(df_raw)):
    row_vals = [str(x).strip() for x in df_raw.iloc[i].tolist()]
    if not row_vals:
      continue
    first = row_vals[0].lower()
    if first in header_first:
      header_row = i
      break
    # Look for typical header tokens anywhere in the row
    if any(patt_dt.match(v) or patt_ft.match(v) for v in row_vals if isinstance(v, str)):
      header_row = i
      break
  if header_row is None:
    header_row = 0
  headers = [str(x).strip() for x in df_raw.iloc[header_row].tolist()]
  data = df_raw.iloc[header_row + 1 :].reset_index(drop=True)
  # Deduplicate headers to avoid DataFrame return on label selection
  seen: Dict[str, int] = {}
  dedup: list[str] = []
  for h in headers:
    key = h
    if key in seen:
      seen[key] += 1
      key = f"{h}_{seen[h]}"
    else:
      seen[key] = 0
    dedup.append(key)
  data.columns = dedup
  # Drop fully empty columns
  data = data.dropna(axis=1, how="all")
  return data


def load_beer_sales(csv_path: Path) -> pd.Series:
  """Load demand column from the provided CSV, resilient to extra rows.

  The sheet mixes headers and annotations. We keep the first 4 columns and
  coerce numeric values. The demand column is named 'D[t]' in the sheet.
  """
  df = _read_sheet_with_header(csv_path)

  # Clean numeric-looking columns, covering (t) and [t] notations
  for col in df.columns:
    cn = str(col).strip().lower()
    if (
      cn in {"t", "d[t]", "d(t)", "d", "naive", "naïve"}
      or cn.startswith("f[t]")
      or cn.startswith("f(t)")
    ):
      df[col] = _clean_numeric(df[col])

  # Identify demand column
  demand_col = None
  for cand in df.columns:
    cn = str(cand).strip().lower()
    if cn in {"d[t]", "d(t)", "d"}:
      demand_col = cand
      break
  if demand_col is None:
    demand_col = df.columns[2] if len(df.columns) >= 3 else None
  if demand_col is None:
    raise ValueError("Could not locate demand column (D[t]) in CSV")

  # Use 't' when present for ordering
  if "t" in df.columns:
    df = df.dropna(subset=["t"]).sort_values("t")
  y = df[demand_col].dropna().astype(float).reset_index(drop=True)
  y.index = pd.RangeIndex(start=1, stop=len(y) + 1, step=1)
  y.name = "D"
  return y


def mae_mse_rmse(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float]:
  err = (y_true - y_pred).astype(float)
  mae = float(np.abs(err).mean())
  mse = float((err ** 2).mean())
  rmse = float(np.sqrt(mse))
  return mae, mse, rmse


def naive_forecast(y: pd.Series) -> pd.Series:
  return y.shift(1).rename("naive")


def moving_average_forecast(y: pd.Series, window: int) -> pd.Series:
  # Predict t using the mean of the previous `window` actuals
  return y.shift(1).rolling(window=window, min_periods=window).mean().rename(f"ma{window}")


def exponential_smoothing_forecast(y: pd.Series, alpha: float, init: float | None = None) -> pd.Series:
  """One-parameter ES: F_t = alpha*y_{t-1} + (1-alpha)*F_{t-1}.

  Initializes F_1 = init or y_1. Returns a series aligned to y.
  """
  if not (0 < alpha <= 1):
    raise ValueError("alpha must be in (0, 1]")
  if len(y) == 0:
    return pd.Series(dtype=float, name=f"es{alpha}")
  f = np.empty(len(y))
  f[:] = np.nan
  f[0] = float(y.iloc[0] if init is None else init)
  for i in range(1, len(y)):
    f[i] = alpha * float(y.iloc[i - 1]) + (1.0 - alpha) * f[i - 1]
  return pd.Series(f, index=y.index, name=f"es{alpha}")


def holt_linear_forecast(
  y: pd.Series,
  alpha: float,
  beta: float,
  *,
  level_init: float | None = None,
  trend_init: float | None = None,
) -> pd.Series:
  """Holt's linear (double exponential smoothing).

  State update (additive trend):
    s_t = alpha * y_t + (1 - alpha) * (s_{t-1} + b_{t-1})
    b_t = beta * (s_t - s_{t-1}) + (1 - beta) * b_{t-1}

  One-step-ahead forecast aligned to y (like others):
    F_t = s_{t-1} + b_{t-1} for t >= 1; F_0 = s_0
  """
  if not (0 < alpha <= 1) or not (0 < beta <= 1):
    raise ValueError("alpha and beta must be in (0, 1]")
  n = len(y)
  if n == 0:
    return pd.Series(dtype=float, name=f"holt{alpha}_{beta}")

  yv = y.astype(float).to_numpy()
  s = np.empty(n)
  b = np.empty(n)
  f = np.empty(n)

  s[0] = float(yv[0] if level_init is None else level_init)
  if n >= 2:
    default_trend = yv[1] - yv[0]
  else:
    default_trend = 0.0
  b[0] = float(default_trend if trend_init is None else trend_init)

  f[0] = s[0]
  for t in range(1, n):
    # Forecast for time t uses previous level+trend
    f[t] = s[t - 1] + b[t - 1]
    # Update state with actual y_t
    s_t = alpha * yv[t] + (1.0 - alpha) * (s[t - 1] + b[t - 1])
    b_t = beta * (s_t - s[t - 1]) + (1.0 - beta) * b[t - 1]
    s[t] = s_t
    b[t] = b_t

  name = f"holt{alpha}_{beta}"
  return pd.Series(f, index=y.index, name=name)


def _es_objective(alpha: float, y: pd.Series, init: float, loss: str = "mse") -> float:
  # Constrain within (0,1)
  if not (0.0 < alpha < 1.0):
    return float("inf")
  f = exponential_smoothing_forecast(y, alpha, init=init)
  df = pd.concat([y.rename("y"), f.rename("f")], axis=1).dropna()
  if len(df) == 0:
    return float("inf")
  err = (df["y"] - df["f"]).astype(float)
  if loss == "mae":
    return float(np.abs(err).mean())
  return float((err ** 2).mean())


def optimize_es_alpha(y: pd.Series, init: float, loss: str = "mse") -> tuple[float, dict[str, float]]:
  res = minimize_scalar(
    lambda a: _es_objective(a, y, init, loss=loss),
    bounds=(1e-6, 1 - 1e-6),
    method="bounded",
    options={"xatol": 1e-6},
  )
  alpha_star = float(res.x)
  f = exponential_smoothing_forecast(y, alpha_star, init=init)
  df = pd.concat([y.rename("y"), f.rename("f")], axis=1).dropna()
  mae, mse, rmse = mae_mse_rmse(df["y"], df["f"])
  dacc = direction_accuracy(df["y"].diff().fillna(0), df["f"].diff().fillna(0))
  metrics = {"MAD": mae, "MSE": mse, "RMSE": rmse, "direction_acc": dacc}
  return alpha_star, metrics


def milp_optimize_es_alpha(
  y: pd.Series,
  *,
  init: float,
  grid_points: int = 101,
) -> tuple[float, pd.Series, dict[str, float]]:
  """MILP: choose alpha from a discrete grid to minimize MAE.

  - Decision: one-hot selection over a uniform grid alpha in [0.01, 0.99].
  - Constraints: big-M activation of ES recursion for the selected alpha.
  - Objective: sum of absolute errors |y_t - F_t| for t>=2.

  Returns (alpha_star, forecast_series, metrics).
  """
  try:
    import pulp as pl
  except Exception as e:
    raise SystemExit(
      "PuLP is required for --milp-es. Install with `pip install pulp` (CBC recommended)."
    ) from e

  y = y.astype(float).reset_index(drop=True)
  n = len(y)
  if n < 2:
    raise ValueError("Need at least 2 observations for ES optimization")

  # Alpha grid
  grid_points = max(3, int(grid_points))
  alphas = np.linspace(0.01, 0.99, grid_points)
  K = range(len(alphas))
  T = range(n)

  # Bounds and big-M
  y_min, y_max = float(y.min()), float(y.max())
  L = min(y_min, float(init)) - 1.0
  U = max(y_max, float(init)) + 1.0
  M = (U - L) * 10.0

  # Model
  model = pl.LpProblem("ES_Alpha_Selection", pl.LpMinimize)

  # Variables
  F = pl.LpVariable.dicts("F", T, lowBound=L, upBound=U, cat=pl.LpContinuous)
  e = pl.LpVariable.dicts("e", T, lowBound=0.0, cat=pl.LpContinuous)  # absolute errors
  z = pl.LpVariable.dicts("z", K, lowBound=0, upBound=1, cat=pl.LpBinary)

  # Initial forecast
  model += F[0] == float(init)

  # One-hot alpha selection
  model += pl.lpSum(z[k] for k in K) == 1

  # ES recursion with big-M activation for each alpha k, times t>=1 (i.e., index 1..n-1)
  pbar_rec = _make_progress((n - 1) * len(K), "MILP: building ES recursion")
  for t in range(1, n):
    for k in K:
      a = float(alphas[k])
      # F_t - (1-a) F_{t-1} - a y_{t-1} = 0 if z_k = 1; relaxed otherwise
      model += F[t] - (1 - a) * F[t - 1] - a * float(y.iloc[t - 1]) <= M * (1 - z[k])
      model += F[t] - (1 - a) * F[t - 1] - a * float(y.iloc[t - 1]) >= -M * (1 - z[k])
      pbar_rec.update(1)
  pbar_rec.close()

  # Absolute error constraints for t>=1 (first error from second point)
  pbar_err = _make_progress((n - 1), "MILP: building abs-error constraints")
  for t in range(1, n):
    model += e[t] >= float(y.iloc[t]) - F[t]
    model += e[t] >= F[t] - float(y.iloc[t])
    pbar_err.update(1)
  pbar_err.close()
  # e[0] unused; fix to 0 to keep model tidy
  model += e[0] == 0

  # Objective: minimize sum of abs errors
  model += pl.lpSum(e[t] for t in range(1, n))

  # Solve with CBC (default in PuLP). Silence output.
  solver = pl.PULP_CBC_CMD(msg=False)
  print("Solving MILP (CBC)...", flush=True)
  status = model.solve(solver)
  if pl.LpStatus[status] != "Optimal":
    raise RuntimeError(f"MILP did not reach optimality: {pl.LpStatus[status]}")

  # Extract alpha
  z_vals = np.array([pl.value(z[k]) for k in K])
  k_star = int(np.argmax(z_vals))
  alpha_star = float(alphas[k_star])

  # Build forecast series from F variables (t=0..n-1)
  f_vals = [pl.value(F[t]) for t in T]
  f_ser = pd.Series(f_vals, index=pd.RangeIndex(start=1, stop=n + 1), name=f"es{alpha_star}")

  # Metrics (align like other evaluations: drop first entry with no error)
  df = pd.concat([
    pd.Series(y.values, index=pd.RangeIndex(start=1, stop=n + 1), name="y"),
    f_ser,
  ], axis=1).iloc[1:]
  mae, mse, rmse = mae_mse_rmse(df["y"], df[f"es{alpha_star}"])
  dacc = direction_accuracy(df["y"].diff().fillna(0), df[f"es{alpha_star}"].diff().fillna(0))
  metrics = {"MAD": mae, "MSE": mse, "RMSE": rmse, "direction_acc": dacc}
  return alpha_star, f_ser, metrics


def parse_sheet_forecasts(csv_path: Path) -> Dict[str, pd.Series]:
  """Parse forecast columns from the sheet if present.

  Returns keys: 'naive', 'ma<k>', 'es<alpha>', 'holt<alpha>_<beta>', and 'holt' (plain F(t)) mapped to series.
  """
  import re

  df = _read_sheet_with_header(csv_path)
  # Clean each positional column to avoid duplicate-label issues
  for i in range(df.shape[1]):
    df.iloc[:, i] = _clean_numeric(df.iloc[:, i])

  out: Dict[str, pd.Series] = {}

  def shape_series(name: str, s: pd.Series) -> pd.Series:
    s = s.dropna().astype(float).reset_index(drop=True)
    s.index = pd.RangeIndex(start=1, stop=len(s) + 1)
    s.name = name
    return s

  # Naive (match any column labeled case-insensitively as naive)
  for i, cand in enumerate(df.columns):
    if str(cand).strip().lower() in {"naive", "naïve"}:
      out["naive"] = shape_series("naive", df.iloc[:, i])

  # F[t](alpha)(beta) or F(t)(alpha)(beta) for Holt double exponential smoothing
  for i, cand in enumerate(df.columns):
    s = str(cand)
    m2 = re.match(r"\s*F\s*(?:\[t\]|\(t\))[^\d-]*\(([^)]+)\)\s*\(([^)]+)\)\s*$", s)
    if not m2:
      continue
    a_val, b_val = m2.group(1).strip(), m2.group(2).strip()
    try:
      a_num = float(a_val)
      b_num = float(b_val)
    except ValueError:
      continue
    key = f"holt{a_num}_{b_num}"
    out[key] = shape_series(key, df.iloc[:, i])

  # F[t](...) or F(t)(...) single-parameter (MA or single ES)
  for i, cand in enumerate(df.columns):
    m = re.match(r"\s*F\s*(?:\[t\]|\(t\))\s*\(([^)]+)\)\s*$", str(cand))
    if not m:
      continue
    val = m.group(1).strip()
    try:
      num = float(val)
    except ValueError:
      continue
    if float(num).is_integer():
      key = f"ma{int(num)}"
    else:
      key = f"es{num}"
    out[key] = shape_series(key, df.iloc[:, i])

  # Plain F(t) or F[t] one-step forecasts from sheet (no parameters)
  for i, cand in enumerate(df.columns):
    if re.match(r"^\s*F\s*(?:\[t\]|\(t\))\s*$", str(cand)):
      out.setdefault("holt", shape_series("holt", df.iloc[:, i]))

  return out


def build_arg_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(description="Beer sales: naive, moving-average, ES, and Holt-linear")
  ap.add_argument("--windows", type=int, nargs="*", default=[3], help="Moving average window sizes (default: 3)")
  ap.add_argument("--alphas", type=float, nargs="*", default=[], help="Exponential smoothing alphas to compute")
  ap.add_argument("--es-init", type=str, default="first", help="ES init: 'first', 'mean', or a number")
  ap.add_argument("--optimize-es", action="store_true", help="Search for alpha that minimizes chosen loss")
  ap.add_argument("--loss", choices=["mse", "mae"], default="mse", help="Loss to optimize for ES alpha")
  ap.add_argument("--milp-es", action="store_true", help="Use MILP (PuLP/CBC) to select alpha from a grid minimizing MAE")
  ap.add_argument("--grid-points", type=int, default=101, help="Number of alpha grid points in [0.01,0.99] for MILP")
  ap.add_argument(
    "--holt",
    type=float,
    nargs=2,
    action="append",
    default=[],
    metavar=("ALPHA", "BETA"),
    help="Compute Holt linear forecasts for (alpha, beta) pairs",
  )
  ap.add_argument(
    "--csv",
    type=Path,
    default=Path(__file__).with_name("BeerSales-Naive.csv"),
    help="Path to the beer sales CSV",
  )
  return ap


def _resolve_es_init(y: pd.Series, es_init: str) -> float:
  s = es_init.strip().lower()
  if s == "first":
    return float(y.iloc[0])
  if s == "mean":
    return float(y.mean())
  try:
    return float(es_init)
  except ValueError:
    raise SystemExit("--es-init must be 'first', 'mean', or a number")


def compute_methods(
  y: pd.Series,
  *,
  windows: list[int],
  alphas: list[float],
  es_init: float,
  holt_pairs: list[tuple[float, float]],
  provided: Dict[str, pd.Series] | None = None,
) -> Dict[str, pd.Series]:
  methods: Dict[str, pd.Series] = {}
  # Built-ins
  methods["naive"] = naive_forecast(y)
  for w in windows:
    methods[f"ma{w}"] = moving_average_forecast(y, w)
  for a in alphas:
    methods[f"es{a}"] = exponential_smoothing_forecast(y, a, init=es_init)
  for a, b in holt_pairs:
    key = f"holt{a}_{b}"
    methods[key] = holt_linear_forecast(y, a, b, level_init=es_init)
  # Merge provided from sheet
  if provided:
    for k, v in provided.items():
      methods.setdefault(k, v)
  return methods


def evaluate_methods(y: pd.Series, methods: Dict[str, pd.Series]) -> Dict[str, dict[str, float | int | None]]:
  out: Dict[str, dict[str, float | int | None]] = {}
  for key in sorted(methods.keys()):
    yhat = methods[key]
    df_eval = pd.concat([y.rename("y"), yhat.rename(key)], axis=1).dropna()
    if len(df_eval) == 0:
      continue
    mae, mse, rmse = mae_mse_rmse(df_eval["y"], df_eval[key])
    dacc = direction_accuracy(df_eval["y"].diff().fillna(0), df_eval[key].diff().fillna(0))
    out[key] = {
      "rows": int(len(df_eval)),
      "MAD": float(round(mae, 6)),
      "MSE": float(round(mse, 6)),
      "RMSE": float(round(rmse, 6)),
      "direction_acc": float(round(dacc, 6)) if not np.isnan(dacc) else None,
    }
  return out


def run(args: argparse.Namespace) -> None:
  y = load_beer_sales(args.csv)
  es_init = _resolve_es_init(y, args.es_init)

  provided = parse_sheet_forecasts(args.csv)
  holt_pairs = [(float(p[0]), float(p[1])) for p in args.holt]
  methods = compute_methods(
    y,
    windows=list(args.windows),
    alphas=list(args.alphas),
    es_init=es_init,
    holt_pairs=holt_pairs,
    provided=provided,
  )

  # Scalar ES optimization
  if args.optimize_es:
    alpha_star, metrics = optimize_es_alpha(y, init=es_init, loss=args.loss)
    methods[f"es{round(alpha_star, 6)}"] = exponential_smoothing_forecast(y, alpha_star, init=es_init)
    print()
    print(f"Optimized ES alpha (loss={args.loss}): {alpha_star:.6f}")
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()})

  # MILP alpha selection
  if args.milp_es:
    try:
      alpha_milp, f_milp, metrics_milp = milp_optimize_es_alpha(y, init=es_init, grid_points=args.grid_points)
      key = f"es{round(alpha_milp, 6)}_milp"
      methods[key] = f_milp
      print()
      print(f"MILP optimized ES alpha: {alpha_milp:.6f} (grid points={args.grid_points})")
      print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics_milp.items()})
    except SystemExit as e:
      print(str(e))
      return

  print("\nInput length:", len(y))
  results = evaluate_methods(y, methods)
  for key in sorted(results.keys()):
    m = results[key]
    print()
    print(f"{key} forecast metrics (rows={m['rows']}):")
    print({
      "MAD": round(float(m["MAD"]), 3),
      "MSE": round(float(m["MSE"]), 3),
      "RMSE": round(float(m["RMSE"]), 3),
      "direction_acc": (round(float(m["direction_acc"]), 3) if m["direction_acc"] is not None else None),
    })


def main() -> None:
  ap = build_arg_parser()
  args = ap.parse_args()
  run(args)


if __name__ == "__main__":
  main()
