#!/usr/bin/env python3
"""
Straightforward trend-plus-seasonality regression for the Commerce 4QA3 dataset.

The script expects a tidy CSV with at least `t` and `Sales (Y)` columns. An optional
`Q` column can be provided; otherwise it is inferred from `t`. A linear trend plus
quarterly seasonal dummies is fit via ordinary least squares, and the script can
emit future forecasts as well as an optional plot.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Fit a trend+seasonality regression on the TV sets datasets."
  )
  parser.add_argument(
    "data_path",
    type=Path,
    nargs="?",
    default=Path(__file__).with_name("time_series_tv_sets.csv"),
    help="Path to the CSV file with columns t, Sales (Y), and optionally Q.",
  )
  parser.add_argument(
    "--horizon",
    type=int,
    default=4,
    help="Number of future quarters to forecast once the model is fitted.",
  )
  parser.add_argument(
    "--export",
    type=Path,
    default=None,
    help="Optional path to persist the cleaned dataset and forecasts as CSV.",
  )
  parser.add_argument(
    "--plot",
    type=Path,
    default=None,
    help="Optional path to save a Matplotlib chart of observed, fitted, and forecast values.",
  )
  parser.add_argument(
    "--linear-trend",
    action="store_true",
    help="Compute and print a pure linear trendline that matches Excel's default chart trend.",
  )
  parser.add_argument(
    "--seasonality",
    choices=("quarter", "none"),
    default="quarter",
    help="Quarter-seasonal dummies (default) or disable with 'none' for a pure trend regression.",
  )
  return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
  tidy_error: Exception | None = None
  try:
    return _load_tidy_csv(path)
  except Exception as exc:
    tidy_error = exc
  try:
    return _load_messy_csv(path)
  except Exception as messy_error:
    message = (
      f"Unable to parse dataset {path}. Tidy parser error: {tidy_error}. "
      f"Messy parser error: {messy_error}."
    )
    raise ValueError(message) from messy_error


def _load_tidy_csv(path: Path) -> pd.DataFrame:
  df = pd.read_csv(path)
  df.columns = [col.strip() for col in df.columns]
  if "Sales" in df.columns and "Sales (Y)" not in df.columns:
    df = df.rename(columns={"Sales": "Sales (Y)"})
  required = {"t", "Sales (Y)"}
  missing = required - set(df.columns)
  if missing:
    missing_cols = ", ".join(sorted(missing))
    raise ValueError(f"Dataset {path} is missing required columns: {missing_cols}.")
  return _finalise_dataframe(df)


def _load_messy_csv(path: Path) -> pd.DataFrame:
  with path.open(newline="") as handle:
    rows = [[cell.strip() for cell in row] for row in csv.reader(handle)]
  expected = ("Year", "Q", "t", "Sales (Y)")
  header_index = None
  for idx, row in enumerate(rows):
    if all(label in row for label in expected):
      header_index = idx
      break
  if header_index is None:
    raise ValueError(f"Dataset {path} does not contain required headers {expected}.")
  header_row = rows[header_index]
  column_index = {label: header_row.index(label) for label in expected}
  max_index = max(column_index.values())
  records: list[dict[str, str]] = []
  last_year: str | None = None
  for raw in rows[header_index + 1:]:
    if not any(raw):
      continue
    if len(raw) <= max_index:
      raw = raw + [""] * (max_index + 1 - len(raw))
    year_token = raw[column_index["Year"]]
    if year_token:
      last_year = year_token
    t_token = raw[column_index["t"]]
    sales_token = raw[column_index["Sales (Y)"]]
    q_token = raw[column_index["Q"]]
    if not t_token or not sales_token:
      continue
    if last_year is None:
      raise ValueError(
        f"Encountered observation {raw} before any year value was observed."
      )
    records.append(
      {
        "Year": last_year,
        "Q": q_token,
        "t": t_token,
        "Sales (Y)": sales_token,
      }
    )
  if not records:
    raise ValueError(f"No usable observations recovered from {path}.")
  df = pd.DataFrame.from_records(records)
  return _finalise_dataframe(df)


def _finalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["t"] = pd.to_numeric(df["t"], errors="coerce")
  df["Sales (Y)"] = pd.to_numeric(df["Sales (Y)"], errors="coerce")
  if "Q" in df.columns:
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce")
  df = df.dropna(subset=["t", "Sales (Y)"]).reset_index(drop=True)
  df["t"] = df["t"].astype(int)
  if "Q" not in df.columns or df["Q"].isna().all():
    df["Q"] = ((df["t"] - 1) % 4 + 1).astype(int)
  else:
    df["Q"] = df["Q"].fillna(0).astype(int)
    df.loc[df["Q"] <= 0, "Q"] = ((df["t"] - 1) % 4 + 1).astype(int)
  if "Year" in df.columns:
    year = pd.to_numeric(df["Year"], errors="coerce")
    if year.notna().any():
      df["Year"] = year.ffill().astype("Int64")
    else:
      df = df.drop(columns=["Year"])
  return df


def build_design_matrix(
  df: pd.DataFrame, seasonality: str
) -> tuple[np.ndarray, list[str], np.ndarray]:
  y = df["Sales (Y)"].to_numpy(dtype=float)
  trend = df["t"].to_numpy(dtype=float).reshape(-1, 1)
  intercept = np.ones((len(df), 1), dtype=float)
  if seasonality == "quarter":
    seasonal = pd.get_dummies(df["Q"].astype(int), prefix="Q", drop_first=True)
    X = np.hstack([intercept, trend, seasonal.to_numpy(dtype=float)])
    feature_names = ["intercept", "trend", *seasonal.columns.tolist()]
  else:
    X = np.hstack([intercept, trend])
    feature_names = ["intercept", "trend"]
  return X, feature_names, y


def solve_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
  beta, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
  fitted = X @ beta
  residual = y - fitted
  ss_res = float(np.dot(residual, residual))
  ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
  r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
  rmse = float(np.sqrt(ss_res / len(y)))
  cond = float(singular_values[0] / singular_values[-1]) if singular_values[-1] else float("inf")
  stats = {
    "fitted": fitted,
    "residuals": residual,
    "rank": float(rank),
    "condition_number": cond,
    "ss_res": ss_res,
    "ss_tot": ss_tot,
    "r2": r2,
    "rmse": rmse,
  }
  return beta, stats


def format_coefficients(beta: np.ndarray, feature_names: list[str]) -> list[str]:
  return [f"{name:>10s}: {value: .4f}" for name, value in zip(feature_names, beta)]


def fit_linear_trend(df: pd.DataFrame) -> tuple[float, float]:
  result = linregress(df["t"].to_numpy(dtype=float), df["Sales (Y)"].to_numpy(dtype=float))
  return float(result.intercept), float(result.slope)


def forecast(
  beta: np.ndarray, feature_names: list[str], last_t: int, horizon: int, seasonality: str
) -> pd.DataFrame:
  records = []
  for step in range(1, horizon + 1):
    t_future = last_t + step
    quarter = (t_future - 1) % 4 + 1
    if seasonality == "quarter":
      vector = np.zeros(len(feature_names), dtype=float)
      vector[0] = 1.0
      vector[1] = t_future
      for idx, name in enumerate(feature_names[2:], start=2):
        vector[idx] = 1.0 if name == f"Q_{quarter}" else 0.0
    else:
      vector = np.array([1.0, float(t_future)], dtype=float)
    prediction = float(np.dot(beta, vector))
    records.append({"future_t": t_future, "quarter": quarter, "prediction": prediction})
  return pd.DataFrame.from_records(records)


def plot_series(
  data: pd.DataFrame,
  fitted: np.ndarray,
  projection: pd.DataFrame,
  linear_coeffs: tuple[float, float] | None,
  plot_path: Path,
  spec_label: str,
) -> None:
  plot_path = plot_path.with_suffix(".png") if plot_path.suffix == "" else plot_path
  plot_path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(8, 4.8))
  ax.plot(data["t"], data["Sales (Y)"], "o", label="observed")
  ax.plot(data["t"], fitted, "-", label="fitted")
  if linear_coeffs is not None:
    intercept, slope = linear_coeffs
    x_vals = np.arange(data["t"].min(), data["t"].max() + len(projection) + 1)
    ax.plot(x_vals, intercept + slope * x_vals, "--", color="#d62728", label="linear trend")
  if not projection.empty:
    ax.plot(projection["future_t"], projection["prediction"], "--", color="#2ca02c", label="forecast")
    ax.scatter(projection["future_t"], projection["prediction"], marker="x", color="#2ca02c")
  ax.set_xlabel("t (time index)")
  ax.set_ylabel("Sales (Y)")
  ax.set_title(f"TV sets regression ({spec_label})")
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend()
  fig.tight_layout()
  fig.savefig(plot_path, dpi=200)
  plt.close(fig)
  print(f"\nSaved plot to {plot_path}")


def main() -> None:
  args = parse_args()
  data = load_dataset(args.data_path)
  X, feature_names, y = build_design_matrix(data, args.seasonality)
  beta, stats = solve_ols(X, y)
  spec = "quarter dummies" if args.seasonality == "quarter" else "no seasonality"
  linear_coeffs = None
  if args.linear_trend:
    linear_coeffs = fit_linear_trend(data)
    intercept, slope = linear_coeffs
    print("Excel-style linear trendline:")
    print(f"  intercept: {intercept:.4f}")
    print(f"  slope: {slope:.4f}")
  print(f"Specification: trend + {spec}")
  print("Fitted coefficients:")
  for line in format_coefficients(beta, feature_names):
    print(f"  {line}")
  print("\nModel diagnostics:")
  print(f"  Observations: {len(y)}")
  print(f"  Rank: {stats['rank']:.0f}")
  print(f"  Condition number: {stats['condition_number']:.2f}")
  print(f"  R^2: {stats['r2']:.4f}")
  print(f"  RMSE: {stats['rmse']:.4f}\n")
  last_t = int(data["t"].iloc[-1])
  horizon = max(args.horizon, 0)
  projection = forecast(beta, feature_names, last_t, horizon, args.seasonality) if horizon else pd.DataFrame()
  if not projection.empty:
    print("Quarterly forecasts:")
    for row in projection.to_dict("records"):
      print(
        f"  t={row['future_t']:2d}, quarter={row['quarter']}: prediction={row['prediction']:.4f}"
      )
  if args.plot is not None:
    plot_series(data, stats["fitted"], projection, linear_coeffs, args.plot, spec)
  if args.export is not None:
    export_path = args.export
    export_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = data.copy()
    cleaned["fitted"] = stats["fitted"]
    cleaned["residual"] = stats["residuals"]
    cleaned.to_csv(export_path, index=False)
    print(f"\nPersisted cleaned dataset with fitted values to {export_path}")


if __name__ == "__main__":
  main()
