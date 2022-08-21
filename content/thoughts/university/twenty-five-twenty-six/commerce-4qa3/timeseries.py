"""
Time series prediction models for financial market data.

This module provides a small, dependency-light toolkit for one-step-ahead
forecasting using classical baselines and scikit-learn regressors, with a
consistent, typed interface oriented toward financial price or return series.

Key features
- Baselines: Naive (last value), SMA, EMA
- ML models: Linear, Ridge, Lasso, RandomForest, GradientBoosting (sklearn)
- Feature maker for lagged values and simple technical features
- Walk-forward backtesting with direction-accuracy and error metrics

Dependencies: numpy, pandas, scikit-learn (already in project deps)

Example
-------
>>> import numpy as np
>>> import pandas as pd
>>> from timeseries import TimeSeriesRegressor, ModelName
>>> idx = pd.date_range("2020-01-01", periods=250, freq="B")
>>> # Simulated price path
>>> rng = np.random.default_rng(0)
>>> ret = rng.normal(0, 0.01, size=len(idx))
>>> price = pd.Series(100 * (1 + pd.Series(ret, index=idx)).cumprod(), index=idx)
>>> # Predict log-returns using a Ridge model and 5 lags
>>> y = price.pct_change().dropna()
>>> model = TimeSeriesRegressor(model="ridge", n_lags=5)
>>> model.fit(y)
>>> fc1 = model.forecast(y, steps=5)  # next 5 one-step recursive forecasts

Notes
-----
- Targets can be prices or returns. For financial forecasting it is common to
  model log-returns or percent-returns for stationarity.
- Exogenous features can be supplied and will be lagged by one step by default.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Literal, Sequence, overload

import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelName = Literal[
  "naive",
  "sma",
  "ema",
  "linear",
  "ridge",
  "lasso",
  "random_forest",
  "gbrt",
]


def _check_series(y: pd.Series, name: str = "y") -> pd.Series:
  """Validate and normalize a pandas Series.

  Ensures a pandas Series with a monotonic increasing index and float dtype.
  """
  if not isinstance(y, pd.Series):
    raise TypeError(f"{name} must be a pandas Series")
  if y.index.has_duplicates:
    y = y[~y.index.duplicated(keep="last")]
  if not y.index.is_monotonic_increasing:
    y = y.sort_index()
  # Ensure float dtype for models
  return y.astype(float)


def _lag(df: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame:
  """Return a DataFrame with `n`-step lag applied to each column.

  The output columns are suffixed with `_lag{n}`.
  """
  if isinstance(df, pd.Series):
    df = df.to_frame(df.name or "y")
  out = df.shift(n)
  out.columns = [f"{c}_lag{n}" for c in df.columns]
  return out


def _rolling_mean(y: pd.Series, window: int) -> pd.Series:
  x = y.rolling(window=window, min_periods=window).mean()
  return x.rename(f"{y.name or 'y'}_sma{window}")


def _ema(y: pd.Series, span: int) -> pd.Series:
  x = y.ewm(span=span, adjust=False, min_periods=span).mean()
  return x.rename(f"{y.name or 'y'}_ema{span}")


@dataclass(slots=True)
class FeatureSpec:
  """Specification for automatic feature generation.

  Attributes:
    n_lags: Number of lagged target features to include.
    sma_windows: Rolling mean window sizes to include (shifted by one step).
    ema_spans: EMA spans to include (shifted by one step).
    add_pct_return: If True, include 1-step lagged percent return feature.
    exog_lag: If provided, lag exogenous columns by this many steps (default 1).
  """

  n_lags: int = 5
  sma_windows: Sequence[int] = ()
  ema_spans: Sequence[int] = ()
  add_pct_return: bool = True
  exog_lag: int = 1


def make_features(
  y: pd.Series,
  *,
  feature_spec: FeatureSpec | None = None,
  exog: pd.DataFrame | None = None,
  horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
  """Transform a univariate target into a supervised learning dataset.

  Predicts ``y[t + horizon]`` from features observable at ``t``.

  Args:
    y: Target series (price or return). Index should increase with time.
    feature_spec: Configuration for lag/technical features. If None, uses
      defaults (5 lags + 1-step lagged percent return).
    exog: Optional exogenous features aligned to ``y``. They will be lagged by
      ``feature_spec.exog_lag`` (default 1), so only information available at
      time ``t`` is used to predict ``t + horizon``.
    horizon: Forecast horizon measured in steps of ``y``. Typically 1.

  Returns:
    (X, y_target) where X is a feature DataFrame and y_target is the future
    target aligned to X.
  """
  y = _check_series(y)
  if y.name is None:
    y = y.rename("y")

  spec = feature_spec or FeatureSpec()
  if spec.n_lags < 1:
    raise ValueError("n_lags must be >= 1")
  if horizon < 1:
    raise ValueError("horizon must be >= 1")

  feats: list[pd.Series | pd.DataFrame] = []

  # Lagged target features
  for i in range(1, spec.n_lags + 1):
    feats.append(_lag(y, i))

  # Percent return feature (lagged so it's known at time t)
  if spec.add_pct_return:
    pct = y.pct_change().rename(f"{y.name}_ret1")
    feats.append(pct.shift(1))

  # SMA and EMA features (shifted to avoid peeking)
  for w in spec.sma_windows:
    feats.append(_rolling_mean(y, w).shift(1))
  for s in spec.ema_spans:
    feats.append(_ema(y, s).shift(1))

  # Exogenous variables (lagged)
  if exog is not None:
    if not isinstance(exog, pd.DataFrame):
      raise TypeError("exog must be a pandas DataFrame when provided")
    exog = exog.sort_index().astype(float)
    exog = exog.reindex(y.index)
    feats.append(_lag(exog, spec.exog_lag))

  X = pd.concat(feats, axis=1)
  y_target = y.shift(-horizon)

  # Align and drop rows with any NaNs
  data = pd.concat([X, y_target.rename("target")], axis=1).dropna()
  X_out = data.drop(columns=["target"])
  y_out = data["target"]
  return X_out, y_out


class _BaseModel:
  """Minimal interface shared by all models used here."""

  def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # pragma: no cover - simple pass-through
    raise NotImplementedError

  def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - simple pass-through
    raise NotImplementedError


class _NaiveModel(_BaseModel):
  """Predicts the last observed target value (naive)."""

  def __init__(self, fallback: float | None = None) -> None:
    self._last: float | None = fallback

  def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
    if len(y) == 0:
      raise ValueError("y is empty")
    self._last = float(y.iloc[-1])

  def predict(self, X: pd.DataFrame) -> np.ndarray:
    if self._last is None:
      raise RuntimeError("Model not fitted")
    return np.repeat(self._last, repeats=len(X))


class _SmaModel(_BaseModel):
  """Predicts with a simple moving average of the last `window` targets."""

  def __init__(self, window: int = 5) -> None:
    if window < 1:
      raise ValueError("window must be >= 1")
    self.window = window
    self._last_mean: float | None = None

  def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
    if len(y) < self.window:
      raise ValueError("Not enough observations for SMA window")
    self._last_mean = float(y.tail(self.window).mean())

  def predict(self, X: pd.DataFrame) -> np.ndarray:
    if self._last_mean is None:
      raise RuntimeError("Model not fitted")
    return np.repeat(self._last_mean, repeats=len(X))


class _EmaModel(_BaseModel):
  """Predicts with an exponential moving average with span `span`."""

  def __init__(self, span: int = 10) -> None:
    if span < 1:
      raise ValueError("span must be >= 1")
    self.span = span
    self._last_ema: float | None = None

  def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
    if len(y) < self.span:
      raise ValueError("Not enough observations for EMA span")
    self._last_ema = float(y.ewm(span=self.span, adjust=False).mean().iloc[-1])

  def predict(self, X: pd.DataFrame) -> np.ndarray:
    if self._last_ema is None:
      raise RuntimeError("Model not fitted")
    return np.repeat(self._last_ema, repeats=len(X))


class _SklearnModel(_BaseModel):
  """Adapter for scikit-learn regressors with sensible defaults."""

  def __init__(self, name: ModelName) -> None:
    self.name = name
    self.model = self._make_estimator(name)

  @staticmethod
  def _make_estimator(name: ModelName) -> Pipeline | RandomForestRegressor | GradientBoostingRegressor:
    if name == "linear":
      return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
      ])
    if name == "ridge":
      return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(alpha=1.0, random_state=0)),
      ])
    if name == "lasso":
      return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Lasso(alpha=0.001, max_iter=50_000, random_state=0)),
      ])
    if name == "random_forest":
      return RandomForestRegressor(
        n_estimators=300,
        random_state=0,
        n_jobs=-1,
        max_depth=None,
        min_samples_split=2,
      )
    if name == "gbrt":
      return GradientBoostingRegressor(random_state=0)
    raise ValueError(f"Unsupported sklearn model: {name}")

  def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
    self.model.fit(X.values, y.values)

  def predict(self, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(self.model.predict(X.values))


def _build_model(name: ModelName) -> _BaseModel:
  if name == "naive":
    return _NaiveModel()
  if name == "sma":
    return _SmaModel(window=5)
  if name == "ema":
    return _EmaModel(span=10)
  if name in {"linear", "ridge", "lasso", "random_forest", "gbrt"}:
    return _SklearnModel(name)
  raise ValueError(f"Unknown model name: {name}")


@dataclass(slots=True)
class TimeSeriesRegressor:
  """One-step-ahead forecaster with automatic feature engineering.

  This wrapper converts a series into a supervised ML problem and fits either a
  simple baseline or an sklearn regressor. Forecasts for multi-step horizons
  use recursive one-step predictions.

  Args:
    model: Model name from ModelName.
    n_lags: Number of lagged target features.
    feature_spec: Optional FeatureSpec; if provided overrides n_lags.
    horizon: One-step ahead training target horizon. Forecasting with
      ``steps > 1`` is done recursively from this base model.
  """

  model: ModelName = "ridge"
  n_lags: int = 5
  feature_spec: FeatureSpec | None = None
  horizon: int = 1
  _impl: _BaseModel = field(init=False, default=None)
  _cols: list[str] | None = None

  def __post_init__(self) -> None:
    if self.feature_spec is None:
      self.feature_spec = FeatureSpec(n_lags=self.n_lags)
    else:
      # Ensure n_lags mirrors provided spec for clarity
      self.n_lags = self.feature_spec.n_lags
    self._impl: _BaseModel = _build_model(self.model)

  def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "TimeSeriesRegressor":
    """Fit the underlying model using generated features.

    Args:
      y: Target series (prices or returns).
      exog: Optional exogenous variables aligned to ``y``.
    """
    X, yt = make_features(y, feature_spec=self.feature_spec, exog=exog, horizon=self.horizon)
    if len(yt) < 1:
      raise ValueError("Not enough data after feature generation; increase data length or reduce lags.")
    self._impl.fit(X, yt)
    self._cols = list(X.columns)
    return self

  @overload
  def predict(self, y: pd.Series, exog: pd.DataFrame | None = None) -> pd.Series: ...

  def predict(self, y: pd.Series, exog: pd.DataFrame | None = None) -> pd.Series:
    """In-sample, aligned one-step-ahead predictions over the fit domain.

    Returns a series indexed like the supervised target (i.e., shifted by
    ``-horizon``). Useful to assess in-sample fit and residual structure.
    """
    X, yt = make_features(y, feature_spec=self.feature_spec, exog=exog, horizon=self.horizon)
    if self._cols is None:
      raise RuntimeError("Model not fitted")
    X = X[self._cols]
    pred = self._impl.predict(X)
    return pd.Series(pred, index=yt.index, name="y_hat")

  def forecast(self, y: pd.Series, exog: pd.DataFrame | None = None, *, steps: int = 1) -> pd.Series:
    """Out-of-sample forecasts for ``steps`` ahead using recursive strategy.

    Args:
      y: Historical target series up to time T.
      exog: Optional exogenous variables up to time T. Future exog is not used
        in this simple implementation.
      steps: Number of forward steps to forecast recursively.

    Returns:
      A series of length ``steps`` indexed with inferred future timestamps if
      possible, otherwise a simple RangeIndex starting at 0.
    """
    if steps < 1:
      raise ValueError("steps must be >= 1")
    y = _check_series(y)

    # Build features on the full history and grab the last row to forecast t+1
    X_hist, _ = make_features(y, feature_spec=self.feature_spec, exog=exog, horizon=self.horizon)
    if self._cols is None:
      raise RuntimeError("Model not fitted")
    X_hist = X_hist[self._cols]
    if len(X_hist) == 0:
      raise ValueError("Not enough history to form features; provide more data.")

    preds: list[float] = []
    y_work = y.copy()

    # Infer future index if DatetimeIndex with fixed freq
    if isinstance(y_work.index, pd.DatetimeIndex) and y_work.index.freq is not None:
      future_index = pd.date_range(start=y_work.index[-1] + y_work.index.freq, periods=steps, freq=y_work.index.freq)
    else:
      future_index = pd.RangeIndex(start=0, stop=steps)

    for _ in range(steps):
      # Recompute features for the current last point
      X_last, _ = make_features(y_work, feature_spec=self.feature_spec, exog=exog, horizon=self.horizon)
      X_last = X_last[self._cols].tail(1)
      y_hat = float(self._impl.predict(X_last)[0])
      preds.append(y_hat)
      # Append prediction for recursive forecasting
      y_work = pd.concat([y_work, pd.Series([y_hat], index=pd.Index([future_index[len(preds) - 1]]), name=y_work.name)])

    return pd.Series(preds, index=future_index, name="forecast")


@dataclass(slots=True)
class BacktestResult:
  """Container for backtest outputs and summary metrics."""

  y_true: pd.Series
  y_pred: pd.Series
  rmse: float
  mae: float
  mape: float
  r2: float
  direction_acc: float


def direction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
  """Compute fraction of times the sign matched (ignores zeros)."""
  s_true = np.sign(y_true.values)
  s_pred = np.sign(y_pred.values)
  mask = s_true != 0
  if mask.sum() == 0:
    return float("nan")
  return float(np.mean(s_true[mask] == s_pred[mask]))


def backtest_walk_forward(
  y: pd.Series,
  *,
  model: TimeSeriesRegressor,
  n_splits: int = 5,
  exog: pd.DataFrame | None = None,
) -> BacktestResult:
  """Walk-forward validation using sklearn's TimeSeriesSplit.

  Args:
    y: Target series.
    model: Configured, unfitted TimeSeriesRegressor instance. It is refit on
      each training fold.
    n_splits: Number of time-ordered splits.
    exog: Optional exogenous features aligned to ``y``.

  Returns:
    BacktestResult with concatenated predictions over all test folds.
  """
  y = _check_series(y)
  X_full, y_full = make_features(y, feature_spec=model.feature_spec, exog=exog, horizon=model.horizon)
  if len(y_full) < n_splits + 1:
    raise ValueError("Not enough data for the requested number of splits")

  tscv = TimeSeriesSplit(n_splits=n_splits)
  preds: list[pd.Series] = []
  trues: list[pd.Series] = []

  for tr_idx, te_idx in tscv.split(X_full):
    X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
    y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]

    # Fit a fresh instance to avoid state leakage
    m = TimeSeriesRegressor(
      model=model.model,
      n_lags=model.n_lags,
      feature_spec=model.feature_spec,
      horizon=model.horizon,
    )
    # Internal fit expects raw series; supply pre-made features via private path
    m._impl = _build_model(m.model)  # reset underlying model
    m._impl.fit(X_tr, y_tr)
    m._cols = list(X_tr.columns)
    y_hat = pd.Series(m._impl.predict(X_te), index=y_te.index, name="y_hat")
    preds.append(y_hat)
    trues.append(y_te)

  y_pred = pd.concat(preds).sort_index()
  y_true = pd.concat(trues).sort_index()

  rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
  mae = float(mean_absolute_error(y_true, y_pred))
  with np.errstate(divide="ignore", invalid="ignore"):
    mape_arr = np.where(y_true.values != 0, np.abs((y_true.values - y_pred.values) / y_true.values), np.nan)
    mape = float(np.nanmean(mape_arr))
  r2 = float(r2_score(y_true, y_pred))
  dacc = direction_accuracy(y_true, y_pred)

  return BacktestResult(y_true=y_true, y_pred=y_pred, rmse=rmse, mae=mae, mape=mape, r2=r2, direction_acc=dacc)


__all__ = [
  "BacktestResult",
  "FeatureSpec",
  "ModelName",
  "TimeSeriesRegressor",
  "backtest_walk_forward",
  "direction_accuracy",
  "make_features",
]


if __name__ == "__main__":
  # Minimal demo: simulate returns and backtest a Ridge forecaster.
  rng = np.random.default_rng(42)
  idx = pd.date_range("2021-01-01", periods=500, freq="B")
  ret = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx, name="ret")
  tsr = TimeSeriesRegressor(model="ridge", n_lags=5)
  tsr.fit(ret)
  res = backtest_walk_forward(ret, model=tsr, n_splits=5)
  print("Backtest metrics:")
  print({
    "rmse": res.rmse,
    "mae": res.mae,
    "mape": res.mape,
    "r2": res.r2,
    "direction_acc": res.direction_acc,
  })
