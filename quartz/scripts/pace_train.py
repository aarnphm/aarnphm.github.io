# /// script
# requires-python = '>=3.11'
# dependencies = [
#     'jax',
#     'jaxlib',
#     'optax',
#     'numpy',
#     'safetensors',
# ]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import save_file

jax.config.update('jax_enable_x64', False)

SCHEMA_VERSION = 2
SPORTS = ('swim', 'bike', 'run')
T_REF_S = 3600.0
RIEGEL_K = {'swim': 1.03, 'bike': 1.05, 'run': 1.06}
FEATURE_NAMES = (
  'sport_swim',
  'sport_bike',
  'sport_run',
  'distance_km',
  'elevation_m',
  'temp_c',
  'wind_kph',
  'ctl',
  'atl',
  'tsb',
  'sport_ctl',
  'hrv',
  'rhr',
  'readiness',
  'sleep_s',
  'temp_dev_c',
  'weight_kg',
  'vthr',
  'hr_max',
  'effort',
)
DF = len(FEATURE_NAMES)
DIN = DF * 2
LN_EPS = 1e-5
VAR_EPS = 1e-6
MASKABLE = (5, 6, 11, 12, 13, 14, 15, 16, 18, 19)


def prev_iso(iso: str) -> str:
  y, m, d = (int(x) for x in iso.split('-'))
  return (date(y, m, d) - timedelta(days=1)).isoformat()


def load_feed(path: Path) -> tuple[dict, dict[str, dict], list[dict]]:
  meta: dict = {}
  days: dict[str, dict] = {}
  acts: list[dict] = []
  for line in path.read_text().splitlines():
    line = line.strip()
    if not line:
      continue
    row = json.loads(line)
    kind = row.get('kind')
    if kind == 'meta':
      meta = row
    elif kind == 'day':
      days[row['date']] = row
    elif kind == 'activity':
      acts.append(row)
  return meta, days, acts


def num(row: dict, key: str) -> float | None:
  v = row.get(key)
  return float(v) if isinstance(v, (int, float)) else None


def day_state(row: dict) -> dict:
  return {
    k: num(row, k)
    for k in (
      'ctl',
      'atl',
      'tsb',
      'swimCtl',
      'bikeCtl',
      'runCtl',
      'hrv',
      'rhr',
      'readiness',
      'sleepDurationS',
      'tempDeviationC',
      'weightKg',
    )
  }


def feature_vector(day: dict, act: dict, vthr: float, hr_max: float | None):
  sport = act.get('sport')
  raw = np.zeros(DF, dtype=np.float64)
  pres = np.ones(DF, dtype=np.float64)

  def put(i: int, v: float | None) -> None:
    if v is None or not np.isfinite(v):
      pres[i] = 0.0
      return
    raw[i] = v

  sport_ctl = day.get(f'{sport}Ctl') if sport in SPORTS else None
  raw[0] = 1.0 if sport == 'swim' else 0.0
  raw[1] = 1.0 if sport == 'bike' else 0.0
  raw[2] = 1.0 if sport == 'run' else 0.0
  raw[3] = num(act, 'distanceKm') or 0.0
  raw[4] = num(act, 'elevationM') or 0.0
  put(5, num(act, 'avgTemp'))
  put(6, num(act, 'windKph'))
  raw[7] = day.get('ctl') or 0.0
  raw[8] = day.get('atl') or 0.0
  raw[9] = day.get('tsb') or 0.0
  raw[10] = sport_ctl or 0.0
  put(11, day.get('hrv'))
  put(12, day.get('rhr'))
  put(13, day.get('readiness'))
  put(14, day.get('sleepDurationS'))
  put(15, day.get('tempDeviationC'))
  put(16, day.get('weightKg'))
  raw[17] = vthr
  put(18, hr_max)
  avg_hr = num(act, 'avgHr')
  put(19, avg_hr / hr_max if avg_hr is not None and hr_max else None)
  return raw, pres


def build_dataset(meta: dict, days: dict, acts: list[dict], target: str):
  vthr_by = {t['sport']: float(t['vThr']) for t in meta.get('thresholds', [])}
  athlete = meta.get('athlete') or {}
  hr_max = athlete.get('hrMaxEst')
  hr_max = float(hr_max) if isinstance(hr_max, (int, float)) else None
  meta_today = meta.get('today')
  today = (
    meta_today
    if isinstance(meta_today, str)
    else datetime.now().strftime('%Y-%m-%d')
  )
  rows_raw, rows_pres, ys, dates, sports = [], [], [], [], []
  vbbs, vreals = [], []
  for act in acts:
    sport = act.get('sport')
    if sport not in SPORTS:
      continue
    if str(act.get('date', '')) > today:
      continue
    vthr = vthr_by.get(sport, 0.0)
    if vthr <= 0:
      continue
    v_bb = math.nan
    if target == 'pace':
      vgap = num(act, 'vGap')
      dist = num(act, 'distanceKm')
      if vgap is None or dist is None or dist <= 0:
        continue
      if not 0.3 <= vgap / vthr <= 2.5:
        continue
      d_ref = vthr * T_REF_S / 1000.0
      v_bb = vthr * (dist / d_ref) ** (1.0 - RIEGEL_K[sport])
      y = math.log(vgap / v_bb)
    else:
      avg_hr = num(act, 'avgHr')
      if avg_hr is None or not hr_max:
        continue
      y = avg_hr / hr_max
      if not 0.3 <= y <= 2.5:
        continue
    iso = act['date']
    day = days.get(prev_iso(iso)) or days.get(iso) or {}
    raw, pres = feature_vector(day, act, vthr, hr_max)
    if target == 'hr':
      raw[19] = 0.0
      pres[19] = 0.0
    else:
      raw[3] = 0.0
      pres[3] = 0.0
    rows_raw.append(raw)
    rows_pres.append(pres)
    ys.append(y)
    dates.append(iso)
    sports.append(sport)
    vbbs.append(v_bb)
    vreals.append(num(act, 'vGap') if target == 'pace' else math.nan)
  return (
    np.array(rows_raw),
    np.array(rows_pres),
    np.array(ys, dtype=np.float64),
    dates,
    sports,
    np.array(vbbs, dtype=np.float64),
    np.array(vreals, dtype=np.float64),
  )


def temporal_split(dates: list[str], val_frac: float) -> np.ndarray:
  order = np.argsort(np.array(dates))
  n_val = max(1, round(len(dates) * val_frac))
  val_idx = set(order[-n_val:].tolist())
  return np.array([i in val_idx for i in range(len(dates))])


def assemble(raw, pres, impute, mu, sigma):
  filled = np.where(pres > 0, raw, impute)
  x = np.concatenate([filled, pres], axis=1)
  return (x - mu) / sigma


def init_params(key, arch, din):
  keys = jax.random.split(key, 4)
  scale = 0.1
  if arch['layers'] == 0:
    return {
      'w1': scale * jax.random.normal(keys[0], (din, 2)),
      'b1': jnp.zeros((2,)),
    }
  h = arch['hidden']
  return {
    'w1': scale * jax.random.normal(keys[0], (din, h)),
    'b1': jnp.zeros((h,)),
    'g': jnp.ones((h,)),
    'beta': jnp.zeros((h,)),
    'w2': scale * jax.random.normal(keys[1], (h, 2)),
    'b2': jnp.zeros((2,)),
  }


def forward(params, x, arch):
  if arch['layers'] == 0:
    out = x @ params['w1'] + params['b1']
  else:
    h = x @ params['w1'] + params['b1']
    m = jnp.mean(h, axis=-1, keepdims=True)
    v = jnp.var(h, axis=-1, keepdims=True)
    h = (h - m) / jnp.sqrt(v + LN_EPS) * params['g'] + params['beta']
    h = jax.nn.gelu(h, approximate=True)
    out = h @ params['w2'] + params['b2']
  mu = out[..., 0]
  var = jax.nn.softplus(out[..., 1]) + VAR_EPS
  return mu, var


def nll_loss(params, x, y, arch, l2):
  mu, var = forward(params, x, arch)
  nll = 0.5 * (jnp.log(var) + (y - mu) ** 2 / var)
  reg = l2 * sum(jnp.sum(p**2) for p in jax.tree.leaves(params))
  return jnp.mean(nll) + reg


def train_member(xtr, ytr, xva, yva, arch, l2, seed, steps):
  key = jax.random.PRNGKey(seed)
  params = init_params(key, arch, xtr.shape[1])
  opt = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adamw(1e-2, weight_decay=0.0)
  )
  state = opt.init(params)
  loss_grad = jax.jit(
    jax.value_and_grad(lambda p, x, y: nll_loss(p, x, y, arch, l2))
  )

  @jax.jit
  def step(params, state, x, y):
    _, grads = loss_grad(params, x, y)
    updates, state = opt.update(grads, state, params)
    return optax.apply_updates(params, updates), state

  best, best_va = params, float('inf')
  bad = 0
  for _ in range(steps):
    params, state = step(params, state, xtr, ytr)
    va = float(nll_loss(params, xva, yva, arch, 0.0))
    if va < best_va - 1e-5:
      best_va, best, bad = va, params, 0
    else:
      bad += 1
      if bad > 50:
        break
  return best, best_va


def ensemble_predict(members, x, arch):
  mus, vars_ = [], []
  for p in members:
    mu, var = forward(p, x, arch)
    mus.append(np.array(mu))
    vars_.append(np.array(var))
  mus = np.stack(mus)
  vars_ = np.stack(vars_)
  mu_bar = mus.mean(0)
  var_bar = vars_.mean(0) + mus.var(0)
  return mu_bar, var_bar


def pack_tensors(members, arch):
  tensors = {}
  for m, p in enumerate(members):
    for k, v in p.items():
      name = {
        'w1': 'fc1.weight',
        'b1': 'fc1.bias',
        'g': 'ln.weight',
        'beta': 'ln.bias',
        'w2': 'fc2.weight',
        'b2': 'fc2.bias',
      }[k]
      arr = np.array(v, dtype=np.float32)
      if name in ('fc1.weight', 'fc2.weight'):
        arr = arr.T.copy()
      tensors[f'member.{m}.{name}'] = arr
  return tensors


def git_commit() -> str:
  try:
    return subprocess.run(
      ['git', 'rev-parse', '--short', 'HEAD'],
      capture_output=True,
      text=True,
      check=True,
    ).stdout.strip()
  except Exception:
    return 'unknown'


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--feed', required=True)
  ap.add_argument('--archive', default='.models-archive')
  ap.add_argument('--target', choices=('pace', 'hr'), default='pace')
  ap.add_argument('--members', type=int, default=5)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--val-frac', type=float, default=0.2)
  ap.add_argument('--steps', type=int, default=2000)
  args = ap.parse_args()

  feed_path = Path(args.feed)
  meta, days, acts = load_feed(feed_path)
  raw, pres, y, dates, sports, v_bb_all, v_real_all = build_dataset(
    meta, days, acts, args.target
  )
  n = len(y)
  if n < 24:
    print(f'pace_train: only {n} samples for {args.target}; skipping')
    return

  is_val = temporal_split(dates, args.val_frac)
  tr, va = ~is_val, is_val
  impute = np.zeros(DF)
  for j in range(DF):
    obs = raw[tr][pres[tr][:, j] > 0, j]
    impute[j] = float(obs.mean()) if obs.size else 0.0

  filled_tr = np.where(pres[tr] > 0, raw[tr], impute)
  xin_tr = np.concatenate([filled_tr, pres[tr]], axis=1)
  mu = xin_tr.mean(0)
  sigma = xin_tr.std(0)
  sigma[sigma < 1e-6] = 1.0

  x_all = assemble(raw, pres, impute, mu, sigma)
  xtr, ytr = jnp.array(x_all[tr]), jnp.array(y[tr])
  xva, yva = jnp.array(x_all[va]), jnp.array(y[va])

  best_arch, best_members, best_score = None, None, float('inf')
  for arch in ({'layers': 0}, {'layers': 1, 'hidden': 16}):
    l2 = 0.05 if arch['layers'] == 0 else 0.1
    members = []
    for m in range(args.members):
      rng = np.random.default_rng(args.seed + m)
      boot = rng.integers(0, xtr.shape[0], xtr.shape[0])
      p, _ = train_member(
        xtr[boot], ytr[boot], xva, yva, arch, l2, args.seed + m, args.steps
      )
      members.append(p)
    mu_sel, _ = ensemble_predict(members, jnp.array(x_all[va]), arch)
    mae_sel = float(np.mean(np.abs(mu_sel - y[va])))
    if mae_sel < best_score:
      best_arch, best_members, best_score = arch, members, mae_sel

  arch = best_arch
  arch.setdefault('hidden', 0)
  arch['activation'] = 'gelu_tanh'
  arch['layerNormEps'] = LN_EPS
  arch['members'] = args.members
  mu_va, var_va = ensemble_predict(best_members, jnp.array(x_all[va]), arch)
  sd = np.sqrt(var_va)
  cov80 = float(np.mean(np.abs(mu_va - y[va]) <= 1.2816 * sd))
  cov90 = float(np.mean(np.abs(mu_va - y[va]) <= 1.6449 * sd))
  nll_va = float(
    np.mean(0.5 * (np.log(var_va) + (mu_va - y[va]) ** 2 / var_va))
  )

  backbone = None
  if args.target == 'pace':
    vbb_va = v_bb_all[va]
    vreal_va = v_real_all[va]
    vhat_va = vbb_va * np.exp(mu_va)
    mae = float(np.mean(np.abs(vhat_va - vreal_va)))
    rmse = float(np.sqrt(np.mean((vhat_va - vreal_va) ** 2)))
    base_mae = float(np.mean(np.abs(vbb_va - vreal_va)))
    val_space = 'velocity'
    backbone = {
      'kind': 'riegel',
      'tRefS': T_REF_S,
      'riegelK': dict(RIEGEL_K),
      'distanceIndex': FEATURE_NAMES.index('distance_km'),
      'sportIndices': {'swim': 0, 'bike': 1, 'run': 2},
    }
  else:
    mae = float(np.mean(np.abs(mu_va - y[va])))
    rmse = float(np.sqrt(np.mean((mu_va - y[va]) ** 2)))
    sport_mean = {
      s: float(np.mean(y[tr][np.array(sports)[tr] == s]))
      for s in set(np.array(sports)[tr])
    }
    base_pred = np.array([
      sport_mean.get(s, float(np.mean(y[tr]))) for s in np.array(sports)[va]
    ])
    base_mae = float(np.mean(np.abs(base_pred - y[va])))
    val_space = 'ratio'
  beats_baseline = mae < base_mae

  scale_feature = 'vthr' if args.target == 'pace' else 'hr_max'
  golden = []
  for gi in list(np.where(va)[0])[:8]:
    gx = assemble(raw[gi : gi + 1], pres[gi : gi + 1], impute, mu, sigma)
    gmu, gvar = ensemble_predict(best_members, jnp.array(gx), arch)
    golden.append({
      'date': dates[gi],
      'raw': [float(v) for v in raw[gi]],
      'presence': [float(v) for v in pres[gi]],
      'mu': float(gmu[0]),
      'sigma': float(np.sqrt(gvar[0])),
    })

  ts = datetime.now().strftime('%Y%m%d-%H%M%S')
  fam_dir = Path(args.archive) / args.target
  existing = [
    int(p.name.rsplit('-v', 1)[1])
    for p in fam_dir.glob('*-v*')
    if p.name.rsplit('-v', 1)[-1].isdigit()
  ]
  version = (max(existing) + 1) if existing else 1
  vdir = fam_dir / f'{ts}-v{version}'
  vdir.mkdir(parents=True, exist_ok=True)

  tensors = pack_tensors(best_members, arch)
  st_path = vdir / 'model.safetensors'
  save_file(tensors, str(st_path))
  sha = hashlib.sha256(st_path.read_bytes()).hexdigest()
  dataset_hash = hashlib.sha256(feed_path.read_bytes()).hexdigest()

  manifest = {
    'schemaVersion': SCHEMA_VERSION,
    'version': version,
    'createdAt': ts,
    'target': args.target,
    'gitCommit': git_commit(),
    'datasetHash': f'sha256:{dataset_hash}',
    'featureNames': list(FEATURE_NAMES),
    'dFeatures': DF,
    'dIn': DIN,
    'standardize': {'mu': mu.tolist(), 'sigma': sigma.tolist()},
    'impute': impute.tolist(),
    'output': {
      'muIndex': 0,
      'varIndex': 1,
      'varTransform': 'softplus',
      'varEps': VAR_EPS,
      'scaleFeature': scale_feature,
      'backbone': backbone,
    },
    'arch': arch,
    'sha256': sha,
    'val': {
      'mae': mae,
      'rmse': rmse,
      'nll': nll_va,
      'coverage80': cov80,
      'coverage90': cov90,
      'nVal': int(va.sum()),
      'nTrain': int(tr.sum()),
      'baselineMae': base_mae,
      'beatsBaseline': beats_baseline,
      'valSpace': val_space,
      'valFromDate': sorted([dates[i] for i in np.where(va)[0]])[0],
    },
    'golden': golden,
  }
  (vdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
  print(
    f'pace_train[{args.target}] v{version}: n={n} arch={arch["layers"]}L '
    f'mae={mae:.4f}({val_space}) baseline={base_mae:.4f} beats={beats_baseline} '
    f'cov90={cov90:.2f} -> {vdir}'
  )


if __name__ == '__main__':
  main()
