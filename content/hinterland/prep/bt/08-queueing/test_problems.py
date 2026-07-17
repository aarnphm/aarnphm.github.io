"""Self-contained test harness for the queueing module. stdlib only.

Usage:
    python3 test_problems.py                            # tests problems.py
    PRACTICE_MODULE=solutions python3 test_problems.py  # tests solutions.py

All randomness is seeded (random.Random with fixed seeds); no wall clocks.
"""

import importlib
import os
import random
import sys

MOD = importlib.import_module(os.environ.get('PRACTICE_MODULE', 'problems'))

PASSED = 0
FAILED = 0


def case(name, fn):
  global PASSED, FAILED
  try:
    fn()
  except Exception as exc:
    FAILED += 1
    print(f'FAIL {name}: {type(exc).__name__}: {exc}')
  else:
    PASSED += 1
    print(f'PASS {name}')


def eq(got, want):
  if got != want:
    raise AssertionError(f'got {got!r}, want {want!r}')


def approx(got, want, rel=1e-9):
  tol = rel * max(abs(got), abs(want), 1e-12)
  if abs(got - want) > tol:
    raise AssertionError(f'got {got!r}, want {want!r} (rel {rel})')


def raises(fn, substr):
  try:
    fn()
  except ValueError as exc:
    msg = str(exc).lower()
    if substr not in msg:
      raise AssertionError(
        f'wrong error class: got {msg!r}, want substring {substr!r}'
      )
  else:
    raise AssertionError(
      f'expected ValueError containing {substr!r}, got none'
    )


# ---------------------------------------------------------------- token bucket


def tb_burst_then_refill():
  tb = MOD.TokenBucket(1.0, 2.0)
  eq(tb.allow(0.0), True)
  eq(tb.allow(0.0), True)
  eq(tb.allow(0.0), False)
  eq(tb.allow(0.5), False)
  eq(tb.allow(1.0), True)
  eq(tb.allow(3.5), True)
  eq(tb.allow(3.5, cost=2), False)
  eq(tb.allow(10.0, cost=2), True)


case(
  'token bucket burst, drain, lazy refill, multi-cost', tb_burst_then_refill
)


def tb_fractional_refill():
  tb = MOD.TokenBucket(4.0, 1.0)
  eq(tb.allow(0.0), True)
  eq(tb.allow(0.125), False)
  eq(tb.allow(0.25), True)
  eq(tb.allow(0.25), False)


case(
  'token bucket fractional accrual (binary-exact rates)', tb_fractional_refill
)


def tb_refill_caps_at_capacity():
  tb = MOD.TokenBucket(100.0, 3.0)
  eq(tb.allow(0.0), True)
  eq(tb.allow(1e9), True)
  eq(tb.allow(1e9), True)
  eq(tb.allow(1e9), True)
  eq(tb.allow(1e9), False)


case('token bucket idle refill caps at capacity', tb_refill_caps_at_capacity)


def tb_cost_above_capacity_never_admits():
  tb = MOD.TokenBucket(100.0, 5.0)
  eq(tb.allow(0.0, cost=6), False)
  eq(tb.allow(1e6, cost=6), False)
  eq(tb.allow(1e6, cost=5), True)


case(
  'token bucket cost > capacity never admitted',
  tb_cost_above_capacity_never_admits,
)


def tb_backwards_clock_clamped():
  tb = MOD.TokenBucket(1.0, 1.0)
  eq(tb.allow(0.0), True)
  eq(tb.allow(-100.0), False)
  eq(tb.allow(0.5), False)
  eq(tb.allow(1.0), True)


case(
  'token bucket clamps a backwards clock (no refill, no rewind)',
  tb_backwards_clock_clamped,
)

case(
  'token bucket rate <= 0 -> positive error',
  lambda: raises(lambda: MOD.TokenBucket(0.0, 1.0), 'positive'),
)
case(
  'token bucket capacity <= 0 -> positive error',
  lambda: raises(lambda: MOD.TokenBucket(1.0, -1.0), 'positive'),
)
case(
  'token bucket cost <= 0 -> positive error',
  lambda: raises(
    lambda: MOD.TokenBucket(1.0, 1.0).allow(0.0, cost=0), 'positive'
  ),
)


def tb_admission_guarantee():
  rng = random.Random(1337)
  rate, cap = 5.0, 20.0
  tb = MOD.TokenBucket(rate, cap)
  now = 0.0
  granted = []
  for _ in range(2000):
    now += rng.random() * 0.2
    cost = rng.choice([1, 1, 1, 2, 5])
    if tb.allow(now, cost):
      granted.append((now, cost))
  if not granted:
    raise AssertionError('workload admitted nothing')
  for i in range(len(granted)):
    total = 0.0
    for j in range(i, len(granted)):
      total += granted[j][1]
      span = granted[j][0] - granted[i][0]
      if total > cap + rate * span + 1e-9:
        raise AssertionError(
          f'admitted {total} in span {span}: violates capacity + rate*T'
        )


case(
  'token bucket guarantee: any-interval admitted <= cap + rate*T (seeded)',
  tb_admission_guarantee,
)

# ------------------------------------------------------- sliding window counter


def swc_weighted_vectors():
  swc = MOD.SlidingWindowCounter(2, 1.0)
  eq(swc.allow(0.0), True)
  eq(swc.allow(0.0), True)
  eq(swc.allow(0.0), False)
  eq(swc.allow(1.0), False)
  eq(swc.allow(1.5), True)
  eq(swc.allow(1.75), False)
  eq(swc.allow(2.5), True)


case('sliding window counter weighted-estimate vectors', swc_weighted_vectors)


def swc_window_skip_resets():
  swc = MOD.SlidingWindowCounter(1, 1.0)
  eq(swc.allow(0.0), True)
  eq(swc.allow(0.25), False)
  eq(swc.allow(5.5), True)


case(
  'sliding window counter resets after a 2+ window gap', swc_window_skip_resets
)


def swc_backwards_clock_clamped():
  swc = MOD.SlidingWindowCounter(1, 1.0)
  eq(swc.allow(5.0), True)
  eq(swc.allow(3.0), False)


case(
  'sliding window counter clamps a backwards clock',
  swc_backwards_clock_clamped,
)

case(
  'sliding window counter limit < 1 -> positive error',
  lambda: raises(lambda: MOD.SlidingWindowCounter(0, 1.0), 'positive'),
)
case(
  'sliding window counter window <= 0 -> positive error',
  lambda: raises(lambda: MOD.SlidingWindowCounter(1, 0.0), 'positive'),
)


def swc_bounds_against_log_oracle():
  rng = random.Random(1337)
  limit, window = 10, 1.0
  swc = MOD.SlidingWindowCounter(limit, window)
  now = 0.0
  admitted = []
  for _ in range(5000):
    now += rng.random() * 0.05
    if swc.allow(now):
      admitted.append(now)
  if not admitted:
    raise AssertionError('workload admitted nothing')
  per_aligned = {}
  for t in admitted:
    k = int(t // window)
    per_aligned[k] = per_aligned.get(k, 0) + 1
  if max(per_aligned.values()) > limit:
    raise AssertionError(f'aligned window exceeded limit: {per_aligned}')
  lo = 0
  for hi, t in enumerate(admitted):
    while admitted[lo] <= t - window:
      lo += 1
    trailing = hi - lo + 1
    if trailing > 2 * limit:
      raise AssertionError(
        f'trailing window admitted {trailing} > 2*limit at t={t}'
      )


case(
  'sliding window counter: aligned <= limit, trailing <= 2*limit (seeded)',
  swc_bounds_against_log_oracle,
)

# ---------------------------------------------------------------- mm1 formulas


def mm1_vectors():
  m = MOD.mm1_metrics(8.0, 10.0)
  approx(m['rho'], 0.8)
  approx(m['L'], 4.0)
  approx(m['W'], 0.5)
  approx(m['Lq'], 3.2)
  approx(m['Wq'], 0.4)


case('mm1_metrics(8, 10) closed-form vector', mm1_vectors)


def mm1_little_identities():
  for lam, mu in ((1.0, 2.0), (8.0, 10.0), (0.5, 0.51), (99.0, 100.0)):
    m = MOD.mm1_metrics(lam, mu)
    approx(m['L'], lam * m['W'])
    approx(m['Lq'], lam * m['Wq'])
    approx(m['L'], m['Lq'] + m['rho'])


case(
  'mm1 Little identities: L = lam*W, Lq = lam*Wq, L = Lq + rho',
  mm1_little_identities,
)


def mm1_hockey_stick():
  for rho, mult in ((0.5, 2), (0.8, 5), (0.9, 10), (0.95, 20), (0.99, 100)):
    m = MOD.mm1_metrics(rho, 1.0)
    approx(m['W'], float(mult))


case(
  'mm1 hockey stick: W/E[S] = 1/(1-rho) at the five canonical rhos',
  mm1_hockey_stick,
)

case(
  'mm1 rho >= 1 -> unstable error',
  lambda: raises(lambda: MOD.mm1_metrics(10.0, 10.0), 'unstable'),
)
case(
  'mm1 lam > mu -> unstable error',
  lambda: raises(lambda: MOD.mm1_metrics(11.0, 10.0), 'unstable'),
)
case(
  'mm1 lam <= 0 -> positive error',
  lambda: raises(lambda: MOD.mm1_metrics(0.0, 10.0), 'positive'),
)
case(
  'mm1 mu <= 0 -> positive error',
  lambda: raises(lambda: MOD.mm1_metrics(1.0, -1.0), 'positive'),
)


def pk_exponential_reproduces_mm1():
  for lam, mu in ((1.0, 2.0), (8.0, 10.0), (0.9, 1.0)):
    mean_s = 1.0 / mu
    approx(
      MOD.pk_wq(lam, mean_s, mean_s * mean_s), MOD.mm1_metrics(lam, mu)['Wq']
    )


case(
  'pk_wq with exponential service (var = mean^2) == M/M/1 Wq',
  pk_exponential_reproduces_mm1,
)


def pk_deterministic_is_half_mm1():
  for lam, mu in ((1.0, 2.0), (8.0, 10.0)):
    approx(MOD.pk_wq(lam, 1.0 / mu, 0.0), MOD.mm1_metrics(lam, mu)['Wq'] / 2)


case(
  'pk_wq with deterministic service == half M/M/1 Wq (M/D/1)',
  pk_deterministic_is_half_mm1,
)


def pk_variance_is_first_class():
  base = MOD.pk_wq(5.0, 0.1, 0.01)
  double = MOD.pk_wq(5.0, 0.1, 0.03)
  if not double > base:
    raise AssertionError('Wq must increase with service variance')
  approx(double / base, (0.03 + 0.01) / (0.01 + 0.01))


case(
  'pk_wq scales linearly in E[S^2] at fixed rho', pk_variance_is_first_class
)

case(
  'pk_wq rho >= 1 -> unstable error',
  lambda: raises(lambda: MOD.pk_wq(10.0, 0.1, 0.0), 'unstable'),
)
case(
  'pk_wq negative variance -> non-negative error',
  lambda: raises(lambda: MOD.pk_wq(1.0, 0.1, -0.1), 'non-negative'),
)
case(
  'pk_wq lam <= 0 -> positive error',
  lambda: raises(lambda: MOD.pk_wq(-1.0, 0.1, 0.0), 'positive'),
)


def kingman_reduces_to_pk():
  for lam, mu, cs2 in ((8.0, 10.0, 0.0), (8.0, 10.0, 1.0), (1.0, 2.0, 4.0)):
    mean_s = 1.0 / mu
    approx(
      MOD.kingman_wq(lam, mu, 1.0, cs2),
      MOD.pk_wq(lam, mean_s, cs2 * mean_s * mean_s),
    )


case('kingman_wq at ca2 = 1 == pk_wq (exact for M/G/1)', kingman_reduces_to_pk)


def kingman_dd1_never_queues():
  eq(MOD.kingman_wq(8.0, 10.0, 0.0, 0.0), 0.0)


case('kingman_wq at ca2 = cs2 = 0 == 0 (D/D/1)', kingman_dd1_never_queues)


def kingman_mm1_vector():
  approx(MOD.kingman_wq(8.0, 10.0, 1.0, 1.0), MOD.mm1_metrics(8.0, 10.0)['Wq'])


case('kingman_wq at ca2 = cs2 = 1 == M/M/1 Wq', kingman_mm1_vector)

case(
  'kingman_wq rho >= 1 -> unstable error',
  lambda: raises(lambda: MOD.kingman_wq(10.0, 10.0, 1.0, 1.0), 'unstable'),
)
case(
  'kingman_wq negative ca2 -> non-negative error',
  lambda: raises(lambda: MOD.kingman_wq(1.0, 2.0, -1.0, 1.0), 'non-negative'),
)

# ---------------------------------------------------------------- fifo sim


def fifo_hand_trace():
  got = MOD.simulate_fifo([0.0, 1.0, 2.0, 10.0], [3.0, 3.0, 3.0, 1.0])
  eq(
    got, [(0.0, 0.0, 3.0), (3.0, 2.0, 5.0), (6.0, 4.0, 7.0), (10.0, 0.0, 1.0)]
  )


case('fifo hand-computed trace (busy period then idle gap)', fifo_hand_trace)

case('fifo empty input -> []', lambda: eq(MOD.simulate_fifo([], []), []))
case(
  'fifo single job -> (arrival, 0, service)',
  lambda: eq(MOD.simulate_fifo([2.0], [0.5]), [(2.0, 0.0, 0.5)]),
)
case(
  'fifo zero service time is legal',
  lambda: eq(
    MOD.simulate_fifo([0.0, 0.0], [0.0, 1.0]),
    [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
  ),
)
case(
  'fifo length mismatch -> mismatch error',
  lambda: raises(lambda: MOD.simulate_fifo([0.0], []), 'mismatch'),
)
case(
  'fifo decreasing arrivals -> non-decreasing error',
  lambda: raises(
    lambda: MOD.simulate_fifo([1.0, 0.5], [1.0, 1.0]), 'non-decreasing'
  ),
)
case(
  'fifo negative service -> non-negative error',
  lambda: raises(lambda: MOD.simulate_fifo([0.0], [-1.0]), 'non-negative'),
)


def fifo_lindley_oracle():
  rng = random.Random(1337)
  n = 500
  arrivals, t = [], 0.0
  for _ in range(n):
    t += rng.expovariate(1.0)
    arrivals.append(t)
  services = [rng.expovariate(1.25) for _ in range(n)]
  got = MOD.simulate_fifo(arrivals, services)
  eq(len(got), n)
  w = 0.0
  for i in range(n):
    if i:
      w = max(0.0, w + services[i - 1] - (arrivals[i] - arrivals[i - 1]))
    approx(got[i][1], w)
    approx(got[i][0], arrivals[i] + w)
    approx(got[i][2], w + services[i])


case(
  'fifo waits match the Lindley recursion oracle (seeded M/M/1)',
  fifo_lindley_oracle,
)


def fifo_littles_law():
  rng = random.Random(2024)
  n = 20000
  lam, mu = 0.8, 1.0
  arrivals, t = [], 0.0
  for _ in range(n):
    t += rng.expovariate(lam)
    arrivals.append(t)
  services = [rng.expovariate(mu) for _ in range(n)]
  res = MOD.simulate_fifo(arrivals, services)
  departures = [arrivals[i] + res[i][2] for i in range(n)]
  events = sorted([(a, 1) for a in arrivals] + [(d_, -1) for d_ in departures])
  area, in_system, prev_t = 0.0, 0, events[0][0]
  for tt, delta in events:
    area += in_system * (tt - prev_t)
    in_system += delta
    prev_t = tt
  approx(area, sum(r[2] for r in res), rel=1e-6)
  horizon = departures[-1]
  lam_hat = n / horizon
  w_bar = sum(r[2] for r in res) / n
  approx(area / horizon, lam_hat * w_bar, rel=1e-6)
  mean_wait = sum(r[1] for r in res) / n
  wq_theory = MOD.mm1_metrics(lam, mu)['Wq']
  if abs(mean_wait - wq_theory) / wq_theory > 0.2:
    raise AssertionError(
      f'simulated mean wait {mean_wait:.3f} too far from M/M/1 Wq {wq_theory:.3f}'
    )


case(
  'fifo Little check: event-integrated area == sum of sojourns; Wq near theory',
  fifo_littles_law,
)

# ---------------------------------------------------------------- bounded queue


def bq_reject():
  q = MOD.BoundedQueue(2, 'reject')
  eq(q.offer('a'), True)
  eq(q.offer('b'), True)
  eq(q.offer('c'), False)
  eq(len(q), 2)
  eq(q.poll(), 'a')
  eq(q.poll(), 'b')
  eq(q.poll(), None)
  eq((q.accepted, q.dropped), (2, 1))


case('bounded queue reject: refuses incoming, FIFO order, counters', bq_reject)


def bq_drop_oldest():
  q = MOD.BoundedQueue(2, 'drop_oldest')
  eq(q.offer('a'), True)
  eq(q.offer('b'), True)
  eq(q.offer('c'), True)
  eq(q.poll(), 'b')
  eq(q.poll(), 'c')
  eq(q.poll(), None)
  eq((q.accepted, q.dropped), (3, 1))


case(
  'bounded queue drop_oldest: ring-buffer eviction of the head', bq_drop_oldest
)


def bq_drop_newest():
  q = MOD.BoundedQueue(2, 'drop_newest')
  eq(q.offer('a'), True)
  eq(q.offer('b'), True)
  eq(q.offer('c'), True)
  eq(q.poll(), 'a')
  eq(q.poll(), 'c')
  eq(q.poll(), None)
  eq((q.accepted, q.dropped), (3, 1))


case(
  'bounded queue drop_newest: evicts the tail, keeps oldest work',
  bq_drop_newest,
)


def bq_poll_reopens_capacity():
  q = MOD.BoundedQueue(1, 'reject')
  eq(q.offer(1), True)
  eq(q.offer(2), False)
  eq(q.poll(), 1)
  eq(q.offer(3), True)
  eq(q.poll(), 3)
  eq((q.accepted, q.dropped), (2, 1))


case(
  'bounded queue poll frees a slot for the next offer',
  bq_poll_reopens_capacity,
)

case(
  'bounded queue capacity < 1 -> error',
  lambda: raises(lambda: MOD.BoundedQueue(0, 'reject'), '>= 1'),
)
case(
  'bounded queue unknown policy -> error',
  lambda: raises(lambda: MOD.BoundedQueue(1, 'drop_random'), 'unknown'),
)


def bq_seeded_oracle():
  class Oracle:
    def __init__(self, capacity, policy):
      self.capacity, self.policy = capacity, policy
      self.items, self.accepted, self.dropped = [], 0, 0

    def offer(self, item):
      if len(self.items) < self.capacity:
        self.items.append(item)
        self.accepted += 1
        return True
      if self.policy == 'reject':
        self.dropped += 1
        return False
      victim = 0 if self.policy == 'drop_oldest' else len(self.items) - 1
      del self.items[victim]
      self.dropped += 1
      self.items.append(item)
      self.accepted += 1
      return True

    def poll(self):
      return self.items.pop(0) if self.items else None

  rng = random.Random(1337)
  for policy in ('reject', 'drop_oldest', 'drop_newest'):
    got, want = MOD.BoundedQueue(3, policy), Oracle(3, policy)
    for i in range(600):
      if rng.random() < 0.6:
        eq(got.offer(i), want.offer(i))
      else:
        eq(got.poll(), want.poll())
      eq(len(got), len(want.items))
    eq((got.accepted, got.dropped), (want.accepted, want.dropped))


case(
  'bounded queue seeded offer/poll interleaving matches list oracle (all policies)',
  bq_seeded_oracle,
)

# ---------------------------------------------------------------- two choices


class ScriptedRng:
  def __init__(self, values):
    self.values = list(values)

  def randrange(self, n):
    v = self.values.pop(0)
    if not 0 <= v < n:
      raise AssertionError(f'scripted value {v} out of range({n})')
    return v


def tc_scripted_trace():
  rng = ScriptedRng([0, 1, 0, 1, 2, 0, 1, 1])
  eq(MOD.two_choices(4, 3, 2, rng), 2)
  eq(rng.values, [])


case(
  'two_choices exact trace: ties go to first sample, d calls per ball',
  tc_scripted_trace,
)


def tc_d1_scripted():
  eq(MOD.two_choices(3, 4, 1, ScriptedRng([2, 2, 0])), 2)


case('two_choices d=1 places blindly (scripted)', tc_d1_scripted)

case(
  'two_choices fixed seed 1337 exact value (d=2)',
  lambda: eq(MOD.two_choices(50, 10, 2, random.Random(1337)), 6),
)
case(
  'two_choices fixed seed 1337 exact value (d=1)',
  lambda: eq(MOD.two_choices(50, 10, 1, random.Random(1337)), 9),
)
case(
  'two_choices single bin holds everything',
  lambda: eq(MOD.two_choices(37, 1, 2, random.Random(1)), 37),
)
case(
  'two_choices zero balls -> 0',
  lambda: eq(MOD.two_choices(0, 5, 2, random.Random(1)), 0),
)

case(
  'two_choices n_balls < 0 -> error',
  lambda: raises(
    lambda: MOD.two_choices(-1, 5, 2, random.Random(1)), 'non-negative'
  ),
)
case(
  'two_choices n_bins < 1 -> error',
  lambda: raises(lambda: MOD.two_choices(1, 0, 2, random.Random(1)), '>= 1'),
)
case(
  'two_choices d < 1 -> error',
  lambda: raises(lambda: MOD.two_choices(1, 5, 0, random.Random(1)), '>= 1'),
)


def tc_power_of_two():
  for seed in (1, 2, 3):
    m1 = MOD.two_choices(20000, 20000, 1, random.Random(seed))
    m2 = MOD.two_choices(20000, 20000, 2, random.Random(seed))
    if not m2 < m1:
      raise AssertionError(
        f'seed {seed}: d=2 max load {m2} not < d=1 max load {m1}'
      )


case(
  'two_choices d=2 max load strictly below d=1 (n=20000, seeds 1..3)',
  tc_power_of_two,
)


def tc_conservation():
  rng = random.Random(7)
  n_balls, n_bins = 500, 50
  peak = MOD.two_choices(n_balls, n_bins, 2, rng)
  if not n_balls / n_bins <= peak <= n_balls:
    raise AssertionError(f'max load {peak} outside [mean, n_balls]')


case('two_choices max load bounded below by mean load', tc_conservation)

# ---------------------------------------------------------------- summary

total = PASSED + FAILED
print(f'{PASSED}/{total} passed')
sys.exit(1 if FAILED else 0)
