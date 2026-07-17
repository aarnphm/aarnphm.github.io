"""Self-contained test harness for the stream-algorithms module. stdlib only.

Usage:
    python3 test_problems.py                            # tests problems.py
    PRACTICE_MODULE=solutions python3 test_problems.py  # tests solutions.py
"""

import functools
import importlib
import math
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


def approx(got, want, tol=1e-9):
  if abs(got - want) > tol:
    raise AssertionError(f'got {got!r}, want {want!r} (tol {tol})')


def raises(fn, exc_type, substr=''):
  try:
    fn()
  except exc_type as exc:
    msg = str(exc).lower()
    if substr and substr not in msg:
      raise AssertionError(
        f'wrong error message: got {msg!r}, want substring {substr!r}'
      )
  else:
    raise AssertionError(f'expected {exc_type.__name__}, got none')


# ---------------------------------------------------------- MovingAverage


def ma_vectors():
  ma = MOD.MovingAverage(3)
  approx(ma.next(1), 1.0)
  approx(ma.next(10), 5.5)
  approx(ma.next(3), 14 / 3)
  approx(ma.next(5), 6.0)


case('MovingAverage LC 346 vectors (size=3)', ma_vectors)


def ma_size_one():
  ma = MOD.MovingAverage(1)
  for v in (7, -2, 0, 100):
    approx(ma.next(v), float(v))


case('MovingAverage size=1 is identity', ma_size_one)


def ma_oracle():
  rng = random.Random(1337)
  for size in (1, 2, 7, 32):
    ma = MOD.MovingAverage(size)
    hist = []
    for _ in range(400):
      v = rng.randrange(-1000, 1000)
      hist.append(v)
      tail = hist[-size:]
      approx(ma.next(v), sum(tail) / len(tail))


case('MovingAverage vs last-k oracle (seeded, sizes 1/2/7/32)', ma_oracle)
case(
  'MovingAverage(0) -> ValueError',
  lambda: raises(lambda: MOD.MovingAverage(0), ValueError),
)

# -------------------------------------------------------- StreamingMedian


def median_vectors():
  m = MOD.StreamingMedian()
  m.add(1)
  m.add(2)
  approx(m.median(), 1.5)
  m.add(3)
  approx(m.median(), 2.0)


case('StreamingMedian LC 295 vectors', median_vectors)


def median_oracle():
  rng = random.Random(1337)
  m = MOD.StreamingMedian()
  vals = []
  for i in range(1500):
    v = rng.randrange(-100, 100)
    m.add(v)
    vals.append(v)
    s = sorted(vals)
    n = len(s)
    want = float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    approx(m.median(), want)


case(
  'StreamingMedian vs sorted-list oracle (1500 adds, duplicates)',
  median_oracle,
)


def median_descending():
  m = MOD.StreamingMedian()
  for i, want in zip(range(9, 0, -1), (9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5)):
    m.add(i)
    approx(m.median(), want)


case(
  'StreamingMedian descending inserts keep heaps balanced', median_descending
)
case(
  'StreamingMedian median() before add -> ValueError(empty)',
  lambda: raises(lambda: MOD.StreamingMedian().median(), ValueError, 'empty'),
)

# ----------------------------------------------------- sliding_window_max

case(
  'sliding_window_max LC 239 vector',
  lambda: eq(
    MOD.sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7]
  ),
)
case(
  'sliding_window_max k=1 is identity',
  lambda: eq(MOD.sliding_window_max([4, -1, 4, 0], 1), [4, -1, 4, 0]),
)
case(
  'sliding_window_max k=len is single max',
  lambda: eq(MOD.sliding_window_max([2, 9, 4], 3), [9]),
)
case(
  'sliding_window_max all-equal values',
  lambda: eq(MOD.sliding_window_max([2, 2, 2, 2], 2), [2, 2, 2]),
)


def swm_oracle():
  rng = random.Random(1337)
  for _ in range(200):
    n = rng.randrange(1, 60)
    vals = [rng.randrange(-50, 50) for _ in range(n)]
    k = rng.randrange(1, n + 1)
    want = [max(vals[i : i + k]) for i in range(n - k + 1)]
    eq(MOD.sliding_window_max(vals, k), want)


case(
  'sliding_window_max vs brute-force oracle (200 seeded trials)', swm_oracle
)
case(
  'sliding_window_max k=0 -> ValueError(range)',
  lambda: raises(
    lambda: MOD.sliding_window_max([1, 2], 0), ValueError, 'range'
  ),
)
case(
  'sliding_window_max k > len -> ValueError(range)',
  lambda: raises(
    lambda: MOD.sliding_window_max([1, 2], 3), ValueError, 'range'
  ),
)

# ------------------------------------------------------- reservoir_sample

case(
  'reservoir_sample exact Algorithm R trace (range(100), k=10, seed 1337)',
  lambda: eq(
    MOD.reservoir_sample(range(100), 10, random.Random(1337)),
    [0, 70, 54, 91, 55, 76, 87, 37, 11, 19],
  ),
)


def reservoir_spec_oracle():
  for seed in (0, 7, 42):
    rng = random.Random(seed)
    want = list(range(25))
    for i in range(25, 500):
      j = rng.randrange(i + 1)
      if j < 25:
        want[j] = i
    eq(MOD.reservoir_sample(range(500), 25, random.Random(seed)), want)


case(
  'reservoir_sample matches in-test spec replay (3 seeds)',
  reservoir_spec_oracle,
)


class _NoRandom:
  def randrange(self, n):
    raise AssertionError('rng consumed while filling the reservoir')


case(
  'reservoir_sample short stream returns all items, rng untouched',
  lambda: eq(MOD.reservoir_sample(range(3), 5, _NoRandom()), [0, 1, 2]),
)
case(
  'reservoir_sample exactly k items, rng untouched',
  lambda: eq(MOD.reservoir_sample(range(4), 4, _NoRandom()), [0, 1, 2, 3]),
)
case(
  'reservoir_sample consumes a one-shot generator',
  lambda: eq(
    sorted(MOD.reservoir_sample((x for x in range(6)), 6, _NoRandom())),
    [0, 1, 2, 3, 4, 5],
  ),
)


def reservoir_frequency():
  n, k, trials = 20, 5, 3000
  hits = [0] * n
  for t in range(trials):
    sample = MOD.reservoir_sample(range(n), k, random.Random(t))
    eq(len(sample), k)
    eq(len(set(sample)), k)
    for x in sample:
      hits[x] += 1
  for x, h in enumerate(hits):
    freq = h / trials
    if abs(freq - k / n) > 0.03:
      raise AssertionError(f'item {x} frequency {freq:.4f}, want ~{k / n}')


case(
  'reservoir_sample inclusion frequency ~ k/n (3000 seeded trials)',
  reservoir_frequency,
)
case(
  'reservoir_sample k=0 -> ValueError',
  lambda: raises(
    lambda: MOD.reservoir_sample(range(5), 0, random.Random(1)), ValueError
  ),
)

# ------------------------------------------------------------ misra_gries

case(
  'misra_gries docstring vector (abababcccc, k=3)',
  lambda: eq(MOD.misra_gries(list('abababcccc'), 3), {'c': 1}),
)
case(
  'misra_gries k=2 is Boyer-Moore majority',
  lambda: eq(MOD.misra_gries([1, 1, 2, 1, 3, 1, 1], 2), {1: 3}),
)
case('misra_gries empty stream -> {}', lambda: eq(MOD.misra_gries([], 4), {}))
case(
  'misra_gries single hot key exact when it fits',
  lambda: eq(MOD.misra_gries(['x'] * 9, 5), {'x': 9}),
)


def mg_guarantee():
  rng = random.Random(1337)
  pop = list(range(30))
  weights = [1 / (i + 1) for i in pop]
  for _ in range(50):
    n = rng.randrange(200, 2000)
    stream = rng.choices(pop, weights, k=n)
    k = rng.randrange(2, 12)
    got = MOD.misra_gries(stream, k)
    true = {}
    for x in stream:
      true[x] = true.get(x, 0) + 1
    if len(got) > k - 1:
      raise AssertionError(f'{len(got)} counters, cap is {k - 1}')
    for x, c in got.items():
      if not (c <= true.get(x, 0) <= c + n / k):
        raise AssertionError(
          f'counter {c} vs true {true.get(x, 0)}, n/k {n / k}'
        )
    for x, f in true.items():
      if f > n / k and x not in got:
        raise AssertionError(f'heavy hitter {x} (count {f} > {n / k}) missing')


case(
  'misra_gries guarantee vs exact counts (50 seeded skewed streams)',
  mg_guarantee,
)
case(
  'misra_gries k=1 -> ValueError',
  lambda: raises(lambda: MOD.misra_gries([1, 2], 1), ValueError),
)

# --------------------------------------------------------- CountMinSketch


def cms_workload():
  cms = MOD.CountMinSketch(200, 8, 42)
  rng = random.Random(1337)
  true = {}
  keys = [f'k{i}' for i in range(300)]
  for key in keys:
    c = rng.randrange(1, 50)
    true[key] = true.get(key, 0) + c
    cms.add(key, c)
  for _ in range(2000):
    key = rng.choice(keys[:30])
    true[key] += 1
    cms.add(key)
  return cms, true, keys


def cms_never_underestimates():
  cms, true, keys = cms_workload()
  for key in keys:
    est = cms.estimate(key)
    if est < true[key]:
      raise AssertionError(f'{key}: estimate {est} < true {true[key]}')


case(
  'CountMinSketch estimate >= true count on every key',
  cms_never_underestimates,
)


def cms_error_bound():
  cms, true, keys = cms_workload()
  bound = math.ceil(math.e * cms.total / 200)
  for key in keys:
    excess = cms.estimate(key) - true[key]
    if excess > bound:
      raise AssertionError(
        f'{key}: excess {excess} > ceil(e*total/width) {bound}'
      )
  for i in range(50):
    est = cms.estimate(f'absent{i}')
    if not 0 <= est <= bound:
      raise AssertionError(f'absent{i}: estimate {est} outside [0, {bound}]')


case(
  'CountMinSketch excess <= ceil(e*total/width), absent keys included',
  cms_error_bound,
)


def cms_exact_when_wide():
  cms = MOD.CountMinSketch(4096, 8, 7)
  for i in range(10):
    cms.add(f'x{i}', i + 1)
  eq(cms.total, sum(range(1, 11)))
  for i in range(10):
    eq(cms.estimate(f'x{i}'), i + 1)


case(
  'CountMinSketch exact on 10 keys in a 4096-wide sketch', cms_exact_when_wide
)


def cms_deterministic_across_instances():
  a = MOD.CountMinSketch(64, 4, 99)
  b = MOD.CountMinSketch(64, 4, 99)
  for item, count in [('alpha', 3), ('beta', 5), (b'gamma', 2)]:
    a.add(item, count)
    b.add(item, count)
  for item in ('alpha', 'beta', b'gamma', 'absent'):
    eq(a.estimate(item), b.estimate(item))


case(
  'CountMinSketch same (width, depth, seed) -> identical estimates',
  cms_deterministic_across_instances,
)


def cms_str_bytes_alias():
  cms = MOD.CountMinSketch(64, 4, 0)
  cms.add('a', 7)
  eq(cms.estimate(b'a'), 7)
  cms.add(b'a', 2)
  eq(cms.estimate('a'), 9)


case(
  'CountMinSketch "a" and b"a" are the same key (UTF-8)', cms_str_bytes_alias
)
case(
  'CountMinSketch negative seed accepted',
  lambda: MOD.CountMinSketch(16, 2, -5).add('x'),
)
case(
  'CountMinSketch(0, 4, 1) -> ValueError',
  lambda: raises(lambda: MOD.CountMinSketch(0, 4, 1), ValueError),
)
case(
  'CountMinSketch(16, 0, 1) -> ValueError',
  lambda: raises(lambda: MOD.CountMinSketch(16, 0, 1), ValueError),
)
case(
  'CountMinSketch add count=0 -> ValueError',
  lambda: raises(lambda: MOD.CountMinSketch(16, 2, 1).add('x', 0), ValueError),
)
case(
  'CountMinSketch add non-str/bytes -> TypeError',
  lambda: raises(lambda: MOD.CountMinSketch(16, 2, 1).add(42), TypeError),
)

# ------------------------------------------------------------- SwagWindow


def swag_fifo():
  w = MOD.SwagWindow(max)
  for v in (3, 1, 4, 1, 5):
    w.push(v)
  eq(w.query(), 5)
  eq(w.pop(), 3)
  eq(w.pop(), 1)
  eq(w.query(), 5)
  eq(w.pop(), 4)
  eq(w.pop(), 1)
  eq(w.query(), 5)
  eq(w.pop(), 5)


case('SwagWindow pop returns oldest (FIFO), query tracks max', swag_fifo)


def swag_non_commutative():
  w = MOD.SwagWindow(lambda a, b: a + b)
  for s in ('a', 'b', 'c'):
    w.push(s)
  eq(w.query(), 'abc')
  eq(w.pop(), 'a')
  eq(w.query(), 'bc')
  w.push('d')
  eq(w.query(), 'bcd')


case(
  'SwagWindow concat preserves window order (non-commutative op)',
  swag_non_commutative,
)


def swag_oracle():
  rng = random.Random(1337)
  ops = [
    ('max', max),
    ('min', min),
    ('add', lambda a, b: a + b),
    ('gcd', math.gcd),
  ]
  for name, op in ops:
    w = MOD.SwagWindow(op)
    window = []
    for _ in range(3000):
      r = rng.random()
      if r < 0.55 or not window:
        v = rng.randrange(1, 1000)
        w.push(v)
        window.append(v)
      elif r < 0.8:
        got, want = w.pop(), window.pop(0)
        if got != want:
          raise AssertionError(f'[{name}] pop got {got}, want {want}')
      if window:
        want = functools.reduce(op, window)
        got = w.query()
        if got != want:
          raise AssertionError(f'[{name}] query got {got}, want {want}')


case(
  'SwagWindow vs reduce oracle (max/min/add/gcd, 3000 steps each)', swag_oracle
)


def swag_drain_refill():
  w = MOD.SwagWindow(math.gcd)
  for v in (12, 18, 30):
    w.push(v)
  eq(w.query(), 6)
  eq(w.pop(), 12)
  eq(w.pop(), 18)
  eq(w.query(), 30)
  eq(w.pop(), 30)
  w.push(8)
  w.push(20)
  eq(w.query(), 4)


case('SwagWindow drain to empty then refill', swag_drain_refill)
case(
  'SwagWindow pop on empty -> IndexError',
  lambda: raises(lambda: MOD.SwagWindow(max).pop(), IndexError),
)
case(
  'SwagWindow query on empty -> ValueError(empty)',
  lambda: raises(lambda: MOD.SwagWindow(max).query(), ValueError, 'empty'),
)

# ---------------------------------------------------------------- summary

total = PASSED + FAILED
print(f'{PASSED}/{total} passed')
sys.exit(1 if FAILED else 0)
