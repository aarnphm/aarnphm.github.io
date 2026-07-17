"""Reference implementations for the queueing problems. stdlib only."""

import math
from collections import deque


class TokenBucket:
  def __init__(self, rate: float, capacity: float) -> None:
    if rate <= 0:
      raise ValueError('rate must be positive')
    if capacity <= 0:
      raise ValueError('capacity must be positive')
    self.rate = float(rate)
    self.capacity = float(capacity)
    self.tokens = float(capacity)
    self.last = None

  def allow(self, now: float, cost: float = 1) -> bool:
    if cost <= 0:
      raise ValueError('cost must be positive')
    if self.last is None:
      self.last = now
    elif now > self.last:
      self.tokens = min(
        self.capacity, self.tokens + (now - self.last) * self.rate
      )
      self.last = now
    if cost <= self.tokens:
      self.tokens -= cost
      return True
    return False


class SlidingWindowCounter:
  def __init__(self, limit: int, window: float) -> None:
    if limit < 1:
      raise ValueError('limit must be positive')
    if window <= 0:
      raise ValueError('window must be positive')
    self.limit = limit
    self.window = float(window)
    self.idx = None
    self.curr = 0
    self.prev = 0

  def allow(self, now: float) -> bool:
    idx = math.floor(now / self.window)
    if self.idx is None:
      self.idx = idx
    elif idx == self.idx + 1:
      self.prev, self.curr, self.idx = self.curr, 0, idx
    elif idx > self.idx + 1:
      self.prev, self.curr, self.idx = 0, 0, idx
    elapsed = now - self.idx * self.window
    weight = 1.0 - min(max(elapsed / self.window, 0.0), 1.0)
    if self.prev * weight + self.curr + 1 <= self.limit:
      self.curr += 1
      return True
    return False


def mm1_metrics(lam: float, mu: float) -> dict:
  if lam <= 0 or mu <= 0:
    raise ValueError('rates must be positive')
  rho = lam / mu
  if rho >= 1:
    raise ValueError('unstable: rho >= 1')
  return {
    'rho': rho,
    'L': rho / (1 - rho),
    'W': 1 / (mu - lam),
    'Lq': rho * rho / (1 - rho),
    'Wq': rho / (mu - lam),
  }


def pk_wq(lam: float, mean_s: float, var_s: float) -> float:
  if lam <= 0 or mean_s <= 0:
    raise ValueError('rates must be positive')
  if var_s < 0:
    raise ValueError('variance must be non-negative')
  rho = lam * mean_s
  if rho >= 1:
    raise ValueError('unstable: rho >= 1')
  es2 = var_s + mean_s * mean_s
  return lam * es2 / (2 * (1 - rho))


def kingman_wq(lam: float, mu: float, ca2: float, cs2: float) -> float:
  if lam <= 0 or mu <= 0:
    raise ValueError('rates must be positive')
  if ca2 < 0 or cs2 < 0:
    raise ValueError('squared CVs must be non-negative')
  rho = lam / mu
  if rho >= 1:
    raise ValueError('unstable: rho >= 1')
  return (rho / (1 - rho)) * ((ca2 + cs2) / 2) * (1 / mu)


def simulate_fifo(arrivals: list, services: list) -> list:
  if len(arrivals) != len(services):
    raise ValueError('length mismatch between arrivals and services')
  out = []
  free_at = float('-inf')
  prev = float('-inf')
  for a, s in zip(arrivals, services):
    if a < prev:
      raise ValueError('arrivals must be non-decreasing')
    if s < 0:
      raise ValueError('service times must be non-negative')
    start = a if a > free_at else free_at
    wait = start - a
    out.append((start, wait, wait + s))
    free_at = start + s
    prev = a
  return out


class BoundedQueue:
  def __init__(self, capacity: int, policy: str) -> None:
    if capacity < 1:
      raise ValueError('capacity must be >= 1')
    if policy not in ('reject', 'drop_oldest', 'drop_newest'):
      raise ValueError('unknown policy')
    self.capacity = capacity
    self.policy = policy
    self.items = deque()
    self.accepted = 0
    self.dropped = 0

  def offer(self, item) -> bool:
    if len(self.items) < self.capacity:
      self.items.append(item)
      self.accepted += 1
      return True
    if self.policy == 'reject':
      self.dropped += 1
      return False
    if self.policy == 'drop_oldest':
      self.items.popleft()
    else:
      self.items.pop()
    self.dropped += 1
    self.items.append(item)
    self.accepted += 1
    return True

  def poll(self):
    if not self.items:
      return None
    return self.items.popleft()

  def __len__(self) -> int:
    return len(self.items)


def two_choices(n_balls: int, n_bins: int, d: int, rng) -> int:
  if n_balls < 0:
    raise ValueError('n_balls must be non-negative')
  if n_bins < 1:
    raise ValueError('n_bins must be >= 1')
  if d < 1:
    raise ValueError('d must be >= 1')
  loads = [0] * n_bins
  for _ in range(n_balls):
    best = rng.randrange(n_bins)
    for _ in range(d - 1):
      cand = rng.randrange(n_bins)
      if loads[cand] < loads[best]:
        best = cand
    loads[best] += 1
  return max(loads)
