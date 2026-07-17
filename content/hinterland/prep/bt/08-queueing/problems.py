"""Queueing practice problems: rate limiters, formulas, simulation.

Implement everything below, then run:

    python3 test_problems.py                             # tests your implementations
    PRACTICE_MODULE=solutions python3 test_problems.py   # tests the reference

No wall clocks anywhere: every time-dependent object takes `now` as a float
seconds argument, every random algorithm takes an injected rng. "screen-core"
marks problems likely to be asked live in a 60-minute systems screen; "depth"
marks problems whose value is the follow-up discussion they prepare you for.
"""


class TokenBucket:
  """Token-bucket rate limiter with lazy refill and an injected clock.

  Semantics: the bucket holds at most `capacity` tokens and refills
  continuously at `rate` tokens per second. It starts full; the first
  allow() call anchors the clock. allow(now, cost) refills lazily —
  tokens = min(capacity, tokens + (now - last) * rate) — then admits and
  deducts iff cost <= tokens. Admission is all-or-nothing; a denied call
  consumes nothing. Guarantee to state in an interview: over any interval
  of length T, admitted cost <= capacity + rate * T (burst up to
  capacity, sustained rate `rate`).

  API:
    __init__(rate, capacity): floats, both must be > 0 else ValueError
        (message mentions "positive").
    allow(now, cost=1) -> bool: `now` is float seconds from an injected
        monotonic clock (never read a wall clock). cost must be > 0 else
        ValueError. If `now` is earlier than the latest `now` seen, clamp:
        no refill, no rewind of the internal timestamp; the call is
        otherwise processed normally.

  Edge cases: cost > capacity can never be admitted, even after infinite
  idle — return False, do not loop or raise. Fractional tokens and costs
  are legal (floats throughout). State is two floats plus a timestamp;
  that is the whole point (contrast: a sliding log is one timestamp per
  admitted request).

  Difficulty: core. screen-core -- the single most-asked rate-limiter
  coding question; the lazy-refill trick (no timers, no threads) is what
  they are checking for.
  """

  def __init__(self, rate: float, capacity: float) -> None:
    raise NotImplementedError

  def allow(self, now: float, cost: float = 1) -> bool:
    raise NotImplementedError


class SlidingWindowCounter:
  """Sliding-window rate limiter using the weighted-previous-window estimate.

  Keep one counter per aligned fixed window of length `window` seconds
  (window k covers [k*window, (k+1)*window)). On allow(now) with now in
  window k at offset e = now - k*window, estimate the trailing-window
  request count as

      est = prev_count * (1 - e/window) + curr_count

  where prev_count is the previous window's total. Admit iff
  est + 1 <= limit, and increment curr_count on admission only.

  Window bookkeeping: when now enters window k+1, the current counter
  becomes the previous one; when now jumps two or more windows ahead,
  both counters reset to 0. If now moves backwards, clamp: a now before
  the current window's start is treated as sitting at that start (weight
  1.0); counters never rewind.

  The approximation, stated for the interviewer: it assumes the previous
  window's requests were uniformly spread. If they were actually bunched
  at the end of the previous window, the weighted estimate undercounts
  and a true trailing window can see up to ~2*limit admitted; bunched at
  the start, it overcounts and rejects traffic a sliding log would admit.
  Aligned windows are exact: at most `limit` admissions per fixed window,
  always (est >= curr_count). The trade: two integers per key versus one
  timestamp per admitted request for the exact log — Cloudflare shipped
  this and measured 0.003 percent wrong decisions over 400M requests.

  API:
    __init__(limit, window): limit int >= 1, window float > 0, else
        ValueError (message mentions "positive").
    allow(now) -> bool: now is float seconds, injected.

  Difficulty: core. screen-core -- the standard follow-up to token bucket
  ("now do it with bounded memory per key and no burst-at-boundary bug").
  """

  def __init__(self, limit: int, window: float) -> None:
    raise NotImplementedError

  def allow(self, now: float) -> bool:
    raise NotImplementedError


def mm1_metrics(lam: float, mu: float) -> dict:
  """Closed-form M/M/1 steady-state metrics.

  With arrival rate lam (Poisson) and service rate mu (exponential),
  utilization rho = lam/mu, return a dict with float values:

    rho: lam/mu
    L:   rho/(1-rho)        mean number in system
    W:   1/(mu-lam)         mean sojourn (wait + service)
    Lq:  rho**2/(1-rho)     mean number in queue
    Wq:  rho/(mu-lam)       mean wait before service

  The Little identities hold exactly: L = lam*W, Lq = lam*Wq, and
  L = Lq + rho.

  Args:
      lam: arrival rate, must be > 0.
      mu: service rate, must be > 0.

  Raises:
      ValueError: lam <= 0 or mu <= 0 (message mentions "positive");
      rho >= 1 (message mentions "unstable").

  Difficulty: warmup. screen-core -- the formulas behind every capacity
  follow-up; know W = E[S]/(1-rho) cold.
  """
  raise NotImplementedError


def pk_wq(lam: float, mean_s: float, var_s: float) -> float:
  """Pollaczek-Khinchine mean queueing delay for M/G/1.

  Wq = lam * E[S^2] / (2 * (1 - rho)) with E[S^2] = var_s + mean_s**2
  and rho = lam * mean_s. Equivalently (rho/(1-rho)) * ((1+Cs^2)/2) *
  mean_s where Cs^2 = var_s/mean_s**2. Service variance is a first-class
  input: exponential service (var_s = mean_s**2) must reproduce the
  M/M/1 Wq exactly; deterministic service (var_s = 0) gives exactly half
  of it.

  Args:
      lam: arrival rate > 0.
      mean_s: mean service time > 0.
      var_s: service-time variance >= 0.

  Raises:
      ValueError: lam <= 0 or mean_s <= 0 ("positive"); var_s < 0
      ("non-negative"); rho >= 1 ("unstable").

  Difficulty: warmup. screen-core -- the formula that explains why one
  10 s request among 10 ms requests wrecks everyone behind it.
  """
  raise NotImplementedError


def kingman_wq(lam: float, mu: float, ca2: float, cs2: float) -> float:
  """Kingman's VUT approximation for G/G/1 mean queueing delay.

  Wq ~= (rho/(1-rho)) * ((ca2 + cs2)/2) * (1/mu), rho = lam/mu, where
  ca2 and cs2 are the squared coefficients of variation of interarrival
  and service times. Variability times utilization times time. With
  ca2 = 1 it coincides with Pollaczek-Khinchine (exact for M/G/1); with
  ca2 = cs2 = 0 it returns 0.0 (D/D/1 never queues).

  Args:
      lam: arrival rate > 0.
      mu: service rate > 0.
      ca2: squared CV of interarrivals, >= 0.
      cs2: squared CV of service, >= 0.

  Raises:
      ValueError: lam <= 0 or mu <= 0 ("positive"); ca2 < 0 or cs2 < 0
      ("non-negative"); rho >= 1 ("unstable").

  Difficulty: warmup. screen-core -- the back-of-envelope workhorse for
  "what happens to latency if arrivals get bursty".
  """
  raise NotImplementedError


def simulate_fifo(arrivals: list, services: list) -> list:
  """Deterministic single-server FIFO queue simulation (Lindley recursion).

  Job i arrives at arrivals[i] (absolute seconds, non-decreasing) and
  needs services[i] seconds. One server, FIFO, no preemption. Job i
  starts at max(arrivals[i], finish of job i-1). Return, per job, the
  tuple (start, wait, sojourn) where wait = start - arrival and sojourn
  = wait + service. Equivalent wait-only form to know for the screen:
  W_0 = 0, W_i = max(0, W_{i-1} + S_{i-1} - (A_i - A_{i-1})).

  Args:
      arrivals: list of floats, non-decreasing.
      services: list of floats, each >= 0, same length as arrivals.

  Returns:
      list of (start, wait, sojourn) float tuples, one per job, in
      arrival order. Empty input -> [].

  Raises:
      ValueError: length mismatch ("mismatch"), arrivals decreasing
      ("non-decreasing"), negative service ("non-negative").

  Difficulty: core. core -- the standard "simulate it and check the
  formula" question; no event heap, single pass, O(n).
  """
  raise NotImplementedError


class BoundedQueue:
  """Bounded FIFO queue with an explicit overflow policy.

  A queue of at most `capacity` items. offer(item) enqueues when there
  is room. When full, the policy decides who loses:

    'reject':      the incoming item is refused; offer returns False.
        Production analogue: load shedding at admission — HTTP 429,
        RabbitMQ reject-publish, a bounded executor's abort policy. The
        producer is told, so backpressure can propagate. Router tail
        drop is this same policy minus the signal: the arriving packet
        silently dies and the sender only learns via missing ACKs.
    'drop_oldest': the head (oldest resident) is evicted, the incoming
        item is enqueued; offer returns True. Production analogue: ring
        buffer — dmesg, flight recorders, metrics pipelines — anywhere
        fresh data outranks stale data.
    'drop_newest': the tail (newest resident) is evicted, the incoming
        item is enqueued; offer returns True (Akka's dropTail).
        Production analogue: keep the oldest committed work — the jobs
        that have waited longest keep their place and the speculative
        fresh burst churns in the last slot.

  Counters: `accepted` counts items that entered the queue (including
  ones later evicted); `dropped` counts losses (a rejected incoming item
  or an evicted resident). Every offered item increments exactly one of
  accepted/dropped except drop_oldest/drop_newest overflow, which
  increments both (one item in, one item out).

  API:
    __init__(capacity, policy): capacity int >= 1 else ValueError
        (">= 1"); policy one of the three strings else ValueError
        ("unknown").
    offer(item) -> bool: True iff the incoming item was enqueued.
    poll() -> item | None: dequeue the head; None when empty.
    __len__() -> current item count.

  Difficulty: core. depth -- the coding vehicle for the backpressure
  discussion: bounded + policy is a decision, unbounded is bufferbloat.
  """

  def __init__(self, capacity: int, policy: str) -> None:
    raise NotImplementedError

  def offer(self, item) -> bool:
    raise NotImplementedError

  def poll(self):
    raise NotImplementedError

  def __len__(self) -> int:
    raise NotImplementedError


def two_choices(n_balls: int, n_bins: int, d: int, rng) -> int:
  """Balls-into-bins with d choices; returns the max bin load.

  For each of n_balls balls, sample d candidate bins with replacement —
  exactly d calls to rng.randrange(n_bins), in order — and place the
  ball in the least-loaded candidate, breaking ties in favor of the
  earliest-sampled candidate (strictly-less comparison against the
  running best). Return max load over all bins after all balls.

  The theory this demonstrates: with n balls in n bins, d = 1 gives max
  load Theta(log n / log log n) w.h.p.; d = 2 gives log log n / log 2 +
  Theta(1) — an exponential drop for one extra probe, and d = 3 buys
  almost nothing more (the log log n term divides by log d). This is the
  power of two choices behind P2C load balancers (Envoy, Finagle).

  Args:
      n_balls: int >= 0 (0 -> returns 0).
      n_bins: int >= 1.
      d: int >= 1 (d = 1 is plain random placement).
      rng: injected randomness with a randrange(n) method (a seeded
          random.Random in tests — never construct randomness inside).

  Returns:
      int, the maximum bin load.

  Raises:
      ValueError: n_balls < 0 ("non-negative"), n_bins < 1 (">= 1"),
      d < 1 (">= 1").

  Difficulty: core. depth -- the simulation interviewers ask to see if
  you can keep an rng call sequence deterministic and testable.
  """
  raise NotImplementedError
