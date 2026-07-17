"""Streaming-algorithm practice problems.

Implement everything below, then run:

    python3 test_problems.py                             # tests your implementations
    PRACTICE_MODULE=solutions python3 test_problems.py   # tests the reference

"screen-core" marks problems likely to be asked live in a 60-minute screen
(MovingAverage is a verbatim Datadog phone-screen question per 00-recon);
"depth" marks problems whose value is the guarantee discussion they prepare
you for. All randomness is injected (an rng parameter) so tests can be
deterministic; never reach for the global `random` module inside a solution.
"""


class MovingAverage:
  """Mean of the last `size` values of a stream, O(1) per call.

  Implementation constraint: a fixed-size circular buffer plus a running
  sum -- no deque, no re-summing the window. Before the buffer is full,
  the average is over however many values have been seen.

  API:
      __init__(size): capacity of the window.
      next(x) -> float: append x, return the mean of the last
          min(values_seen, size) values.

  Vectors (size=3): next(1) -> 1.0; next(10) -> 5.5; next(3) -> 14/3;
  next(5) -> 6.0.

  Raises:
      ValueError: in __init__ if size < 1.

  Difficulty: warmup. screen-core -- LC 346 verbatim; the follow-up is
  "why not sum a deque" (O(k) per call) and float-drift of the running
  sum over long streams.
  """

  def __init__(self, size: int) -> None:
    raise NotImplementedError

  def next(self, x: float) -> float:
    raise NotImplementedError


class StreamingMedian:
  """Exact running median via two heaps.

  Invariant: a max-heap holds the low half, a min-heap the high half,
  every low element <= every high element, and the size difference is at
  most 1 (low may hold the extra element). add is O(log n); median is
  O(1). For an even count the median is the mean of the two middle
  values (LC 295 semantics); return float either way.

  API:
      add(x): insert one value (duplicates fine, negatives fine).
      median() -> float: current median.

  Vectors: add(1), add(2) -> median 1.5; add(3) -> median 2.0.

  Raises:
      ValueError: median() before any add (message mentions "empty").

  Difficulty: core. screen-core -- LC 295, the canonical quantile
  warm-up; the depth follow-up is "now over a sliding window" (lazy
  deletion) and "now approximate in bounded space" (GK / t-digest).
  """

  def __init__(self) -> None:
    raise NotImplementedError

  def add(self, x: float) -> None:
    raise NotImplementedError

  def median(self) -> float:
    raise NotImplementedError


def sliding_window_max(values, k: int) -> list:
  """Maximum of every length-k window of values, O(n) total.

  Implementation constraint: monotonic deque of INDICES, front-to-back
  decreasing values; pop the back while <= incoming (ties evicted --
  keeping the newest of equals is what makes eviction-by-index work),
  drop the front once it leaves the window. Each index enters and leaves
  the deque once, hence O(n) for the whole stream, not O(nk).

  Vector: [1, 3, -1, -3, 5, 3, 6, 7], k=3 -> [3, 3, 5, 5, 6, 7].

  Args:
      values: sequence of comparable values (ints in tests).
      k: window length.

  Returns:
      list of len(values) - k + 1 maxima, oldest window first.

  Raises:
      ValueError: unless 1 <= k <= len(values) (message mentions
      "range").

  Difficulty: core. screen-core -- LC 239; this deque is also the SWAG
  alternative when the op is max, so expect "what if the op were sum, or
  gcd?" as the bridge to SwagWindow.
  """
  raise NotImplementedError


def reservoir_sample(stream, k: int, rng) -> list:
  """Uniform sample of k items from a stream of unknown length: Algorithm R.

  Spec EXACTLY (tests replay the rng trace, so no deviation):
    - the first k items (indices 0..k-1) fill the reservoir in order,
      consuming NO randomness;
    - for item at index i >= k: j = rng.randrange(i + 1); if j < k,
      reservoir[j] = item; one rng call per item, nothing else drawn.

  Guarantee: after n items every item is in the reservoir with
  probability exactly k/n (induction: item n enters w.p. k/n; a
  survivor's k/(n-1) is multiplied by (n-1)/n).

  If the stream has fewer than k items, return them all in arrival
  order. Return a new list; do not return an internal buffer.

  Args:
      stream: any iterable (may be a one-shot generator).
      k: reservoir capacity.
      rng: a random.Random instance (injected so tests are seeded).

  Returns:
      list of at most k sampled items.

  Raises:
      ValueError: if k < 1.

  Difficulty: core. screen-core -- "sample k lines from a huge file in
  one pass" is a standard screen prompt; the depth follow-ups are
  Algorithm L (skip counting, O(k log(n/k)) rng calls) and weighted keys
  u**(1/w) (Efraimidis-Spirakis).
  """
  raise NotImplementedError


def misra_gries(stream, k: int) -> dict:
  """Deterministic heavy-hitter candidates with k-1 counters (Misra-Gries).

  Algorithm: for each item x -- if x has a counter, increment it; else
  if fewer than k-1 counters exist, start x at 1; else decrement EVERY
  counter by 1 and delete those that hit 0 (the incoming item is
  absorbed by the decrement and gets no counter).

  Guarantee (n = stream length): every returned counter c_x satisfies
  c_x <= f_x <= c_x + n/k where f_x is the true count -- an undercount
  of at most n/k -- and every item with f_x > n/k is guaranteed to be
  present. A second pass over the stream counting only the returned
  candidates makes the answer exact. k=2 (one counter) is Boyer-Moore
  majority vote.

  Vector: misra_gries(list("abababcccc"), 3) -> {"c": 1}
  (true counts a=3, b=3, c=4; n/k = 10/3; only c exceeds it).

  Args:
      stream: iterable of hashable items.
      k: threshold parameter; at most k-1 counters are kept.

  Returns:
      dict mapping candidate -> counter value (all values >= 1).

  Raises:
      ValueError: if k < 2.

  Difficulty: core. depth -- the interview points are the guarantee
  with its constant (undercount <= n/k), why decrements are legal
  (each decrement event destroys k units of mass, so there are at most
  n/k of them), and the two-pass exact variant.
  """
  raise NotImplementedError


class CountMinSketch:
  """Count-Min sketch: fixed-size frequency table that only overestimates.

  Layout: depth rows x width counters. Row r has its own hash function;
  add(item, count) adds count to one counter per row; estimate(item)
  returns the MINIMUM of the item's depth counters. Collisions only ever
  inflate counters, so estimate(item) >= true_count(item) always, and
  with width w each row's excess is <= e * total / w with probability
  >= 1 - 1/e (Markov), so the min over d rows fails only with
  probability e**-d. Standard sizing: width = ceil(e / epsilon),
  depth = ceil(ln(1 / delta)).

  Hashing must be deterministic across processes and instances: derive
  row r's hash from hashlib.blake2b keyed by a digest of (seed, r) --
  e.g. key = blake2b(seed_bytes + row_bytes).digest() and index =
  int.from_bytes(blake2b(data, digest_size=8, key=key).digest(), "big")
  % width. Python's
  builtin hash() is banned (PYTHONHASHSEED randomizes it per process).

  API:
      __init__(width, depth, seed): all ints; seed may be negative.
      add(item, count=1): item is str or bytes; str is UTF-8 encoded
          first, so "a" and b"a" are the same key. count must be >= 1.
      estimate(item) -> int: never below the true count.
      total: attribute, sum of all added counts.

  Raises:
      ValueError: width < 1 or depth < 1 in __init__; count < 1 in add.
      TypeError: item that is neither str nor bytes.

  Difficulty: core. depth -- the discussion is the guarantee shape
  (additive error epsilon * ||f||_1, overestimate-only), conservative
  update (only raise counters to the new minimum; cuts error on skew),
  Count Sketch as the signed/unbiased contrast, and "how do I get top-k
  out of this?" (you don't -- keep a heap next to it).
  """

  def __init__(self, width: int, depth: int, seed: int) -> None:
    raise NotImplementedError

  def add(self, item, count: int = 1) -> None:
    raise NotImplementedError

  def estimate(self, item) -> int:
    raise NotImplementedError


class SwagWindow:
  """Sliding-Window Aggregation (the two-stack trick): O(1) amortized
  windowed fold for ANY associative op -- no inverse required.

  Two stacks, each entry (value, running fold):
    - back stack: push side; entry fold = op(fold below it, value),
      i.e. the fold of the back stack bottom-to-top in arrival order.
    - front stack: pop side; entry fold = op(value, fold below it),
      i.e. the fold of that entry down to the bottom, in window order.
  push appends to back. pop takes from front; if front is empty, drain
  back into front first (reversing order, recomputing folds). query is
  op(front_top_fold, back_top_fold), or whichever exists. Each element
  crosses each stack at most once: O(1) amortized, O(n) worst-case for
  a single pop. Window order is FIFO: pop returns the OLDEST element.
  Must be correct for non-commutative ops too (front fold left of back
  fold), though tests focus on max, min, add, gcd.

  Raises:
      IndexError: pop() on an empty window.
      ValueError: query() on an empty window (message mentions "empty").

  Difficulty: hard. depth -- this is "sliding window max but the op is
  gcd" and the queue-from-two-stacks idea wearing a monoid; the
  follow-ups are why sum doesn't need it (subtraction exists), why max
  has the deque shortcut, and DABA for worst-case O(1).
  """

  def __init__(self, op) -> None:
    raise NotImplementedError

  def push(self, x) -> None:
    raise NotImplementedError

  def pop(self):
    raise NotImplementedError

  def query(self):
    raise NotImplementedError
