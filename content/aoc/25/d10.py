from __future__ import annotations

import re
from collections import deque
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Sequence

def parse_line(s: str) -> tuple[str, list[list[int]], list[int]]:
  # [pattern] (btn1) (btn2) ... {targets}
  pattern = re.search(r'\[([.#]+)\]', s).group(1)
  buttons = [list(map(int, m.group(1).split(','))) for m in re.finditer(r'\(([0-9,]+)\)', s)]
  targets = list(map(int, re.search(r'\{([0-9,]+)\}', s).group(1).split(',')))
  return pattern, buttons, targets

# part 1: BFS over XOR state space
def pattern_to_mask(pat: str) -> int:
  """convert .##. to bitmask where # = 1"""
  return sum(1 << i for i, c in enumerate(pat) if c == '#')

def button_to_mask(btn: list[int]) -> int:
  """convert button indices to XOR mask"""
  mask = 0
  for i in btn:
    mask ^= 1 << i
  return mask

def min_press_lights(target: int, masks: list[int]) -> int:
  """BFS to find minimum presses to reach target state from 0"""
  if target == 0:
    return 0
  seen = {0}
  q = deque([(0, 0)])
  while q:
    state, dist = q.popleft()
    for m in masks:
      nxt = state ^ m
      if nxt == target:
        return dist + 1
      if nxt not in seen:
        seen.add(nxt)
        q.append((nxt, dist + 1))
  raise RuntimeError('unreachable')

# part 2: integer linear programming via gaussian elimination
def gauss_jordan(a: list[list[Fraction]], b: list[Fraction]) -> tuple[list[list[Fraction]], list[Fraction], list[int]]:
  """reduce to row echelon form, return (matrix, rhs, pivot_columns)"""
  rows, cols = len(a), len(a[0]) if a else 0
  mat = [row[:] for row in a]
  rhs = b[:]
  pivots = []
  r = 0
  for c in range(cols):
    # find pivot
    pivot_row = None
    for i in range(r, rows):
      if mat[i][c] != 0:
        pivot_row = i
        break
    if pivot_row is None:
      continue
    # swap
    mat[r], mat[pivot_row] = mat[pivot_row], mat[r]
    rhs[r], rhs[pivot_row] = rhs[pivot_row], rhs[r]
    # normalize
    scale = mat[r][c]
    mat[r] = [mat[r][j] / scale for j in range(cols)]
    rhs[r] /= scale
    # eliminate
    for i in range(rows):
      if i != r and mat[i][c] != 0:
        factor = mat[i][c]
        mat[i] = [mat[i][j] - factor * mat[r][j] for j in range(cols)]
        rhs[i] -= factor * rhs[r]
    pivots.append(c)
    r += 1
    if r >= rows:
      break
  return mat, rhs, pivots

def solve_square(mat: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
  """solve square system via back substitution after gaussian elimination"""
  n = len(mat)
  mat2, rhs2, _ = gauss_jordan(mat, rhs)
  # check for singularity
  for i in range(n):
    if all(mat2[i][j] == 0 for j in range(n)):
      return None
  # back substitute (already in RREF)
  sol = [Fraction(0)] * n
  for i in range(n - 1, -1, -1):
    lead = next((j for j in range(n) if mat2[i][j] != 0), None)
    if lead is None:
      return None
    rest = sum(mat2[i][j] * sol[j] for j in range(lead + 1, n))
    sol[lead] = rhs2[i] - rest
  return sol

def bounds_for_free(constraints: list[tuple[list[Fraction], Fraction]], f: int) -> tuple[list[Fraction], list[Fraction]]:
  """find min/max bounds for free variables by vertex enumeration"""
  from itertools import combinations

  mins = [None] * f
  maxs = [Fraction(0)] * f
  for subset in combinations(range(len(constraints)), f):
    mat = [constraints[i][0] for i in subset]
    rhs = [constraints[i][1] for i in subset]
    sol = solve_square(mat, rhs)
    if sol is None:
      continue
    # check feasibility
    if all(sum(c[j] * sol[j] for j in range(f)) <= b for c, b in constraints):
      for j in range(f):
        if mins[j] is None or sol[j] < mins[j]:
          mins[j] = sol[j]
        if sol[j] > maxs[j]:
          maxs[j] = sol[j]
  return [m if m is not None else Fraction(0) for m in mins], maxs

def min_press_jolts(buttons: Sequence[Sequence[int]], targets: Sequence[int]) -> int:
  # minimize sum(x) subject to Ax = b, x >= 0, x \in \mathcal{R}
  import math

  m, n = len(targets), len(buttons)
  # build matrix: a[i][j] = 1 if button j affects counter i
  a = [[Fraction(1 if i in btn else 0) for btn in buttons] for i in range(m)]
  b = [Fraction(t) for t in targets]
  a_red, b_red, pivots = gauss_jordan(a, b)
  free = [c for c in range(n) if c not in pivots]
  f_count = len(free)

  if f_count == 0:
    # unique solution
    pivot_vals = [b_red[r] for r in range(len(pivots))]
    if all(v >= 0 and v.denominator == 1 for v in pivot_vals):
      return sum(int(v) for v in pivot_vals)
    raise RuntimeError('no feasible solution')

  # build constraints from reduced system
  # pivot_r = b_red[r] - sum(a_red[r][f] * x_f for f in free)
  # need pivot_r >= 0 => sum(a_red[r][f] * x_f) <= b_red[r]
  row_info = [(b_red[r], [a_red[r][f] for f in free]) for r in range(len(pivots))]
  constraints = [(coeff, b_val) for b_val, coeff in row_info]
  # x_f >= 0 => -x_f <= 0
  for j in range(f_count):
    neg = [Fraction(-1) if k == j else Fraction(0) for k in range(f_count)]
    constraints.append((neg, Fraction(0)))

  mins, maxs = bounds_for_free(constraints, f_count)
  bounds = [(max(0, math.ceil(float(mi))), math.floor(float(mx))) for mi, mx in zip(mins, maxs)]

  def search(idx: int, vals: list[int]) -> int:
    if idx == f_count:
      pivot_vals = [b_val - sum(coeff[j] * vals[j] for j in range(f_count)) for b_val, coeff in row_info]
      if all(v >= 0 and v.denominator == 1 for v in pivot_vals):
        return sum(vals) + sum(int(v) for v in pivot_vals)
      return 10**18
    lo, hi = bounds[idx]
    best = 10**18
    for v in range(lo, hi + 1):
      best = min(best, search(idx + 1, vals + [v]))
    return best

  return search(0, [])

def main():
  with open('d10.txt') as f:
    lines = [ln.strip() for ln in f if ln.strip()]

  p1 = 0
  for ln in lines:
    pat, btns, _ = parse_line(ln)
    target = pattern_to_mask(pat)
    masks = [button_to_mask(b) for b in btns]
    p1 += min_press_lights(target, masks)

  p2 = 0
  for ln in lines:
    _, btns, targets = parse_line(ln)
    p2 += min_press_jolts(btns, targets)

  print(f'p1: {p1}')
  print(f'p2: {p2}')

if __name__ == '__main__':
  main()
