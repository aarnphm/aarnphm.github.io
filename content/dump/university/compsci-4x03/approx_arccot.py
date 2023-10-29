import math

def find_nth_terms(x: float, eps: float = 1e-12):
  n = 0
  term = x
  while abs(term) >= eps:
    n += 1
    term = math.pow(-1, n) * math.pow(x, 2*n+1) / (2*n+1)
  return n
