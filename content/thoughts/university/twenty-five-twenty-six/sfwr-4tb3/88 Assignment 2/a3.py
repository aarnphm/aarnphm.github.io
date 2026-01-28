#!/usr/bin/env python3
"""
A3: Context-Sensitive Grammar for {a^n b^{2n} c^n | n ≥ 1}

The grammar must be monotonic (context-sensitive): |α| ≤ |β| for all α → β.

Strategy:
1. Generate structure: S → a B B C | a S B B C
   - Each 'a' comes with two B's and one C
   - 'a' is already terminal, B and C are nonterminals

2. Permute to sort: C B → B C
   - Move all B's to the left of all C's

3. Convert nonterminals to terminals (left-to-right):
   - a B → a b  (start conversion at boundary)
   - b B → b b  (propagate through B's)
   - b C → b c  (convert first C)
   - c C → c c  (propagate through C's)

All productions are monotonic (length-preserving or increasing).
"""


class Grammar:
  def __init__(self, T: set[str], N: set[str], P: set[tuple[str, str]], S: str):
    self.T, self.N, self.P, self.S = T, N, P, S

  def derivable(self, ω: str, log=False, stats=False) -> bool:
    dd, d, ω = set(), {self.S}, ω.strip()
    if log:
      print('    ', self.S)
    while d:
      if stats:
        print('# added derivations:', len(d))
      if log:
        print()
      dd.update(d)
      d = set()
      for π in sorted(dd, key=len):
        for σ, τ in self.P:
          i = π.find(σ, 0)
          while i != -1:
            χ = π[0:i] + τ + π[i + len(σ) :]
            χ = χ.replace('  ', ' ')
            if (χ not in dd) and (χ not in d):
              if χ.strip() == ω:
                return True
              elif len(χ.strip()) <= len(ω):
                if log:
                  print('    ', π, '⇒', χ)
                d.add(χ)
            i = π.find(σ, i + 1)
    return False


# Grammar for {a^n b^{2n} c^n | n ≥ 1}
# Terminals: a, b, c
# Nonterminals: S, B, C
# Start symbol: S
T = {'a', 'b', 'c'}
N = {'S', 'B', 'C'}
P = {
  ('S', 'a B B C'),  # base case: n=1 generates a B B C
  ('S', 'a S B B C'),  # recursive: each level adds a, B B, C
  ('C B', 'B C'),  # permutation: move B's left of C's
  ('a B', 'a b'),  # conversion: terminal 'a' triggers B→b
  ('b B', 'b b'),  # propagate: b triggers next B→b
  ('b C', 'b c'),  # conversion: last b triggers C→c
  ('c C', 'c c'),  # propagate: c triggers next C→c
}
S = 'S'

G = Grammar(T, N, P, S)


def main():
  print('=' * 70)
  print('A3: Context-Sensitive Grammar for {a^n b^{2n} c^n | n ≥ 1}')
  print('=' * 70)

  print('\n--- Grammar ---')
  print(f'T = {T}')
  print(f'N = {N}')
  print(f'S = {S}')
  print('P = {')
  for lhs, rhs in sorted(P, key=lambda x: (x[0], x[1])):
    print(f'    {lhs} → {rhs}')
  print('}')

  print('\n--- Monotonicity Check ---')
  for lhs, rhs in P:
    lhs_len = len(lhs.split())
    rhs_len = len(rhs.split())
    status = '✓' if lhs_len <= rhs_len else '✗'
    print(f'  {status} |{lhs}| = {lhs_len} ≤ |{rhs}| = {rhs_len}')

  print('\n--- Testing Derivability ---')

  # Should be derivable
  valid = ['a b b c', 'a a b b b b c c', 'a a a b b b b b b c c c']
  print('\nValid strings (should be derivable):')
  for s in valid:
    result = G.derivable(s)
    status = '✓' if result else '✗'
    n = s.count('a')
    print(f'  {status} n={n}: "{s}" → {result}')

  # Should NOT be derivable
  invalid = ['a b c', 'a b b b c', 'a b b c c', 'a a b b c c']
  print('\nInvalid strings (should NOT be derivable):')
  for s in invalid:
    result = G.derivable(s)
    status = '✓' if not result else '✗'
    print(f'  {status} "{s}" → {result}')

  print('\n--- Sample Derivation (n=1) ---')
  print('Target: a b b c')
  G.derivable('a b b c', log=True)

  print('\n--- Sample Derivation (n=2) ---')
  print('Target: a a b b b b c c')
  G.derivable('a a b b b b c c', log=True)


if __name__ == '__main__':
  main()
