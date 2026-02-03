class RegEx:
  pass


class ε(RegEx):
  def __repr__(self):
    return 'ε'


class Sym(RegEx):
  def __init__(self, a: str):
    self.a = a

  def __repr__(self):
    return str(self.a)


class Choice(RegEx):
  def __init__(self, E1: RegEx, E2: RegEx):
    self.E1, self.E2 = E1, E2

  def __repr__(self):
    return '(' + str(self.E1) + '|' + str(self.E2) + ')'


class Conc(RegEx):
  def __init__(self, E1: RegEx, E2: RegEx):
    self.E1, self.E2 = E1, E2

  def __repr__(self):
    return '(' + str(self.E1) + str(self.E2) + ')'


class Star(RegEx):
  def __init__(self, E: RegEx):
    self.E = E

  def __repr__(self):
    return '(' + str(self.E) + ')*'


class fset(frozenset):
  def __repr__(self):
    return '{' + ', '.join(str(e) for e in self) + '}'


TransFunc = dict[str, dict[str, set[str]]]


class FiniteStateAutomaton:
  Σ: set[str]
  Q: set[str]
  I: set[str]
  δ: TransFunc
  F: set[str]
  vars = ()

  def __init__(self, Σ, Q, I, δ, F):
    self.Σ, self.Q, self.I, self.δ, self.F = Σ, fset(Q), fset(I), δ, fset(F)


def setunion(S: set[set]) -> set:
  return set.union(set(), *S)


def δ̂(δ: TransFunc, P: set[str], a: str) -> set[str]:
  return fset(setunion(δ[p][a] for p in P if p in δ if a in δ[p]))


def ε_closure(Q, δ) -> set:
  C, W = set(Q), Q
  while W != set():
    W = δ̂(δ, W, 'ε') - C
    C |= W
  return fset(C)


def accepts(A: FiniteStateAutomaton, α: str):
  W = ε_closure(A.I, A.δ)
  for a in α:
    W = ε_closure(δ̂(A.δ, W, a), A.δ)
  return W & A.F != set()


def merge(γ: TransFunc, δ: TransFunc) -> TransFunc:
  return (
    {q: γ[q] for q in γ.keys() - δ.keys()}
    | {q: δ[q] for q in δ.keys() - γ.keys()}
    | {q: {a: γ[q].get(a, set()) | δ[q].get(a, set()) for a in γ[q].keys() | δ[q].keys()} for q in γ.keys() & δ.keys()}
  )


def RegExToFSA(re) -> FiniteStateAutomaton:
  def ToFSA(re) -> FiniteStateAutomaton:
    nonlocal QC
    match re:
      case ε():
        q = QC
        QC += 1
        return FiniteStateAutomaton(set(), {q}, {q}, {}, {q})
      case Sym(a=a):
        q = QC
        QC += 1
        r = QC
        QC += 1
        return FiniteStateAutomaton({a}, {q, r}, {q}, {q: {a: {r}}}, {r})
      case Choice(E1=E1, E2=E2):
        A1, A2 = ToFSA(E1), ToFSA(E2)
        q = QC
        QC += 1
        δ = A1.δ | A2.δ | {q: {'ε': A1.I | A2.I}}
        return FiniteStateAutomaton(A1.Σ | A2.Σ, A1.Q | A2.Q | {q}, {q}, δ, A1.F | A2.F)
      case Conc(E1=E1, E2=E2):
        A1, A2 = ToFSA(E1), ToFSA(E2)
        δ = merge(A1.δ | A2.δ, {q: {'ε': A2.I} for q in A1.F})
        return FiniteStateAutomaton(A1.Σ | A2.Σ, A1.Q | A2.Q, A1.I, δ, A2.F)
      case Star(E=E):
        A = ToFSA(E)
        δ = merge(A.δ, {q: {'ε': A.I} for q in A.F})
        return FiniteStateAutomaton(A.Σ, A.Q, A.I, δ, A.I | A.F)
      case E:
        raise Exception(str(E) + ' not a regular expression')

  QC = 0
  return ToFSA(re)


# === HELPERS ===


def choices(s: str) -> RegEx:
  result = Sym(s[0])
  for c in s[1:]:
    result = Choice(result, Sym(c))
  return result


def plus(e: RegEx) -> RegEx:
  return Conc(e, Star(e))


def opt(e: RegEx) -> RegEx:
  return Choice(ε(), e)


def seq(*args: RegEx) -> RegEx:
  result = args[0]
  for e in args[1:]:
    result = Conc(result, e)
  return result


# === PART 1: IDENTIFIERS ===
# identifier = letter (letter | digit)*

LETTERS = 'abcdefghijklmnopqrstuvwxyz'
DIGITS = '0123456789'

I = Conc(choices(LETTERS), Star(Choice(choices(LETTERS), choices(DIGITS))))


# === PART 2: DOLLAR AMOUNTS ===
# $ followed by digits (with optional consistent comma grouping), optional .dd cents
#
# Without commas: d+
# With commas: (d|dd|ddd)(,ddd)+
# Cents: (.dd)?


def digit():
  return choices(DIGITS)


def dn(n: int) -> RegEx:
  if n == 1:
    return digit()
  return Conc(digit(), dn(n - 1))


d_plus = plus(digit())
d1_or_d2_or_d3 = Choice(digit(), Choice(dn(2), dn(3)))
comma_d3 = Conc(Sym(','), dn(3))
amount_with_commas = Conc(d1_or_d2_or_d3, plus(comma_d3))
amount = Choice(d_plus, amount_with_commas)
cents = opt(Conc(Sym('.'), dn(2)))

C = seq(Sym('$'), amount, cents)


# === TESTS ===

if __name__ == '__main__':
  A = RegExToFSA(I)
  assert accepts(A, 'cloud7')
  assert accepts(A, 'if')
  assert accepts(A, 'b12')
  assert not accepts(A, '007')
  assert not accepts(A, '15b')
  assert not accepts(A, 'B12')
  assert not accepts(A, 'e-mail')
  print('Part 1 (identifiers): all tests passed')

  B = RegExToFSA(C)
  assert accepts(B, '$27.04')
  assert accepts(B, '$11,222,333')
  assert accepts(B, '$0')
  assert not accepts(B, '27.04')
  assert not accepts(B, '$11222,333')
  assert not accepts(B, '$35.5')
  assert not accepts(B, '$9.409')
  print('Part 2 (dollar amounts): all tests passed')
