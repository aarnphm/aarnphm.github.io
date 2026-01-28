#!/usr/bin/env python3
"""
A1: Grammar variations for expressions with identifiers a, b, c, d and operators +, -

Key principles:
- Tighter binding = deeper in grammar hierarchy (lower nonterminal)
- Left-associativity = left recursion (E -> E op T)
- Right-associativity = right recursion (E -> T op E)
"""

from nltk import CFG
from nltk.parse.chart import ChartParser
from nltk.tree import Tree


def parse_and_draw(grammar: CFG, sentence: str, title: str = ''):
  parser = ChartParser(grammar)
  tokens = list(sentence.replace(' ', ''))
  trees = list(parser.parse(tokens))
  if trees:
    if title:
      print(f'\n{title}')
    trees[0].pretty_print()
    return trees[0]
  print(f'No parse found for: {sentence}')
  return None


# Q1: + binds tighter than -, both left-associative
# a+b+c -> (a+b)+c
# a-b+c-d -> (a-(b+c))-d
grammar_q1 = CFG.fromstring("""
    E -> E '-' T | T
    T -> T '+' F | F
    F -> 'a' | 'b' | 'c' | 'd'
""")

# Q2: - binds tighter than +, both left-associative
# a+b+c -> (a+b)+c
# a-b+c-d -> (a-b)+(c-d)
grammar_q2 = CFG.fromstring("""
    E -> E '+' T | T
    T -> T '-' F | F
    F -> 'a' | 'b' | 'c' | 'd'
""")

# Q3: + and - bind equally, left-associative
# a+b+c -> (a+b)+c
# a-b+c-d -> ((a-b)+c)-d
grammar_q3 = CFG.fromstring("""
    E -> E '+' T | E '-' T | T
    T -> 'a' | 'b' | 'c' | 'd'
""")

# Q4: + and - bind equally, right-associative
# a+b+c -> a+(b+c)
# a-b+c-d -> a-(b+(c-d))
grammar_q4 = CFG.fromstring("""
    E -> T '+' E | T '-' E | T
    T -> 'a' | 'b' | 'c' | 'd'
""")

# Q5: unary - binds tighter than binary +, + left-associative
# -a+b+c -> ((-a)+b)+c
# a+-b+c -> ((a+(-b))+c)
# NOTE: the assignment says a+-b+c -> a+((-b)+c) but that contradicts left-assoc
# implementing standard left-associativity here
grammar_q5 = CFG.fromstring("""
    E -> E '+' T | T
    T -> '-' T | F
    F -> 'a' | 'b' | 'c' | 'd'
""")

# Q6: binary + binds tighter than unary -, + left-associative
# -a+b+c -> -((a+b)+c)
# a+-b+c -> a+(-(b+c))
grammar_q6 = CFG.fromstring("""
    E -> '-' E | T
    T -> T '+' U | U
    U -> '-' E | F
    F -> 'a' | 'b' | 'c' | 'd'
""")


def main():
  # Q1 tests
  parse_and_draw(grammar_q1, 'a+b+c', 'Q1: a+b+c -> (a+b)+c')
  parse_and_draw(grammar_q1, 'a-b+c-d', 'Q1: a-b+c-d -> (a-(b+c))-d')

  # Q2 tests
  parse_and_draw(grammar_q2, 'a+b+c', 'Q2: a+b+c -> (a+b)+c')
  parse_and_draw(grammar_q2, 'a-b+c-d', 'Q2: a-b+c-d -> (a-b)+(c-d)')

  # Q3 tests
  parse_and_draw(grammar_q3, 'a+b+c', 'Q3: a+b+c -> (a+b)+c')
  parse_and_draw(grammar_q3, 'a-b+c-d', 'Q3: a-b+c-d -> ((a-b)+c)-d')

  # Q4 tests
  parse_and_draw(grammar_q4, 'a+b+c', 'Q4: a+b+c -> a+(b+c)')
  parse_and_draw(grammar_q4, 'a-b+c-d', 'Q4: a-b+c-d -> a-(b+(c-d))')

  # Q5 tests
  parse_and_draw(grammar_q5, '-a+b+c', 'Q5: -a+b+c -> ((-a)+b)+c')
  parse_and_draw(grammar_q5, 'a+-b+c', 'Q5: a+-b+c -> ((a+(-b))+c)')

  # Q6 tests
  parse_and_draw(grammar_q6, '-a+b+c', 'Q6: -a+b+c -> -((a+b)+c)')
  parse_and_draw(grammar_q6, 'a+-b+c', 'Q6: a+-b+c -> a+(-(b+c))')


if __name__ == '__main__':
  main()
