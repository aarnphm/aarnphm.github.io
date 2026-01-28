#!/usr/bin/env python3
"""
A4: Explain the languages described by regular expressions.

The task is to describe WHAT strings are in the language,
not to paraphrase the regex structure.
"""


def main():
  print('=' * 70)
  print('A4: Languages Described by Regular Expressions')
  print('=' * 70)

  explanations = [
    # Q1
    (
      r'(a*b*)*',
      'All strings over the alphabet {a, b}, including the empty string.',
      """This is equivalent to {a,b}* (Kleene closure of the alphabet).
Every possible sequence of a's and b's in any order and any length is included.
There are no restrictions on adjacency, ordering, or counts.

Formally: L = {w ∈ {a,b}* | w is any string}

In:  "", "a", "b", "aa", "bb", "ab", "ba", "aab", "bba", "abab", "baba", "aaabbb", "bbbaaa"
     (everything over {a,b})""",
      ['', 'a', 'b', 'ab', 'ba', 'aabb', 'bbaa', 'abab', 'baba', 'aaabbb'],
    ),
    # Q2
    (
      r'(a*[b])*',
      'All strings over the alphabet {a, b}, including the empty string.',
      """Equivalent to {a,b}* — the same language as Q1, just expressed differently.
[b] means "zero or one b", so each iteration is "any number of a's, then optionally one b".
Any string over {a,b} can be decomposed into such segments.

Formally: L = {w ∈ {a,b}* | w is any string}

In:  "", "a", "b", "aa", "bb", "ab", "ba", "bba", "abba"
     (everything over {a,b})""",
      ['', 'a', 'b', 'ab', 'ba', 'bb', 'aa', 'aab', 'bba', 'abba'],
    ),
    # Q3
    (
      r'(a*ba*b)*a*',
      "Strings over {a, b} containing an even number of b's (0, 2, 4, 6, ...).",
      """Each iteration of (a*ba*b) adds exactly two b's to the string.
The a* segments allow arbitrary a's anywhere. Zero iterations means zero b's.

Formally: L = {w ∈ {a,b}* | #b(w) ≡ 0 (mod 2)}
          where #b(w) counts occurrences of b in w.

In:  "", "a", "aa", "bb", "abb", "bba", "abab", "baba", "abba", "bbbb", "aabbaa"
Out: "b", "bbb", "ababab" (odd number of b's)""",
      ['', 'a', 'aa', 'bb', 'abab', 'baba', 'aabbaa', 'bbbb', 'abba'],
    ),
    # Q4
    (
      r'(a*[ba*c])*',
      'Strings over {a, b, c} where b and c occur only in non-overlapping (b...c) blocks, '
      "with only a's between each b and its matching c.",
      """Structure: a*(ba*c)* with a's allowed between blocks.
Each b must pair with exactly one c that follows it. Between a b and its c,
only a's may appear. The pairs cannot nest (no b..b..c..c) or interleave.
Strings with only a's (no b or c) are included.

Formally: L = {w ∈ {a,b,c}* | w matches pattern a*(ba*ca*)*}
          Equivalently: #b(w) = #c(w), and in every prefix, #b ≥ #c,
          and between each b and its corresponding c, only a's appear.

In:  "", "a", "aaa", "bc", "bac", "baac", "abc", "abca", "bcbc", "abcabc", "abacabaca"
Out: "b", "c", "cb", "bcc", "bbc", "acb", "bcb" (unmatched or misordered b/c)""",
      ['', 'a', 'aa', 'bc', 'bac', 'abc', 'abca', 'bcbc', 'abacbc', 'aabaaaca'],
    ),
    # Q5
    (
      r'(a|ba)*[b]',
      'Strings over {a, b} that do not contain two consecutive b\'s (no "bb" substring).',
      """The pattern (a|ba)* builds strings where every b is immediately followed by a.
The final [b] allows an optional trailing b. Combined: no position has bb.

Formally: L = {w ∈ {a,b}* | w does not contain "bb" as a substring}

In:  "", "a", "b", "aa", "ab", "ba", "aba", "bab", "abab", "baba", "ababa"
Out: "bb", "abb", "bba", "abba", "bbaa", "aabb" (contain "bb")""",
      ['', 'a', 'b', 'ab', 'ba', 'aba', 'bab', 'abab', 'baba'],
    ),
    # Q6
    (
      r'a*(ba+)*',
      "Strings over {a, b} that either contain no b's, or where every b is followed by "
      'at least one a (equivalently: no "bb" and does not end in b).',
      """The a+ after each b requires at least one a to follow every b.
This means: (1) string cannot end with b, (2) no two b's can be adjacent.
Strings of pure a's (including empty) satisfy this vacuously.

Formally: L = {w ∈ {a,b}* | w does not end in b, and w does not contain "bb"}
          Equivalently: every b in w is immediately followed by at least one a.

In:  "", "a", "aa", "ba", "baa", "aba", "abaa", "baba", "babaa", "aababaa"
Out: "b", "ab", "bb", "bab", "abb", "abab" (ends in b, or contains "bb")""",
      ['', 'a', 'aa', 'ba', 'baa', 'aba', 'baba', 'aabaa'],
    ),
  ]

  for i, (regex, short_desc, explanation, examples) in enumerate(explanations, 1):
    print(f'\n{"─" * 70}')
    print(f'Q{i}: {regex}')
    print(f'{"─" * 70}')
    print(f'\n  Language: {short_desc}')
    print(f'\n  Explanation:')
    for line in explanation.strip().split('\n'):
      print(f'    {line}')
    print(f'\n  Examples in L: {", ".join(repr(s) if s else "ε" for s in examples)}')

  # Verification section
  print(f'\n{"=" * 70}')
  print('Verification with Python regex')
  print('=' * 70)

  import re

  # Convert mathematical notation to Python regex
  patterns = [
    (r'(a*b*)*', r'^(a*b*)*$'),
    (r'(a*[b])*', r'^(a*b?)*$'),
    (r'(a*ba*b)*a*', r'^(a*ba*b)*a*$'),
    (r'(a*[ba*c])*', r'^(a*(ba*c)?)*$'),
    (r'(a|ba)*[b]', r'^(a|ba)*b?$'),
    (r'a*(ba+)*', r'^a*(ba+)*$'),
  ]

  test_cases = [
    # Q1, Q2: all strings over {a,b}
    (['', 'a', 'b', 'ab', 'ba', 'bb', 'aa', 'abba', 'baba'], [0, 1]),
    # Q3: even number of b's
    (['', 'a', 'bb', 'abab', 'bbbb'], [2]),
    (['b', 'bbb', 'abb'], [2]),  # odd b's - should fail Q3
    # Q4: matched b-c pairs
    (['', 'a', 'bc', 'bac', 'abca', 'bcbc'], [3]),
    (['b', 'c', 'cb', 'bcc', 'bbc'], [3]),  # unmatched - should fail Q4
    # Q5: no consecutive b's
    (['', 'a', 'b', 'ab', 'ba', 'aba', 'bab'], [4]),
    (['bb', 'abb', 'bba', 'abba'], [4]),  # has bb - should fail Q5
    # Q6: doesn't end in b, no consecutive b's
    (['', 'a', 'ba', 'aba', 'baba', 'aabaa'], [5]),
    (['b', 'ab', 'bb', 'bab'], [5]),  # ends in b or has bb - should fail Q6
  ]

  print('\nPattern matching tests:')
  for strings, question_indices in test_cases:
    for qi in question_indices:
      pattern = patterns[qi][1]
      for s in strings:
        match = bool(re.match(pattern, s))
        display = repr(s) if s else 'ε'
        print(f'  Q{qi + 1}: {display:10} → {"✓" if match else "✗"}')


if __name__ == '__main__':
  main()
