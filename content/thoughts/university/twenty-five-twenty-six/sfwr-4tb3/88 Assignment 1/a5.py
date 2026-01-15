from typing import Generator
from collections.abc import Iterable


def printleaves(node):
  if isinstance(node, Iterable):
    for c in node:
      printleaves(c)
  else:
    print(node)


def leaves(tree):
  if isinstance(tree, Iterable):
    for c in tree: yield from leaves(c)
  else: yield tree


if __name__ == '__main__':
  L = ((1, 2), 3, (4, (5, 6)))

  printleaves(L)
  gen = leaves(L)
  assert isinstance(gen, Generator)
  assert list(leaves(L)) == [1, 2, 3, 4, 5, 6]
  assert sum(leaves(L)) == 21
