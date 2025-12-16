from __future__ import annotations

import random, typing as t

type Cell = tuple[int, int]
type Color = t.Literal['white', 'black']

GRID_SIZE = 12

def color(cell: Cell) -> Color: return 'white' if (cell[0] + cell[1]) % 2 == 0 else 'black'

def neighbors(cell: Cell) -> list[Cell]:
  i, j = cell
  return [(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if 1 <= x <= GRID_SIZE and 1 <= y <= GRID_SIZE]

def find_domino(remaining: set[Cell]) -> tuple[Cell, Cell] | None:
  for cell in remaining:
    for neighbor in neighbors(cell):
      if neighbor in remaining: return (cell, neighbor)

def simulate_optimal() -> int:
  r = {(i,j) for i in range(1, GRID_SIZE + 1) for j in range(1, GRID_SIZE + 1)}
  lily = 0

  while True:
    whites = [c for c in r if color(c) == 'white']
    if not whites: break
    ben = whites[0]
    r.remove(ben)

    domino = find_domino(r)
    if domino is None: break
    r.remove(domino[0])
    r.remove(domino[1])
    lily+=2
  return lily


def simulate_random(seed: int = 42) -> int:
  random.seed(seed)
  r = {(i, j) for i in range(1, GRID_SIZE + 1) for j in range(1, GRID_SIZE + 1)}
  lily = 0

  while r:
    ben = random.choice(list(r))
    r.remove(ben)

    domino = find_domino(r)
    if domino is None: break
    r.remove(domino[0])
    r.remove(domino[1])
    lily += 2

  return lily

if __name__ == '__main__':
  print(f"optimal: {simulate_optimal()}")
  print(f"randoms: min={min(random_results:=[simulate_random(seed=s) for s in range(100)])}, max={max(random_results)}, avg={sum(random_results) / len(random_results):.1f}")
