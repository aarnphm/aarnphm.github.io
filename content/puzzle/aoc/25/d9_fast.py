from pathlib import Path
from bisect import bisect_left, bisect_right
from collections import defaultdict


def parse(path: str) -> list[tuple[int, int]]:
  return [tuple(map(int, line.split(','))) for line in Path(path).read_text().strip().split('\n')]


def area(p1: tuple[int, int], p2: tuple[int, int]) -> int:
  return (abs(p2[0] - p1[0]) + 1) * (abs(p2[1] - p1[1]) + 1)


def p1(points: list[tuple[int, int]]) -> int:
  # optimization: only need to check pairs once, area is symmetric
  best = 0
  for i, p1 in enumerate(points):
    for p2 in points[i + 1 :]:
      best = max(best, area(p1, p2))
  return best


def p2(points: list[tuple[int, int]]) -> int:
  n = len(points)

  # group edges by coordinate for fast lookup
  h_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)  # y -> [(x1, x2), ...]
  v_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)  # x -> [(y1, y2), ...]

  for i in range(n):
    (ax, ay), (bx, by) = points[i], points[(i + 1) % n]
    if ay == by:
      h_edges[ay].append((min(ax, bx), max(ax, bx)))
    else:
      v_edges[ax].append((min(ay, by), max(ay, by)))

  sorted_ys = sorted(h_edges.keys())
  sorted_xs = sorted(v_edges.keys())

  def rect_valid(pa, pb):
    x1, x2 = (pa[0], pb[0]) if pa[0] < pb[0] else (pb[0], pa[0])
    y1, y2 = (pa[1], pb[1]) if pa[1] < pb[1] else (pb[1], pa[1])

    # check horizontal edges with y in (y1, y2)
    for i in range(bisect_right(sorted_ys, y1), bisect_left(sorted_ys, y2)):
      for ex1, ex2 in h_edges[sorted_ys[i]]:
        if max(x1, ex1) < min(x2, ex2):
          return False

    # check vertical edges with x in (x1, x2)
    for i in range(bisect_right(sorted_xs, x1), bisect_left(sorted_xs, x2)):
      for ey1, ey2 in v_edges[sorted_xs[i]]:
        if max(y1, ey1) < min(y2, ey2):
          return False

    return True

  # sort pairs by area descending, early termination on first valid
  pairs = sorted(((area(points[i], points[j]), i, j) for i in range(n) for j in range(i + 1, n)), reverse=True)

  for a, i, j in pairs:
    if rect_valid(points[i], points[j]):
      return a
  return 0


if __name__ == '__main__':
  points = parse('d9.txt')
  print(f'p1: {p1(points)}')
  print(f'p2: {p2(points)}')
