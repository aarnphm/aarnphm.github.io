from pathlib import Path

def parse(path: str) -> list[tuple[int, int]]:
  return [tuple(map(int, line.split(','))) for line in Path(path).read_text().strip().split('\n')]

def area(p1: tuple[int, int], p2: tuple[int, int]) -> int:
  return (abs(p2[0] - p1[0]) + 1) * (abs(p2[1] - p1[1]) + 1)

def p1(points: list[tuple[int, int]]) -> int:
  return max(area(p1, p2) for p1 in points for p2 in points)

def p2(points: list[tuple[int, int]]) -> int:
  n = len(points)
  edges = [(points[i], points[(i + 1) % n]) for i in range(n)]

  def edge_cuts_through(pa, pb, edge):
    x1, x2 = min(pa[0], pb[0]), max(pa[0], pb[0])
    y1, y2 = min(pa[1], pb[1]), max(pa[1], pb[1])
    (ax, ay), (bx, by) = edge
    if ay == by:  # horizontal
      ey, ex1, ex2 = ay, min(ax, bx), max(ax, bx)
      return y1 < ey < y2 and max(x1, ex1) < min(x2, ex2)
    else:  # vertical
      ex, ey1, ey2 = ax, min(ay, by), max(ay, by)
      return x1 < ex < x2 and max(y1, ey1) < min(y2, ey2)

  def rect_valid(pa, pb):
    return not any(edge_cuts_through(pa, pb, e) for e in edges)

  return max((area(pa, pb) for pa in points for pb in points if rect_valid(pa, pb)), default=0)

if __name__ == '__main__':
  points = parse('d9.txt')
  print(f'p1: {p1(points)}')
  print(f'p2: {p2(points)}')
