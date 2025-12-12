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

  # separate vertical edges for ray casting
  vertical_edges = [(ax, min(ay, by), max(ay, by)) for (ax, ay), (bx, by) in edges if ax == bx]

  # ray casting counting vertical edges. odd is inside, even is outside
  def point_in_polygon(px: int, py: int) -> bool:
    crossings = 0
    for ex, ey1, ey2 in vertical_edges:
      if ex > px and ey1 <= py < ey2:
        crossings += 1
    return crossings % 2 == 1

  def on_boundary(px: int, py: int) -> bool:
    for (ax, ay), (bx, by) in edges:
      if ay == by:  # horizontal edge
        if py == ay and min(ax, bx) <= px <= max(ax, bx):
          return True
      else:  # vertical edge
        if px == ax and min(ay, by) <= py <= max(ay, by):
          return True
    return False

  def point_valid(px: int, py: int) -> bool:
    return on_boundary(px, py) or point_in_polygon(px, py)

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
    x1, x2 = min(pa[0], pb[0]), max(pa[0], pb[0])
    y1, y2 = min(pa[1], pb[1]), max(pa[1], pb[1])

    # no edge cuts through rectangle interior
    if any(edge_cuts_through(pa, pb, e) for e in edges):
      return False

    # all four corners must be valid (on boundary or inside)
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    if not all(point_valid(cx, cy) for cx, cy in corners):
      return False

    # center point must be inside or on boundary
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if not point_valid(cx, cy):
      return False

    return True

  return max((area(pa, pb) for pa in points for pb in points if rect_valid(pa, pb)), default=0)


if __name__ == '__main__':
  points = parse('d9.txt')
  print(f'p1: {p1(points)}')
  print(f'p2: {p2(points)}')
