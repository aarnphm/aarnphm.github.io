import pathlib


def parse(path: str) -> list[tuple[int, int]]:
  return [tuple(map(int, line.split(','))) for line in pathlib.Path(path).read_text().strip().split('\n')]


def draw_polygon(points: list[tuple[int, int]], width: int = 120, height: int = 60):
  if not points:
    return

  min_x = min(p[0] for p in points)
  max_x = max(p[0] for p in points)
  min_y = min(p[1] for p in points)
  max_y = max(p[1] for p in points)

  print(f'bounding box: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]')
  print(f'dimensions: {max_x - min_x + 1} × {max_y - min_y + 1}')
  print()

  scale_x = (max_x - min_x) / (width - 1) if max_x > min_x else 1
  scale_y = (max_y - min_y) / (height - 1) if max_y > min_y else 1

  grid = [[' ' for _ in range(width)] for _ in range(height)]

  def to_grid(x, y):
    gx = int((x - min_x) / scale_x)
    gy = int((y - min_y) / scale_y)
    return min(gx, width - 1), min(gy, height - 1)

  n = len(points)
  for i in range(n):
    p1, p2 = points[i], points[(i + 1) % n]
    x1, y1 = to_grid(*p1)
    x2, y2 = to_grid(*p2)

    if y1 == y2:
      for x in range(min(x1, x2), max(x1, x2) + 1):
        if grid[y1][x] == ' ':
          grid[y1][x] = '-'
    elif x1 == x2:
      for y in range(min(y1, y2), max(y1, y2) + 1):
        if grid[y][x1] == ' ':
          grid[y][x1] = '|'

  # red tiles
  for x, y in points:
    gx, gy = to_grid(x, y)
    grid[gy][gx] = '#'

  for row in grid:
    print(''.join(row))


if __name__ == '__main__':
  points = parse('d9.txt')
  draw_polygon(points)
