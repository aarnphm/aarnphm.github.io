import numpy as np, pathlib


def parse(path: str) -> np.ndarray:
  return np.array([list(map(int, line.split(','))) for line in pathlib.Path(path).read_text().strip().split('\n')])


def p1(pts: np.ndarray) -> int:
  return int(((np.abs(pts[:, 0, None] - pts[None, :, 0]) + 1) * np.abs(pts[:, 1, None] - pts[None, :, 1]) + 1).max())


def p2(pts: np.ndarray) -> int:
  n = len(pts)

  # build edge arrays
  next_pts = np.roll(pts, -1, axis=0)
  is_horiz = pts[:, 1] == next_pts[:, 1]

  h_y = pts[is_horiz, 1]
  h_x1 = np.minimum(pts[is_horiz, 0], next_pts[is_horiz, 0])
  h_x2 = np.maximum(pts[is_horiz, 0], next_pts[is_horiz, 0])

  v_x = pts[~is_horiz, 0]
  v_y1 = np.minimum(pts[~is_horiz, 1], next_pts[~is_horiz, 1])
  v_y2 = np.maximum(pts[~is_horiz, 1], next_pts[~is_horiz, 1])

  # precompute all pair bounds
  i_idx, j_idx = np.triu_indices(n, k=1)
  x1 = np.minimum(pts[i_idx, 0], pts[j_idx, 0])
  x2 = np.maximum(pts[i_idx, 0], pts[j_idx, 0])
  y1 = np.minimum(pts[i_idx, 1], pts[j_idx, 1])
  y2 = np.maximum(pts[i_idx, 1], pts[j_idx, 1])
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)

  # sort by area descending
  order = np.argsort(-areas)

  for idx in order:
    rx1, rx2, ry1, ry2 = x1[idx], x2[idx], y1[idx], y2[idx]

    # check horizontal edges: y in (ry1, ry2) and x-overlap
    h_mask = (ry1 < h_y) & (h_y < ry2)
    if h_mask.any():
      h_overlap = np.maximum(rx1, h_x1[h_mask]) < np.minimum(rx2, h_x2[h_mask])
      if h_overlap.any():
        continue

    # check vertical edges: x in (rx1, rx2) and y-overlap
    v_mask = (rx1 < v_x) & (v_x < rx2)
    if v_mask.any():
      v_overlap = np.maximum(ry1, v_y1[v_mask]) < np.minimum(ry2, v_y2[v_mask])
      if v_overlap.any():
        continue

    return int(areas[idx])

  return 0


if __name__ == '__main__':
  pts = parse('d9.txt')
  print(f'p1: {p1(pts)}')
  print(f'p2: {p2(pts)}')
