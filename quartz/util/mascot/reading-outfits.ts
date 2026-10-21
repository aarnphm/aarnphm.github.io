import {
  CRUST,
  curve,
  densify,
  EYES,
  filled,
  hand,
  INK,
  PAPER,
  PINK,
  pressure,
  ring,
  rounded,
  SAGE,
  stroke,
  type Pt,
} from './pen'

// Bases: the librarian. Round glasses round the eyes, a temple arm back over the ear, and a stack
// of three books at its side in the site's colours.
export function librarian(w: number, paper = PAPER, ink = INK, spines = [PINK, SAGE, CRUST]) {
  const lenses = EYES.map((x, i) => ring(321 + i, x + 0.02, -0.2, 0.28))
  const bridge = hand(
    curve(
      [EYES[0] + 0.3, -0.24],
      [(EYES[0] + EYES[1]) / 2 + 0.01, -0.34],
      [EYES[1] - 0.26, -0.24],
      10,
    ),
    0.02,
    0.006,
    323,
  )
  const arm = hand(curve([EYES[1] + 0.3, -0.24], [0.68, -0.3], [0.94, -0.2], 10), 0.02, 0.006, 324)
  const glasses = `${lenses.map((l, i) => stroke(l, pressure(325 + i, w * 0.9, 0.2), ink)).join('')}${stroke(bridge, () => w * 0.8, ink)}${stroke(arm, () => w * 0.8, ink)}`
  const books = (
    [
      [1.02, 1.2, 0.78, 0.18, 0.04],
      [0.98, 1.01, 0.66, 0.2, -0.05],
      [1.05, 0.81, 0.72, 0.17, 0.06],
    ] as const
  )
    .map(([x, y, bw, bh, tilt], i) => {
      const c = Math.cos(tilt)
      const sn = Math.sin(tilt)
      const turn = (pts: Pt[]) =>
        pts.map(
          ([px, py]) =>
            [x + (px - x) * c - (py - y) * sn, y + (px - x) * sn + (py - y) * c] as const,
        )
      const cover = hand(turn(rounded(x, y, bw, bh, 0.03)), 0.03, 0.008, 330 + i)
      const pages = hand(
        turn([
          [x + bw / 2 - 0.1, y - bh / 2 + 0.04],
          [x + bw / 2 - 0.1, y + bh / 2 - 0.04],
        ]),
        0.02,
        0.004,
        334 + i,
      )
      const band = hand(
        turn([
          [x - bw / 2 + 0.12, y - bh / 2 + 0.03],
          [x - bw / 2 + 0.12, y + bh / 2 - 0.03],
        ]),
        0.02,
        0.004,
        338 + i,
      )
      return `${filled(cover, spines[i % spines.length], pressure(342 + i, w, 0.25), ink)}${stroke(pages, () => w * 0.5, ink)}${stroke(band, () => w * 0.7, paper)}`
    })
    .join('')
  return { glasses, books }
}

// Slides: the presenter. A bow tie under the chin and a pointer held out toward the slide. The
// site icon adds the slide itself, a small screen with a bar chart, for the pointer to touch.
export function presenter(w: number, paper = PAPER, ink = INK, tie = PINK) {
  const mx = (EYES[0] + EYES[1]) / 2
  const wing = (side: number) =>
    hand(
      [
        [mx, 0.44],
        [mx + side * 0.22, 0.32],
        [mx + side * 0.24, 0.58],
        [mx, 0.46],
      ],
      0.02,
      0.008,
      351 + side,
    )
  const bow = `${[-1, 1].map(side => filled(densify(wing(side), 0.03), tie, pressure(353 + side, w, 0.25), ink)).join('')}${filled(hand(rounded(mx, 0.45, 0.09, 0.11, 0.03), 0.01, 0.004, 355), tie, () => w * 0.7, ink)}`
  const stick = hand(
    [
      [-0.86, 0.42],
      [-1.66, -0.74],
    ],
    0.04,
    0.006,
    356,
  )
  const pointer = `${stroke(stick, t => w * (1.5 - 0.8 * t), ink)}${filled(ring(357, -1.68, -0.77, 0.05), paper, () => w * 0.6, ink)}`
  const frame = hand(rounded(-1.66, -1.02, 1.0, 0.72, 0.03), 0.03, 0.01, 358)
  const bars = (
    [
      [-1.94, 0.22, SAGE],
      [-1.72, 0.36, CRUST],
      [-1.5, 0.5, tie],
    ] as const
  )
    .map(([x, h, fill], i) =>
      filled(
        hand(rounded(x, -0.72 - h / 2, 0.14, h, 0.01), 0.02, 0.004, 360 + i),
        fill,
        () => w * 0.55,
        ink,
      ),
    )
    .join('')
  const legs = (
    [
      [
        [-1.94, -0.66],
        [-2.08, -0.2],
      ],
      [
        [-1.38, -0.66],
        [-1.24, -0.2],
      ],
    ] as Pt[][]
  )
    .map((pts, i) => stroke(hand(pts, 0.03, 0.006, 364 + i), () => w * 0.9, ink))
    .join('')
  const screen = `${legs}${filled(frame, paper, pressure(366, w, 0.3), ink)}${bars}`
  return { bow, pointer, screen }
}

// Stream: the reader. The librarian's glasses on, and a sheet held up in front at a tilt, a stubby
// hand at each side, its top corner dog-eared, four lines of text on it, one marked in `mark`.
export function reader(w: number, paper = PAPER, ink = INK, mark = PINK) {
  const [cx, cy, sw, sh, tilt] = [-0.42, 0.74, 1.02, 0.72, -0.12]
  const c = Math.cos(tilt)
  const sn = Math.sin(tilt)
  const turn = (pts: readonly Pt[]): Pt[] =>
    pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * sn, cy + (x - cx) * sn + (y - cy) * c])
  const [l, r, t, b] = [cx - sw / 2, cx + sw / 2, cy - sh / 2, cy + sh / 2]
  const ear = 0.15
  const outline = hand(
    turn([
      [l, t],
      [r - ear, t],
      [r, t + ear],
      [r, b],
      [l, b],
      [l, t],
    ]),
    0.03,
    0.01,
    391,
  )
  const fold = hand(
    turn([
      [r - ear, t],
      [r - ear + 0.02, t + ear - 0.02],
      [r, t + ear],
    ]),
    0.02,
    0.004,
    392,
  )
  const rows = [0.74, 0.62, 0.8, 0.46].map((len, i) => {
    const y = t + 0.16 + i * 0.13
    return hand(
      turn([
        [l + 0.11, y],
        [l + 0.11 + (sw - 0.22) * len, y],
      ]),
      0.03,
      0.008,
      393 + i,
    )
  })
  const text = rows
    .map(
      (row, i) =>
        `${i === 1 ? stroke(row, () => w * 2.6, mark) : ''}${stroke(row, () => w * 0.55, ink)}`,
    )
    .join('')
  const paws = [l + 0.02, r - 0.02]
    .map((x, i) =>
      filled(
        hand(turn(rounded(x, cy + 0.08, 0.2, 0.17, 0.08)), 0.02, 0.006, 398 + i),
        paper,
        pressure(400 + i, w, 0.2),
        ink,
      ),
    )
    .join('')
  const sheet = `${filled(outline, paper, pressure(402, w, 0.3), ink)}${stroke(fold, () => w * 0.7, ink)}${text}${paws}`
  return { glasses: librarian(w, paper, ink).glasses, sheet }
}
