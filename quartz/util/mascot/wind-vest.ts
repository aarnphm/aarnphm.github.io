import {
  curve,
  even,
  filled,
  hand,
  INK,
  PINK,
  poly,
  pressure,
  rounded,
  SAGE,
  stroke,
  type Pt,
} from './pen'

// Triathlon in the cold: the wind vest a rider zips on for a cold descent. `body` is the pet's own
// outline, so the shell's fill tucks under its line from the mouth down. A stand-up collar, a zip
// down the front (the left, where it faces) with its pull, a reflective band across the chest, and
// the back panel caught by the wind: `flaps` are two drawings of it to swap between.
export function windVest(body: readonly Pt[], w: number, ink = INK, shell = SAGE, tape = PINK) {
  // A V neck from shoulder to shoulder, the arms coming out above it, down to where the zip stops
  // under the mouth (it faces left, so the V sits left of centre); piping a little inside it.
  const zx = -0.16
  const top = (x: number) => 0.43 - 0.48 * ((x - zx) / (x < zx ? 0.74 : 1.06)) ** 2
  const across = (f: (x: number) => number, from: number, to: number) =>
    Array.from({ length: 33 }, (_, i) => {
      const x = from + ((to - from) * i) / 32
      return [x, f(x)] as const
    })
  // Half the body's width at height y, a little inside its line.
  const edge = (y: number) => 0.95 * Math.sqrt(Math.max(0, 1 - y * y))
  const inside = (pts: Pt[]) => pts.filter(([x, y]) => Math.abs(x) < edge(y))
  const lower = body.map(([x, y]) => [x * 0.975, y * 0.975] as const).filter(([x, y]) => y > top(x))
  const shape: Pt[] = [...lower, ...across(top, lower[lower.length - 1][0], lower[0][0])]
  // The silhouette's lines follow the theme's ink; the marks on the shell stay dark on its fill.
  const line = (pts: Pt[], seed: number, k = 1, color = INK) =>
    stroke(pts, pressure(seed, w * k, 0.3), color)
  const collar = line(hand(inside(across(top, -1, 1)), 0.03, 0.008, 701), 702, 1, ink)
  const crease = line(hand(inside(across(x => top(x) + 0.11, -1, 1)), 0.03, 0.008, 703), 704, 0.55)
  const mid = (x: number) => 0.7 + 0.04 * (1 - x * x)
  const tapeEdge = (d: number, from: number, to: number) => across(x => mid(x) + d, from, to)
  const reach = edge(0.78)
  const band = hand(
    [
      ...tapeEdge(-0.05, -reach, reach),
      ...tapeEdge(0.05, reach, -reach),
      [-reach, mid(-reach) - 0.05],
    ],
    0.03,
    0.006,
    705,
  )
  const zipLine = curve([zx, top(zx)], [zx - 0.1, 0.68], [zx - 0.14, 0.94], 16)
  const zip = line(hand(zipLine, 0.03, 0.006, 706), 707, 0.75)
  const teeth = zipLine
    .filter((_, i) => i % 2 && i < 15)
    .map(([x, y]) =>
      stroke(
        [
          [x - 0.035, y],
          [x + 0.035, y],
        ],
        () => w * 0.35,
        INK,
      ),
    )
    .join('')
  const pull = filled(
    hand(rounded(zx + 0.01, top(zx) + 0.1, 0.07, 0.15, 0.03), 0.02, 0.004, 708),
    INK,
    () => w * 0.3,
    INK,
  )
  const vest =
    `<path d="${poly(shape)}" style="fill:${shell}"/>${crease}` +
    `<path d="${poly(band)}" style="fill:${tape}"/>${stroke(band, () => w * 0.45, INK)}${zip}${teeth}${collar}${pull}`

  // The back panel, out from the body's own line behind it (on its right) and back, its hem
  // rippling; the wind's streaks trail off behind.
  const back = body.filter(([x, y]) => {
    const a = Math.atan2(y, x)
    return a > 0.35 && a < 1.25
  })
  const flaps = [
    [0.12, 0.25, 0],
    [0.17, 0.2, 2.2],
  ].map(([amp, ripple, phase], k) => {
    const n = back.length - 1
    const out = back.map(([x, y], i) => {
      const t = i / n
      const s =
        1 + amp * Math.sin(Math.PI * t) ** 0.8 * (1 + ripple * Math.sin(4 * Math.PI * t + phase))
      return [x * s, y * s] as const
    })
    const panel = [...out, ...back.map(([x, y]) => [x * 0.92, y * 0.92] as const).reverse()]
    const streaks = [
      [1.2 + 0.05 * k, 0.36, 0.34],
      [1.12 - 0.03 * k, 0.74, 0.26],
    ]
      .map(([x, y, len], i) =>
        stroke(
          hand(
            curve([x, y], [x + len / 2, y + 0.04], [x + len, y - 0.02], 10),
            0.03,
            0.006,
            710 + 2 * k + i,
          ),
          even(w * 0.6),
          ink,
        ),
      )
      .join('')
    return `<path d="${poly(panel)}" style="fill:${shell}"/>${line(out, 714 + k, 1, ink)}${streaks}`
  })
  return { vest, flaps }
}
