// The campfire on the night card, in the pen of the camper and the tent and drawn to read at card
// size: firelight pooling on the ledge, four flat stones sunk round it, two crossed logs with their
// end grain showing, one flame in three tongues with a pink heart, two sparks and a wisp of smoke.

import { mulberry } from '../../components/scripts/404-pixel'
import { curve, hand, loop, poly, pressure, stroke, type Pt } from '../../util/mascot/pen'
import { H, mix, shape, W } from './scene'

export type FireTone = { ember: string; pink: string; moonlight: string; night: string }

// The tent and camper are inked at 2.6 to 3; the fire takes the same pen.
const PEN = 2.6

export function fire(fx: number, fg: number, ledge: readonly Pt[], tone: FireTone) {
  const rand = mulberry(99)
  const pen = (seed: number, base = PEN) => pressure(seed, base, 0.3)
  const grey = (t: number) => mix(tone.night, tone.moonlight, t)

  // Light on the ground only: the glow is clipped under the ridge line.
  const glow = `<defs><clipPath id="fire-c"><path d="${poly([...ledge, [W + 20, H + 20], [-20, H + 20]])}"/></clipPath><radialGradient id="fire-g" gradientUnits="userSpaceOnUse" cx="0" cy="0" r="1" gradientTransform="translate(${fx} ${fg + 2}) scale(120 20)"><stop offset="0" stop-color="${tone.ember}" stop-opacity="0.32"/><stop offset="0.5" stop-color="${tone.ember}" stop-opacity="0.1"/><stop offset="1" stop-color="${tone.ember}" stop-opacity="0"/></radialGradient></defs><g clip-path="url(#fire-c)"><ellipse cx="${fx}" cy="${fg + 2}" rx="120" ry="20" fill="url(#fire-g)"/></g>`

  // Stones: flat-bottomed, each sunk to its own depth, warmed on top by the fire.
  const stones = (
    [
      [-44, 2, 10, 1],
      [-18, 9, 7, 0],
      [20, 9, 8, 0],
      [40, 3, 9, 1],
    ] as const
  ).map(([dx, dy, w, near], i) => {
    const sink = 0.35 + 0.3 * rand()
    const pts = loop(330 + i, 0.14, 0, 20)
      .map(([x, y]) => [fx + dx + x * w, fg + dy + y * w * 0.55] as const)
      .map(([x, y]) => [x, Math.min(y, fg + dy + w * 0.55 * sink)] as const)
    return {
      near,
      svg: shape(pts, mix(grey(0.26), tone.ember, 0.16), pen(340 + i, 2.2), grey(0.7)),
    }
  })
  const ring = (near: 0 | 1) =>
    stones
      .filter(s => s.near === near)
      .map(s => s.svg)
      .join('')

  // A log: a bar from a to b, `w` across, its end grain an oval with one ring at the nearer end.
  const log = (a: Pt, b: Pt, w: number, seed: number) => {
    const [dx, dy] = [b[0] - a[0], b[1] - a[1]]
    const len = Math.hypot(dx, dy)
    const [nx, ny] = [(-dy / len) * w * 0.5, (dx / len) * w * 0.5]
    const body = hand(
      [
        [a[0] + nx, a[1] + ny],
        [b[0] + nx, b[1] + ny],
        [b[0] - nx, b[1] - ny],
        [a[0] - nx, a[1] - ny],
        [a[0] + nx, a[1] + ny],
      ],
      6,
      0.5,
      seed,
    )
    const grain = loop(seed + 1, 0.08, 0, 18).map(
      ([x, y]) => [a[0] + x * w * 0.34, a[1] + y * w * 0.52] as const,
    )
    const ring = stroke(
      grain
        .slice(3, 14)
        .map(([x, y]) => [a[0] + (x - a[0]) * 0.45, a[1] + (y - a[1]) * 0.45] as const),
      pen(seed + 2, 1.2),
      tone.moonlight,
    )
    return `${shape(body, grey(0.2), pen(seed), tone.moonlight)}${shape(grain, grey(0.36), pen(seed + 3, 1.8), tone.moonlight)}${ring}`
  }
  const logs = `${log([fx + 30, fg - 11], [fx - 42, fg + 3], 11, 101)}${log([fx - 26, fg - 14], [fx + 42, fg + 2], 10, 104)}`

  // One flame: a silhouette of three tongues, the tallest in the middle leaning a little right, drawn
  // in one closed line.
  const flank = (a: Pt, c: Pt, b: Pt, n: number) => curve(a, c, b, n).slice(1)
  const foot = fg - 3
  const silhouette = hand(
    [
      [fx - 19, foot],
      ...flank([fx - 19, foot], [fx - 26, foot - 20], [fx - 13, foot - 33], 8),
      ...flank([fx - 13, foot - 33], [fx - 9, foot - 24], [fx - 6, foot - 25], 3),
      ...flank([fx - 6, foot - 25], [fx - 10, foot - 50], [fx + 3, foot - 64], 10),
      ...flank([fx + 3, foot - 64], [fx + 13, foot - 44], [fx + 9, foot - 30], 8),
      ...flank([fx + 9, foot - 30], [fx + 17, foot - 33], [fx + 16, foot - 42], 3),
      ...flank([fx + 16, foot - 42], [fx + 26, foot - 22], [fx + 19, foot], 8),
    ],
    1000,
    0.7,
    110,
  )
  const heart = hand(
    [
      [fx - 7, foot],
      ...flank([fx - 7, foot], [fx - 11, foot - 20], [fx + 1, foot - 36], 8),
      ...flank([fx + 1, foot - 36], [fx + 11, foot - 20], [fx + 8, foot], 8),
    ],
    1000,
    0.6,
    112,
  )
  const flame = `${shape(silhouette, tone.ember, pen(111), tone.moonlight)}<path d="${poly(heart)}" fill="${tone.pink}"/>`

  const sparks = (
    [
      [fx + 18, fg - 78, 2],
      [fx - 8, fg - 100, 1.6],
    ] as const
  )
    .map(([x, y, r]) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${tone.ember}"/>`)
    .join('')
  const smoke = `<g opacity="0.2">${stroke(
    hand(curve([fx - 6, fg - 70], [fx - 38, fg - 100], [fx - 22, fg - 136], 14), 1000, 1.6, 120),
    t => 3.4 * (1 - t) ** 0.7 + 0.3,
    tone.moonlight,
  )}</g>`

  return { glow, back: ring(0), body: `${logs}${flame}${sparks}${smoke}`, front: ring(1) }
}
