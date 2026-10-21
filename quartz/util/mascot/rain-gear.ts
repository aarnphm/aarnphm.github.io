import {
  curve,
  even,
  filled,
  hand,
  INK,
  PAPER,
  PINK,
  poly,
  pressure,
  rounded,
  SAGE,
  stroke,
  type Pt,
} from './pen'

// Rain: a little umbrella held up in the paw on its right, the canopy tipped forward over its head,
// pink with sage panels between the ribs and a scalloped hem. `held` has it open and, hidden, blown
// inside out for storms. `rain` is two drawings of the streaks falling past it and the drips off the
// rib tips, to swap between.
export function umbrella(w: number, paper = PAPER, ink = INK, cloth = PINK, panel = SAGE) {
  const tilt = 0.2
  const hold: Pt = [0.84, 0.2]
  const up: Pt = [-Math.sin(tilt), -Math.cos(tilt)]
  const along: Pt = [Math.cos(tilt), -Math.sin(tilt)]
  const centre: Pt = [hold[0] + 1.9 * up[0], hold[1] + 1.9 * up[1]]
  const half = 0.74
  const dome = 0.4
  // Canopy points: u across the hem (-1 to 1), v up the shaft.
  const at = (u: number, v: number): Pt => [
    centre[0] + u * half * along[0] + v * up[0],
    centre[1] + u * half * along[1] + v * up[1],
  ]
  const tips = [-1, -0.5, 0, 0.5, 1]
  const rib = (u: number) =>
    curve([u, 0], [u * 0.92, dome * 0.86], [0, dome], 12).map(([a, b]) => at(a, b))
  const hem = (u0: number, u1: number) =>
    Array.from({ length: 9 }, (_, i) => {
      const t = i / 8
      return at(u0 + (u1 - u0) * t, 0.08 * Math.sin(Math.PI * t))
    })
  const panelShape = (k: number) => [
    ...hem(tips[k], tips[k + 1]),
    ...rib(tips[k + 1]).slice(1),
    ...rib(tips[k]).reverse(),
  ]
  const canopy = hand(
    [
      ...tips.slice(0, -1).flatMap((u, k) => hem(u, tips[k + 1]).slice(k ? 1 : 0)),
      ...rib(1).slice(1),
      ...rib(-1).reverse(),
    ],
    0.03,
    0.008,
    801,
  )
  const panels = [1, 3]
    .map(
      k => `<path d="${poly(hand(panelShape(k), 0.03, 0.006, 802 + k))}" style="fill:${panel}"/>`,
    )
    .join('')
  const ribs = [-0.5, 0, 0.5]
    .map((u, i) => stroke(hand(rib(u), 0.03, 0.006, 806 + i), () => w * 0.5, INK))
    .join('')
  const foot: Pt = [hold[0] - 0.22 * up[0], hold[1] - 0.22 * up[1]]
  const shaft = stroke(hand([centre, foot], 0.03, 0.006, 810), pressure(811, w * 0.9, 0.2), ink)
  const crook = stroke(
    curve(foot, [foot[0] - 0.04, foot[1] + 0.2], [foot[0] - 0.2, foot[1] + 0.06], 10),
    pressure(812, w * 0.9, 0.2),
    ink,
  )
  const top = at(0, dome)
  const ferrule = stroke([top, [top[0] + 0.1 * up[0], top[1] + 0.1 * up[1]]], () => w * 0.9, ink)
  const paw = filled(
    hand(rounded(hold[0], hold[1], 0.24, 0.2, 0.09), 0.02, 0.006, 813),
    paper,
    pressure(814, w, 0.2),
    ink,
  )
  const open =
    `<path d="${poly(canopy)}" style="fill:${cloth}"/>${panels}${ribs}` +
    `${stroke(canopy, pressure(815, w, 0.3), ink)}${ferrule}`

  // Inside out in a storm: the ribs bent up past the top of the shaft into a cup open to the sky, the
  // bare wire tips poking past the hem, and the end rib on the right snapped so its corner droops.
  // Two drawings of the cloth flapping, to swap between.
  const shaftTop = stroke(hand([centre, at(0, dome + 0.04)], 0.03, 0.004, 830), () => w * 0.9, ink)
  const inverted = [0, 1].map(k => {
    const tip = (u: number): Pt =>
      u === 1
        ? [1, dome + 0.06 + 0.05 * k]
        : [u, dome + 0.52 + 0.07 * (k ? u : -u) - 0.13 * Math.sqrt(1 - u * u)]
    const bend = (u: number): Pt => [tip(u)[0] * 0.62, dome + 0.05]
    const rib = (u: number) => curve([0, dome], bend(u), tip(u), 10)
    const rim = (u0: number, u1: number) => {
      const [a, b] = [tip(u0), tip(u1)]
      return Array.from({ length: 9 }, (_, i): Pt => {
        const t = i / 8
        const dip = (0.07 + 0.03 * k) * Math.sin(Math.PI * t)
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t - dip]
      })
    }
    // The far side of the hem, seen over the near one: the inside of the cup.
    const far = Array.from({ length: 17 }, (_, i): Pt => {
      const t = i / 16
      const [a, b] = [tip(-1), tip(1)]
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t + 0.14 * Math.sin(Math.PI * t)]
    })
    const near = tips.slice(0, -1).flatMap((u, j) => rim(u, tips[j + 1]).slice(j ? 1 : 0))
    const xy = (pts: readonly Pt[]) => pts.map(([a, b]) => at(a, b))
    const inside = hand(xy([...near, ...far.slice().reverse()]), 0.03, 0.006, 832 + k)
    const outside = hand(
      xy([...rib(-1), ...near.slice(1), ...rib(1).reverse().slice(1)]),
      0.03,
      0.008,
      834 + k,
    )
    const cups = [1, 3]
      .map(j => {
        const shape = [
          ...rim(tips[j], tips[j + 1]),
          ...rib(tips[j + 1])
            .reverse()
            .slice(1),
          ...rib(tips[j]).slice(1),
        ]
        return `<path d="${poly(hand(xy(shape), 0.03, 0.006, 836 + j + k))}" style="fill:${panel}"/>`
      })
      .join('')
    const wires = [-0.5, 0, 0.5]
      .map((u, i) => stroke(hand(xy(rib(u)), 0.03, 0.006, 840 + i + k), () => w * 0.5, INK))
      .join('')
    const bare = tips
      .map(u => {
        const [t, b] = [tip(u), bend(u)]
        const len = Math.hypot(t[0] - b[0], t[1] - b[1])
        const end: Pt = [t[0] + ((t[0] - b[0]) / len) * 0.1, t[1] + ((t[1] - b[1]) / len) * 0.1]
        return stroke(xy([t, end]), () => w * 0.6, ink)
      })
      .join('')
    return (
      `<path d="${poly(inside)}" style="fill:color-mix(in srgb, ${cloth} 80%, ${ink})"/>` +
      `${stroke(hand(xy(far), 0.03, 0.006, 846 + k), pressure(847, w * 0.8, 0.3), ink)}` +
      `<path d="${poly(outside)}" style="fill:${cloth}"/>${cups}${wires}` +
      `${stroke(outside, pressure(848 + k, w, 0.3), ink)}${bare}`
    )
  })
  const held =
    `<g class="pet-brolly">${shaft}${crook}<g class="pet-brolly-open">${open}</g>` +
    `<g class="pet-brolly-out">${shaftTop}${inverted.map(f => `<g class="pet-out-frame">${f}</g>`).join('')}</g>` +
    `${paw}</g>`

  // Streaks slanting a little, clear of the canopy, the face and the breath; each drawing has them
  // a step further down, and a drop hanging off each end rib that falls the next.
  const streaks = [
    [-1.3, -1.5],
    [-0.85, -2.05],
    [-1.55, -0.85],
    [1.62, -1.35],
    [1.68, -0.6],
    [0.95, -2.3],
  ] as const
  const rain = [0, 1].map(k => {
    const lines = streaks
      .map(([x, y]) => {
        const [sx, sy] = [x - 0.04 * k, y + 0.24 * k]
        return stroke(
          [
            [sx, sy],
            [sx - 0.05, sy + 0.18],
          ],
          even(w * 0.55),
          ink,
        )
      })
      .join('')
    const drips = [at(-1, 0), at(1, 0)]
      .map(([x, y]) => {
        const d = 0.1 + 0.2 * k
        return filled(
          hand(rounded(x, y + d, 0.06, 0.09, 0.03), 0.02, 0.004, 820 + k),
          paper,
          () => w * 0.45,
          ink,
        )
      })
      .join('')
    return `${lines}${drips}`
  })
  return { held, rain }
}
