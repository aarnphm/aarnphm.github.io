import type { Masonry, Print } from './masonry'
import { GATE, TILT } from '../404-gate'
import { paintIvy } from '../404-overgrowth'
import { col, INK, mix, put, putMix, row, type Grid, type RGB } from '../404-pixel'
import { FIRE } from './camp'
import { CREAM, FLAME, LUME, RIFT, ROSE } from './light-marks'

// Hand angles in degrees, clockwise from twelve.
export type Dial = { hour: number; minute: number; second: number }

// How the little people are lit: by day from the sun's side, at night by the campfire when they
// are near it and otherwise by the open gate.
export type Rim = { side: number; color: RGB; contrast: number; night: boolean; fire: boolean }

// Rim light and form shadow a pixel deep: an edge facing the light takes its colour, the edge
// away from it darkens, so a figure four pixels wide reads as round. Pixel-wide limbs and inked
// pixels keep their colour.
export function relight(d: Uint8ClampedArray, g: Grid, rim: Rim, heat: number) {
  const { bw, bh, sx } = g
  const open = (p: number) => d[p * 4 + 3] !== 255
  for (let j = 1; j < bh; j++)
    for (let i = 1; i < bw - 1; i++) {
      const p = j * bw + i
      const k = p * 4
      if (open(p) || d[k] + d[k + 1] + d[k + 2] < 90) continue
      let { side, color } = rim
      let strength = 0.4 * rim.contrast
      if (rim.night) {
        const x = sx[i]
        const fire = rim.fire && Math.abs(x - FIRE.x) < 170
        side = Math.sign((fire ? FIRE.x : GATE.x) - x) || 1
        color = fire ? FLAME[1] : ROSE
        strength = fire ? 0.45 : 0.15 + 0.3 * heat
      }
      const toward = open(p + side)
      const away = open(p - side)
      if (toward && away) continue
      if (toward || (open(p - bw) && !away)) putMix(d, k, d[k], d[k + 1], d[k + 2], color, strength)
      else if (away) putMix(d, k, d[k], d[k + 1], d[k + 2], INK, 0.28 * rim.contrast)
    }
}

// Rose light from the open gate falls on the stones under the opening and the tablet below as a
// flat wash, one step weaker for every band further down: multiplied into the stone by day,
// screened over it by night.
export function paintSpill(d: Uint8ClampedArray, m: Masonry, glow: number, dark: boolean) {
  if (glow <= 0) return
  const rose = m.print.rose[2]
  for (let s = 0; s < m.spill.length; s++) {
    const k = m.spill[s] * 4
    const a = glow * (0.5 - 0.1 * m.bands[s])
    for (let c = 0; c < 3; c++) {
      const lit = dark
        ? 255 - ((255 - d[k + c]) * (255 - rose[c])) / 255
        : (d[k + c] * rose[c]) / 255
      d[k + c] += (lit - d[k + c]) * a
    }
  }
}

const HUB = 15

// Station-clock hands on the pole star's arbor, laid into the rift pixel by pixel. The hour and
// minute hands are bars lit on their sunward edge and keylined in the rift's deepest ink, which
// holds them apart from the rose arms of the nebula; the red second hand is a pixel wide with its
// disc; the hub is an open ring so the star shows through. The rift is always night, so by day the
// hands take the stone's light face. After sunset the stone greys with the rest of the ruin, so
// `lume` (0 by day, 1 at night) carries the hands over to lume instead.
export function paintHands(d: Uint8ClampedArray, g: Grid, dial: Dial, print: Print, lume: number) {
  const { bw, bh, sx, sy } = g
  const face = mix(print.stone[3], mix(CREAM, LUME, 0.45), lume)
  const shade = mix(print.stone[2], LUME, lume)
  const key = RIFT[0]
  const red = print.rose[lume > 0.5 ? 3 : 2]
  const bars = [
    { deg: dial.hour, w: 14, len: 112, tail: 40, lit: face },
    { deg: dial.minute, w: 10, len: 150, tail: 44, lit: face },
    { deg: dial.second, w: 5, len: 102, tail: 52, lit: red },
  ].map(b => {
    // The hands turn on the ring's own dial, so they lean with it.
    const a = ((b.deg + TILT) * Math.PI) / 180
    const ux = Math.sin(a)
    const uy = -Math.cos(a)
    // Which side of the bar faces the sun, up and to the right.
    const sun = -uy * 0.6 - ux * 0.8 >= 0 ? 1 : -1
    const half = Math.max(b.w / 2, g.px / 2)
    const edged = b.w > g.px * 2
    // The keyline rings the broad hands a pixel out, ends included.
    return { ...b, ux, uy, sun, half, edged, rim: edged ? g.px : 0 }
  })
  const disc = [GATE.x + bars[2].ux * 114, GATE.y + bars[2].uy * 114]
  const i0 = Math.max(0, col(g, GATE.x - 160))
  const i1 = Math.min(bw - 1, col(g, GATE.x + 160))
  const j0 = Math.max(0, row(g, GATE.y - 160))
  const j1 = Math.min(bh - 1, row(g, GATE.y + 160))
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      const dx = sx[i] - GATE.x
      const dy = sy[j] - GATE.y
      let c: RGB | null = null
      for (const b of bars) {
        const u = dx * b.ux + dy * b.uy
        if (u > b.len + b.rim || u < -b.tail - b.rim || Math.abs(u) < HUB - b.rim) continue
        const v = dy * b.ux - dx * b.uy
        if (Math.abs(v) > b.half + b.rim) continue
        // Later bars lie over earlier ones, so a keyline only paints where nothing is yet.
        if (Math.abs(v) > b.half || u > b.len || u < -b.tail || Math.abs(u) < HUB) c ??= key
        else c = b.edged && v * b.sun < g.px - b.half ? (b.lit === red ? red : shade) : b.lit
      }
      if (Math.hypot(sx[i] - disc[0], sy[j] - disc[1]) <= 12) c = red
      if (Math.abs(Math.hypot(dx, dy) - HUB) <= g.px / 2) c = face
      if (c) put(d, (j * bw + i) * 4, c)
    }
  paintIvy(d, g, bars[0].ux, bars[0].uy, bars[0].half, () => true)
}
