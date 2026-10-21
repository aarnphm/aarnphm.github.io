import { frac, hex, type RGB } from '../404-pixel'

// A broken ring of pixels, the dashes turning with `turn`: the halos of the Saint-Rémy stars.
export function halo(
  plot: (x: number, y: number, c: RGB, alpha: number) => void,
  cx: number,
  cy: number,
  radius: number,
  step: number,
  c: RGB,
  alpha: number,
  turn: number,
  dashes: number,
) {
  const n = Math.ceil((2 * Math.PI * radius) / step)
  for (let k = 0; k < n; k++) {
    const u = k / n
    if (frac(u * dashes + turn) < 0.38) continue
    plot(cx + Math.cos(u * 2 * Math.PI) * radius, cy + Math.sin(u * 2 * Math.PI) * radius, c, alpha)
  }
}

export const CROSS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
]

// Lume, in the sage of the site's palette: Super-LumiNova's C3 glows yellow-green, and it is what
// the hands turn to once the daylight goes.
export const LUME = hex('#cdd597')

// The rift through the gate, always night whatever the hour: a whirlpool of brush strokes
// under a ringed pole star, which is also the arbor the clock hands turn on. Motes orbit with
// differential rotation and lay their strokes back along their own paths; hovering opens an inflow,
// so the strokes lengthen into spirals and echoes of the dial sink into the pole. Comets fall past
// on Kepler arcs.
export const RIFT = [
  '#0b0b0e',
  '#10131f',
  '#172036',
  '#22305a',
  '#34427a',
  '#5d5a8c',
  '#9a7094',
  '#e3a19a',
].map(hex)

export const CREAM = hex('#fffcf0')
export const GOLDEN = hex('#f1d67e')
export const ROSE = hex('#e3a19a')
export const FLAME = ['#fffcf0', '#f1d67e', '#da702c', '#af3029'].map(hex)
