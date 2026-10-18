import type { Landscape, Point, Rise, RiseKind } from './landscape'
import { bayerThreshold } from '../../util/bayer'
import { trailRandom } from './random'

export type Face = {
  x: number
  y: number
  brow: number
  span: number
  tall: number
  turn: number
  form: RiseKind | null
}

// One face per crest column of each landform, from its crest down to its foot. `turn` > 0 where
// the crest drops to the right, so the face looks away from the light.
export function contourFaces(landscape: Landscape): Face[][] {
  return landscape.rises.map(({ crest }) => {
    const yAt = (k: number) => crest[Math.max(0, Math.min(crest.length - 1, k))].y
    return crest.map(({ x, y, brow, foot, form }, k) => ({
      x,
      y,
      brow,
      span: 4,
      tall: foot - y,
      form,
      // The slope over a few columns, so one jagged step does not flip a column's shade.
      turn: Math.max(-0.6, Math.min(0.6, (yAt(k + 6) - yAt(k - 6)) / 48)),
    }))
  })
}

// The nearest face whose slope contains the point, with how far below its crest the point lies.
export function faceAt(faces: readonly Face[][], point: Point) {
  let best: (Face & { below: number }) | null = null
  // contourFaces retains the landscape's contiguous four-pixel columns, starting at x = 0.
  const column = Math.floor(point.x / 4)
  for (const contour of faces) {
    const face = contour[column]
    if (!face || point.x < face.x || point.x >= face.x + face.span) continue
    const below = point.y - face.y
    if (below >= 0 && below < face.tall && (!best || below < best.below)) best = { ...face, below }
  }
  return best
}

// Whether a point stands on the steep lower slope of a landform, between its brow and its foot.
// Trees root on the tops and at the feet, never halfway down a face.
export function onSteepGround(faces: readonly Face[][], point: Point) {
  const face = faceAt(faces, point)
  return face !== null && face.below >= face.brow - face.y && face.below < face.tall - 1
}

const tones = [
  'top',
  'face-lit',
  'face',
  'face-dark',
  'shadow',
  'outline',
  'crest',
  'rock',
  'rock-lit',
  'lip',
  'strata',
] as const
type Tone = (typeof tones)[number]
// The ramp a slope is dithered along, light to dark.
const slope: readonly Tone[] = ['top', 'face-lit', 'face', 'face-dark']
const RAMP = 1
const fixed = (tone: Tone) => tones.indexOf(tone) + 2

const smooth = (a: number, b: number, t: number) => {
  const x = Math.max(0, Math.min(1, (t - a) / (b - a)))
  return x * x * (3 - 2 * x)
}
// Which way a spur's flank turns at a phase, +1 to the right: sharp over the spur's crest, eased
// across the gully floor between two spurs.
const flank = (u: number) => {
  const f = u - Math.floor(u)
  return f < 0.5 ? 1 - 2 * smooth(0.4, 0.6, f) : 2 * smooth(0.4, 0.6, 1 - f) - 1
}
export const hash = (a: number, b: number, seed: number) => {
  let h = Math.imul(a + 1, 374761393) ^ Math.imul(b + 1, 668265263) ^ seed
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}
// How far past the foot a landform's shadow reaches.
const shade = (tall: number) => Math.min(16, Math.round(tall / 4) * 2)

// One landform on a 4 × 2 cell grid. Hills and ridges are domes: the shade deepens from the crest
// to the foot, the flank turned from the light is darker, spurs fan down the slope, and the back
// of the swell dissolves into the meadow. Buttes and knobs keep a flat top and sheer sides. Every
// landform throws a shadow past its foot, leaning right because the light comes from the upper
// left. The cells are then laid as strokes along each row: a stroke runs a few cells and picks its
// ramp stop once, as the 404 landscape brushes its ranges, so a slope shows as short strokes
// along its contours instead of a halftone screen.
function paintRise(crest: Rise['crest'], faces: readonly Face[], seed: number) {
  const live = faces.filter(face => face.tall > 0)
  if (live.length === 0) return ''
  const top = Math.min(...live.map(face => face.y))
  const bottom = Math.max(...live.map(face => face.y + face.tall + shade(face.tall)))
  const rows = (bottom - top) / 2 + 1
  const columns = crest.length
  const kind = new Uint8Array(columns * rows)
  const value = new Float32Array(columns * rows)
  const cover = new Float32Array(columns * rows).fill(1)
  const set = (k: number, y: number, tone: Tone | typeof RAMP, level = 0, coverage = 1) => {
    if (k < 0 || k >= columns || y < top || y > bottom) return
    const index = k * rows + (y - top) / 2
    if (tone === RAMP) {
      kind[index] = RAMP
      value[index] = level
    } else if (kind[index] === 0 || coverage === 1) kind[index] = fixed(tone)
    else return
    cover[index] = coverage
  }
  const phase = (seed % 1000) / 1000
  faces.forEach((face, k) => {
    const { x, y, brow, tall, turn, form } = face
    if (tall <= 0) return
    const side = Math.max(-1, Math.min(1, turn / 0.25))
    const column = x / 4
    const sheer = form === 'butte' || form === 'knob'
    const period = form === 'ridge' ? 44 : 30
    for (let at = y; at < y + tall; at += 2) {
      const row = at / 2
      const fall = (at - y) / tall
      const grain = (hash(column, row, seed) - 0.5) * 0.12
      if (sheer) {
        if (at < brow) {
          // The crest is a broken outline, in strokes three cells long.
          set(k, at, at === y && hash(Math.floor(k / 3), 7, seed) < 0.6 ? 'outline' : 'top')
          continue
        }
        if (at === brow) {
          set(k, at, 'crest')
          continue
        }
        const down = (at - brow) / Math.max(2, tall - (brow - y))
        if (form === 'butte' && down < 0.6 && ((at - brow) / 2) % 3 === 2) {
          set(k, at, bayerThreshold(column, Math.floor(row / 3)) < 0.9 ? 'strata' : 'face')
          continue
        }
        if (form === 'knob' && down < 0.55) {
          set(k, at, hash(column, row, seed) < 0.18 ? 'lip' : side < -0.25 ? 'rock-lit' : 'rock')
          continue
        }
        set(k, at, RAMP, 0.42 + 0.4 * down + side * 0.22 + grain)
        continue
      }
      // Spurs leave the crest square to it and lean downhill with the flank.
      const u = (x - side * 0.35 * (at - y)) / period + Math.sin(x / 97 + phase * 6.283) * 0.4
      const spur = tall >= 14 ? flank(u) * 0.14 * smooth(0.2, 0.6, fall) : 0
      const tone = 0.1 + 0.62 * fall ** 1.35 + side * 0.18 * smooth(0.15, 0.6, fall) + spur + grain
      set(k, at, RAMP, tone, fall < 0.2 ? fall / 0.2 : 1)
    }
  })
  faces.forEach((face, k) => {
    if (face.tall <= 0) return
    const reach = shade(face.tall)
    const lean = Math.min(3, Math.floor(face.tall / 14))
    const foot = face.y + face.tall
    for (let r = 0; r < reach; r += 2) {
      const t = r / reach
      set(k + Math.floor(t * lean), foot + r, 'shadow', 0, 0.6 * (1 - t) ** 1.3)
    }
  })
  const runs = new Map<Tone, string[]>()
  for (let row = 0; row < rows; row++) {
    const y = top + row * 2
    let current = 0
    let start = 0
    let end = -1
    let threshold = 0
    for (let k = 0; k <= columns; k++) {
      let tone = 0
      if (k < columns) {
        if (k >= end) {
          threshold = hash(k, row, seed ^ 0x5bd1e995)
          end = k + 2 + Math.floor(hash(k, row, seed ^ 0x1b873593) * 6)
        }
        const index = k * rows + row
        const painted = kind[index]
        if (painted === 0 || (cover[index] < 1 && threshold > cover[index])) tone = 0
        else if (painted === RAMP) {
          const p = Math.max(0, Math.min(0.9999, value[index])) * (slope.length - 1)
          const i = Math.floor(p)
          tone = tones.indexOf(slope[p - i > threshold ? i + 1 : i]) + 1
        } else tone = painted - 1
      }
      if (tone === current) continue
      if (current) {
        const list = runs.get(tones[current - 1]) ?? []
        list.push(`M${start * 4} ${y}h${(k - start) * 4}v2h-${(k - start) * 4}Z`)
        runs.set(tones[current - 1], list)
      }
      current = tone
      start = k
    }
  }
  return tones
    .filter(tone => runs.has(tone))
    .map(tone => `<path class="trail-ravine-${tone}" d="${runs.get(tone)?.join('')}"/>`)
    .join('')
}

// A few boulders rest on the taller faces.
function boulders(crest: Rise['crest'], random: () => number) {
  const runs: string[] = []
  const run = (tone: Tone, x: number, y: number, width: number, height = 2) =>
    runs.push(`<path class="trail-ravine-${tone}" d="M${x} ${y}h${width}v${height}h${-width}Z"/>`)
  for (let boulder = 0; boulder < 4; boulder++) {
    const column = crest[Math.floor(random() * crest.length)]
    const tall = column.foot - column.y
    if (tall < 24 || !column.form) continue
    const size = random() < 0.5 ? 8 : 6
    const x = column.x
    const y = Math.round((column.y + 8 + random() * (tall - 16)) / 2) * 2
    run('rock-lit', x + 2, y, size - 4)
    run('rock', x, y + 2, size, 2)
    run('rock', x + size / 2, y, size / 2)
    run('lip', x, y + 4, size)
  }
  return runs.join('')
}

// Landforms paint back to front, each in its own cell grid, so a nearer hill covers the shadow of
// one behind it.
export function trailRelief(landscape: Landscape, seed: number) {
  const random = trailRandom(seed ^ 0x0b4c0f27)
  const faces = contourFaces(landscape)
  const forms = landscape.rises
    .map(
      ({ crest }, index) =>
        paintRise(crest, faces[index], (seed ^ Math.imul(index + 1, 0x9e3779b9)) >>> 0) +
        boulders(crest, random),
    )
    .join('')
  return `<g class="trail-relief" aria-hidden="true">${forms}</g>`
}
