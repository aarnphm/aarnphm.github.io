import type { Landscape, Point, Rise, RiseKind } from './landscape'
import { trailRandom } from './random'

export type Face = {
  x: number
  y: number
  brow: number
  span: number
  tall: number
  turn: number
  form: RiseKind | null
  // The first row below this face's landform that its lifted summit does not hide; see
  // `paintRise`, which may hide more, never less.
  hidden: number
}

// The rows behind a landform's crest that its lifted summit hides, from the summit's height.
const summitLift = (tall: number) => Math.max(2, Math.round(tall / 4) * 2)

// One face per crest column of each landform, from its crest down to its foot. `turn` > 0 where
// the crest drops to the right, so the face looks away from the light. Each run of raised columns
// is one landform, and every face in it knows the row where its summit's lift ends.
export function contourFaces(landscape: Landscape): Face[][] {
  return landscape.rises.map(({ crest }) => {
    const yAt = (k: number) => crest[Math.max(0, Math.min(crest.length - 1, k))].y
    const faces = crest.map(({ x, y, brow, foot, form }, k) => ({
      x,
      y,
      brow,
      span: 4,
      tall: foot - y,
      form,
      // The slope over a few columns, so one jagged step does not flip a column's shade.
      turn: Math.max(-0.6, Math.min(0.6, (yAt(k + 6) - yAt(k - 6)) / 48)),
      hidden: y,
    }))
    let from = -1
    for (let k = 0; k <= faces.length; k++) {
      const raised = k < faces.length && faces[k].tall > 0
      if (raised && from < 0) from = k
      if (raised || from < 0) continue
      const summit = faces.slice(from, k).reduce((top, face) => (face.y < top.y ? face : top))
      const hidden = summit.y + summitLift(summit.tall)
      for (let at = from; at < k; at++) faces[at].hidden = hidden
      from = -1
    }
    return faces
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

// Whether a point stands on the meadow just behind a crest, where only what rises above the crest
// line shows. Trees planted there read as a stand on the far side of the hill.
export function behindCrest(faces: readonly Face[][], point: Point) {
  const face = faceAt(faces, point)
  return face !== null && point.y < face.hidden
}

const tones = [
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
const bodies = ['top', 'ground'] as const

const hash = (a: number, b: number, seed: number) => {
  let h = Math.imul(a + 1, 374761393) ^ Math.imul(b + 1, 668265263) ^ seed
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

// Rows of one cell grid merged into horizontal runs, one path per tone.
function runs<Name extends string>(
  grid: Uint8Array,
  names: readonly Name[],
  columns: number,
  rows: number,
  top: number,
  left: number,
  prefix: string,
) {
  const paths = new Map<Name, string[]>()
  for (let row = 0; row < rows; row++) {
    const y = top + row * 2
    let current = 0
    let start = 0
    for (let k = 0; k <= columns; k++) {
      const tone = k < columns ? grid[k * rows + row] : 0
      if (tone === current) continue
      if (current) {
        const list = paths.get(names[current - 1]) ?? []
        list.push(`M${(left + start) * 4} ${y}h${(k - start) * 4}v2h-${(k - start) * 4}Z`)
        paths.set(names[current - 1], list)
      }
      current = tone
      start = k
    }
  }
  return names
    .filter(name => paths.has(name))
    .map(name => `<path class="${prefix}${name}" d="${paths.get(name)?.join('')}"/>`)
    .join('')
}

// Ledge planes fade with distance, the way a painter compresses the values of the far ground and
// keeps the darkest darks for the foreground. Indexed by contour, far to near.
export const reliefStrengths = [0.7, 0.8, 0.9, 1]

// One landform on the scene's 4 × 2 cell grid, in the scene's own language of flat planes: the
// slope is cut into ledges, as a top-down pixel map steps its terrain. Each ledge is a rim along
// its top edge, a riser turned toward or away from the light, and a line of shadow on the plateau
// under it; the plateaus between ledges are the meadow itself. Hills and ridges carry a ledge
// every `step` pixels of height, so the ledges run around the swell like contours. Buttes and
// knobs are one sheer riser from brow to foot, banded or rocky, under a hard outline.
//
// The landform stands up from the ground: its summit is lifted by the height of its top surface,
// so the strip of meadow behind the crest, as many rows as that lift, lies hidden behind it. The
// group paints that strip opaque (the lit top, then the field's own tone below the rim) and joins
// the depth order at the strip's last row, so trees, walkers and the far wood rooted in the strip
// show only above the crest, and whatever stands on the plateaus below paints over the ledges.
function paintRise(faces: readonly Face[], left: number, seed: number, level: number) {
  // A swell too low or too narrow for a ledge is left to the meadow.
  if (faces.length < 4 || Math.max(...faces.map(face => face.tall)) < 8) return ''
  const top = Math.min(...faces.map(face => face.y))
  const bottom = Math.max(...faces.map(face => face.y + face.tall)) + 2
  const rows = (bottom - top) / 2 + 1
  const columns = faces.length
  const grid = new Uint8Array(columns * rows)
  const body = new Uint8Array(columns * rows)
  const set = (k: number, y: number, tone: Tone) => {
    if (k < 0 || k >= columns || y < top || y > bottom) return
    grid[k * rows + (y - top) / 2] = tones.indexOf(tone) + 1
  }
  // Where each column's top surface ends: the rim of its highest ledge, or the brow.
  const capFoot: number[] = []
  faces.forEach((face, k) => {
    const { y, brow, tall, turn, form } = face
    const foot = y + tall
    const riser: Tone = turn < -0.1 ? 'face-lit' : turn > 0.1 ? 'face-dark' : 'face'
    const column = face.x / 4
    if (form === 'butte' || form === 'knob') {
      set(k, y, 'outline')
      set(k, brow, 'crest')
      for (let at = brow + 2; at < foot; at += 2) {
        const down = (at - brow) / Math.max(2, foot - brow)
        if (form === 'butte' && down < 0.6 && ((at - brow) / 2) % 3 === 0) set(k, at, 'strata')
        else if (form === 'knob' && down < 0.55)
          set(k, at, hash(column, at / 2, seed) < 0.18 ? 'lip' : turn < -0.1 ? 'rock-lit' : 'rock')
        else set(k, at, riser)
      }
      set(k, foot, 'shadow')
      capFoot[k] = brow
      return
    }
    // Taller swells get taller risers; the plateau between ledges stays eight pixels at the
    // foot. The swell is a dome, so a ledge's row comes from the height it marks: near the summit
    // the ledges spread, and toward the flanks each one climbs to the crest line and ends there,
    // instead of being cut off where the swell runs out of height.
    const height = tall >= 36 ? 6 : 4
    const step = height + 8
    let above = Infinity
    let rim = foot
    for (let level = 0; level * step < tall; level++) {
      const base = y + Math.round((tall * Math.sqrt(1 - (level * step) / tall)) / 2) * 2
      if (above - base < height + 2 || base - height < y) continue
      above = base - height
      rim = above
      set(k, above, 'crest')
      for (let at = above + 2; at < base; at += 2) set(k, at, riser)
      set(k, base, 'shadow')
    }
    capFoot[k] = rim
  })
  // The summit stands at least half the silhouette's height above the meadow behind it; a dome
  // whose top rim sits lower than that lifts to the rim, so the band ends on a ledge.
  const summit = faces.findIndex(face => face.y === top)
  const lift = Math.max(capFoot[summit] - top, faces[summit].hidden - top)
  const hidden = top + lift
  faces.forEach((face, k) => {
    const cap = Math.min(capFoot[k], hidden)
    const foot = Math.min(face.y + face.tall + 2, hidden)
    for (let at = face.y; at < foot; at += 2) body[k * rows + (at - top) / 2] = at < cap ? 1 : 2
  })
  // The ground behind a landform is the field of the terrace above its contour. The first contour
  // has none above it: the valley floor there wears the first field (`.trail-valley-field`).
  const field = `var(--trail-field-${Math.max(0, level - 1)})`
  return `<g class="trail-landform" data-ground-y="${hidden - 1}" style="--trail-landform-field:${field};--trail-relief-strength:${reliefStrengths[level] ?? 1}">${runs(body, bodies, columns, rows, top, left, 'trail-landform-')}${runs(grid, tones, columns, rows, top, left, 'trail-ravine-')}`
}

// A few boulders rest on the taller faces.
function boulders(crest: Rise['crest'], random: () => number) {
  const marks: string[] = []
  const run = (tone: Tone, x: number, y: number, width: number, height = 2) =>
    marks.push(`<path class="trail-ravine-${tone}" d="M${x} ${y}h${width}v${height}h${-width}Z"/>`)
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
  return marks.join('')
}

// Each run of raised columns along a contour is one landform and one group, painted back to front
// so a nearer hill covers the shadow of one behind it. In the page the depth order lifts every
// group out of here and sorts it among the trees and walkers; see `groundObjectSelector`.
export function trailRelief(landscape: Landscape, seed: number) {
  const random = trailRandom(seed ^ 0x0b4c0f27)
  const faces = contourFaces(landscape)
  const forms = landscape.rises
    .map(({ crest }, index) => {
      const contour = faces[index]
      const groups: string[] = []
      let from = -1
      for (let k = 0; k <= contour.length; k++) {
        const raised = k < contour.length && contour[k].tall > 0
        if (raised && from < 0) from = k
        if (raised || from < 0) continue
        const form = paintRise(
          contour.slice(from, k),
          from,
          (seed ^ Math.imul(index * 512 + from + 1, 0x9e3779b9)) >>> 0,
          index,
        )
        if (form) groups.push(`${form}${boulders(crest.slice(from, k), random)}</g>`)
        from = -1
      }
      return groups.join('')
    })
    .join('')
  return `<g class="trail-relief" aria-hidden="true">${forms}</g>`
}
