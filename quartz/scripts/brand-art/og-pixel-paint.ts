// Painting the pixel cards: the scene is sampled cell by cell at the card's low resolution and
// written as horizontal runs, one path per colour, as the trail's mountain backdrop paints its
// rock. The rock, snow and gullies follow mountain-backdrop.ts's alpine sampler on the 4 × 4 Bayer
// screen every pixel scene shares.

import type { TreeKind } from '../../components/visitor-trail/scenery/species'
import type { Pt } from '../../util/mascot/pen'
import type { Palette, Range, Rock } from './og-pixel-tones'
import { pixels } from '../../components/visitor-trail/pixels'
import { trailRandom } from '../../components/visitor-trail/random'
import { sceneSettings } from '../../components/visitor-trail/scene-settings'
import { tree } from '../../components/visitor-trail/scenery/tree-growth'
import { bayerThreshold } from '../../util/bayer'
import { ridgeAt, shadeFaces } from './scene'

export type Sample = (x: number, y: number) => string | undefined

// Every cell of a w × h canvas, as runs of one colour: `M x y h n v1 h -n Z` per run.
export function runs(w: number, h: number, sample: Sample) {
  const marks = new Map<string, string[]>()
  for (let y = 0; y < h; y++) {
    let tone: string | undefined
    let start = 0
    for (let x = 0; x <= w; x++) {
      const next = x === w ? undefined : sample(x, y)
      if (next === tone) continue
      if (tone) {
        if (!marks.has(tone)) marks.set(tone, [])
        marks.get(tone)!.push(`M${start} ${y}h${x - start}v1h${start - x}Z`)
      }
      tone = next
      start = x
    }
  }
  return [...marks].map(([fill, d]) => `<path fill="${fill}" d="${d.join('')}"/>`).join('')
}

// The front-most layer that paints a cell wins.
export const stack =
  (...layers: Sample[]): Sample =>
  (x, y) => {
    for (const layer of layers) {
      const tone = layer(x, y)
      if (tone) return tone
    }
    return undefined
  }

const slopeAt = (line: readonly Pt[], x: number) =>
  (ridgeAt(line, x + 2) - ridgeAt(line, x - 2)) / 4

// Whether the cell's centre lies inside a face polygon.
const inside = (poly: readonly Pt[], x: number, y: number) => {
  let hit = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [ax, ay] = poly[i]
    const [bx, by] = poly[j]
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) hit = !hit
  }
  return hit
}

// The faces the light cannot reach, the pen cards' own wedges (scene.ts shadeFaces) at a third.
const shade = (line: readonly Pt[], lightX: number) => {
  const faces = shadeFaces(line, line, lightX, 10).map(f => f.pts)
  return (x: number, y: number) => faces.some(f => inside(f, x + 0.5, y + 0.5))
}

// A far range: its body in one tone, the faces turned from the light a shade darker, their edges
// broken on the screen, a rim of light on the lit crests.
export function hills(line: readonly Pt[], tone: Range, lightX: number, seed: number): Sample {
  const shaded = shade(line, lightX)
  return (x, y) => {
    const crest = ridgeAt(line, x)
    if (y < crest) return undefined
    const depth = y - crest
    const threshold = bayerThreshold(x + seed, y + seed * 3)
    if (depth < 1.5 && slopeAt(line, x) > 0.1) return tone.light
    if (shaded(x, y)) return threshold < 0.85 ? tone.shade : tone.fill
    return threshold < 0.1 && depth > 4 ? tone.shade : tone.fill
  }
}

// The peak: lit faces in rock with light breaking through on the screen, the shaded flank in shade
// cut by gullies that run straight downhill, snow on the summit with a dithered edge, strata shelves
// and a sparse scatter of lichen below the snow.
export function alpine(
  line: readonly Pt[],
  tone: Record<Rock, string>,
  summit: Pt,
  snowReach: number,
  lightX: number,
  seed: number,
): Sample {
  const phase = trailRandom(seed)() * 6
  const dark = shade(line, lightX)
  return (x, y) => {
    const crest = ridgeAt(line, x)
    if (y < crest) return undefined
    const slope = slopeAt(line, x)
    const depth = y - crest
    const shaded = dark(x, y)
    const threshold = bayerThreshold(x + seed, y + seed)
    const grain = Math.sin(y / 8 + x / 30 + phase) * Math.sin(x / 13 - y / 21)
    if (depth < 1.5 && !shaded && slope > 0.1) return tone.light
    // Snow: `snowReach` deep under the summit, thinning out thirty cells either side, its edge
    // breaking up over three cells of screen.
    const reach =
      snowReach * Math.max(0, 1 - Math.abs(x - summit[0]) / 32) + Math.sin(x / 5 + phase)
    const under = (reach - depth) / 3
    if (under > 0 && threshold < under + 0.1) return shaded ? tone.cap : tone.snow
    if (shaded) {
      const lean = Math.max(-0.7, Math.min(0.7, -slope * 0.5))
      const u = x + depth * lean
      const rib = Math.sin(((u / (5 + depth * 0.06)) * Math.PI) / 2 + Math.sin(u / 17 + phase))
      if (rib < -0.82) return tone.fracture
      if (rib > 0.6 && threshold < 0.3) return tone.rock
      return threshold < 0.125 && grain > 0.1 ? tone.rock : tone.shade
    }
    // Shelves: bands of broken rock across the lit side, slanting with the ground, each a run of
    // some forty cells with gaps between, never a ruled stripe.
    const shelf = Math.sin((y + x * 0.12) / 13 + phase)
    if (shelf > 0.93 && depth > 6 && Math.sin(x / 19 + y / 7 + phase) > 0.1 && threshold < 0.7)
      return tone.fracture
    if (threshold < 0.125 && grain > 0.35 && depth > 10) return tone.lichen
    // Light and shade break through the rock where the grain runs high or low, in patches rather
    // than a lattice: the screen's cover follows the grain.
    if (threshold < 0.22 * Math.max(0, grain + 0.6) - depth * 0.002) return tone.light
    if (threshold > 1 - 0.3 * Math.max(0, -grain - 0.1) - depth * 0.004) return tone.shade
    return tone.rock
  }
}

// A meadow under its ridge: the trail's field tones stepping darker toward the viewer through the
// screen, with tufts of grass scattered over it.
export function meadow(
  line: readonly Pt[],
  tones: string[],
  tuft: string,
  seed: number,
  h: number,
): Sample {
  const random = trailRandom(seed)
  const tufts = new Set<number>()
  for (let i = 0; i < 70; i++) {
    const x = Math.floor(random() * 400)
    const y = Math.floor(ridgeAt(line, x) + 3 + random() * (h - ridgeAt(line, x)) * 0.8)
    tufts.add(x * 1000 + y)
    tufts.add((x - 1) * 1000 + y - 1)
    tufts.add((x + 1) * 1000 + y - 1)
  }
  return (x, y) => {
    const crest = ridgeAt(line, x)
    if (y < crest) return undefined
    if (tufts.has(x * 1000 + y)) return tuft
    const t = ((y - crest) / Math.max(1, h - crest)) * (tones.length - 0.4)
    const i = Math.floor(t)
    const up = t - i > bayerThreshold(x, y)
    return tones[Math.min(tones.length - 1, i + Number(up))]
  }
}

// The sky in bands from the top of the card to the skyline, each dithered into the next.
export function dawn(bands: string[], top: number, horizon: number): Sample {
  return (x, y) => {
    const t = Math.max(0, Math.min(0.999, (y - top) / (horizon - top))) * (bands.length - 1)
    const i = Math.floor(t)
    return bands[Math.min(bands.length - 1, i + Number(t - i > bayerThreshold(x, y)))]
  }
}

// A disc: the sun or the moon, its edge stepped on the screen, seas in a second tone on the moon.
export function orb(
  cx: number,
  cy: number,
  r: number,
  tones: string[],
  seas: readonly Pt[] = [],
): Sample {
  return (x, y) => {
    const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy)
    if (d > r + 0.5) return undefined
    if (d > r - 1.2 && bayerThreshold(x, y) > 0.5) return undefined
    for (const [sx, sy] of seas) if (Math.hypot(x - sx, y - sy) < 2.2) return tones[1]
    return tones[0]
  }
}

// Single cells of starlight above the skyline, a few of them crossed, and one falling.
export function stars(seed: number, skyline: (x: number) => number, tones: string[]) {
  const random = trailRandom(seed)
  const cells: string[] = []
  for (let i = 0; i < 70; i++) {
    const x = Math.floor(random() * 400)
    const y = Math.floor(random() * 120)
    if (y > skyline(x) - 6) continue
    const bright = random() < 0.3
    cells.push(`<path fill="${tones[bright ? 0 : 1]}" d="M${x} ${y}h1v1h-1Z"/>`)
    if (bright && random() < 0.35)
      cells.push(
        `<path fill="${tones[1]}" d="M${x - 1} ${y}h1v1h-1ZM${x + 1} ${y}h1v1h-1ZM${x} ${y - 1}h1v1h-1ZM${x} ${y + 1}h1v1h-1Z"/>`,
      )
  }
  for (let i = 0; i < 7; i++)
    cells.push(`<path fill="${tones[i < 3 ? 0 : 1]}" d="M${46 + i * 3} ${18 + i}h2v1h-2Z"/>`)
  return cells.join('')
}

// Cells every `step` units along a line: footprints at three, a stick at one.
export function dots(pts: readonly Pt[], tone: string, step = 3) {
  const d: string[] = []
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]]
    const n = Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / step)
    for (let k = 0; k < n; k++) {
      const t = k / n
      d.push(
        `M${Math.round(a[0] + (b[0] - a[0]) * t)} ${Math.round(a[1] + (b[1] - a[1]) * t)}h1v1h-1Z`,
      )
    }
  }
  return `<path fill="${tone}" d="${d.join('')}"/>`
}

// Trees from the trail's growth grammar, rooted on a ridge, `height` in scene units; the painter
// picks 1-unit cells for small trees and 2-unit cells for tall ones.
export function grove(
  scope: 'near' | 'far',
  line: readonly Pt[],
  plan: readonly (readonly [number, TreeKind, number, number])[],
) {
  const settings = { ...sceneSettings.forest, dapple: 70 }
  return `<g class="${scope}">${plan
    .map(([x, kind, height, seed]) => {
      const { markup } = tree(kind, seed, height / 2.4 + 8, settings, 1)
      return `<g transform="translate(${x} ${Math.round(ridgeAt(line, x))})">${markup}</g>`
    })
    .join('')}</g>`
}

// Rows of 2-unit cells in hex tones; pixels() names tones as classes, so they are turned to fills.
export const sprite = (
  rows: readonly string[],
  tones: Record<string, string>,
  x: number,
  y: number,
) =>
  `<g transform="translate(${x} ${y})">${pixels(rows, tones, 2).replace(/class="(#[0-9a-f]{6})"/g, 'fill="$1"')}</g>`

export const style = (p: Palette) => `<style>${p.css}</style>`
