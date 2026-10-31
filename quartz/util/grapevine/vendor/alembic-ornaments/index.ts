import {
  composeFrame,
  frameInset,
  FRAMES,
  FRAME_SCALE_MAX,
  FRAME_SCALE_MIN,
  type FrameClear,
  type FramePreset,
} from './internal/frames.js'
import { apply, clamp, type Vec } from './internal/geometry.js'
import { exp, log } from './internal/math.js'
import {
  composePreset,
  PRESETS,
  type OrnamentPreset,
  type PresetInfo,
  type Rect,
} from './internal/presets.js'
import { render } from './internal/render.js'
import { createRng } from './internal/rng.js'

/**
 * Seeded woodcut ornaments, drawn as SVG. Everything here is pure and has
 * no dependencies. This file is the public surface: anything not exported
 * from it (or from ./react) is internal and free to change.
 */

export { PRESETS, type OrnamentPreset, type PresetInfo, type Vec }

/**
 * Everything needed to regenerate an ornament: which composition, and which
 * of its variations. Small on purpose — this is what a document stores.
 */
export interface OrnamentParams {
  preset: OrnamentPreset
  seed: number
}

export interface Ornament {
  params: OrnamentParams
  svg: string
  width: number
  height: number
}

export const DEFAULT_ORNAMENT: OrnamentParams = { preset: 'vine', seed: 1 }

const isPreset = (v: unknown): v is OrnamentPreset => PRESETS.some(p => p.id === v)

export function normalizeParams(p: Partial<OrnamentParams>): OrnamentParams {
  return {
    preset: isPreset(p.preset) ? p.preset : DEFAULT_ORNAMENT.preset,
    seed: Math.floor(Math.abs(Number(p.seed))) || 0,
  }
}

export function presetInfo(id: OrnamentPreset): PresetInfo {
  return PRESETS.find(p => p.id === id) ?? PRESETS[0]
}

const cache = new Map<string, Ornament | Frame>()

function cached<T extends Ornament | Frame>(key: string, draw: () => T): T {
  const hit = cache.get(key)
  if (hit) return hit as T
  const out = draw()
  if (cache.size > 200) cache.clear()
  cache.set(key, out)
  return out
}

/**
 * Draws an ornament. Pure and deterministic: the same params always give the
 * same SVG, in `currentColor` so it takes the ink of whatever it sits in.
 */
export function generateOrnament(input: Partial<OrnamentParams>): Ornament {
  const params = normalizeParams(input)
  const key = `${params.preset}:${params.seed}`
  return cached(key, () => ({
    params,
    ...render(composePreset(params.preset, createRng(key)), key),
  }))
}

/**
 * The ornament's ink as polygons in its own width × height box, holes left
 * out: what text flowing round it has to clear. Same params, same polygons
 * as the drawing.
 */
export function ornamentOutline(input: Partial<OrnamentParams>): {
  polys: Vec[][]
  width: number
  height: number
} {
  const params = normalizeParams(input)
  const c = composePreset(params.preset, createRng(`${params.preset}:${params.seed}`))
  if (c.block) {
    const { x, y, w, h } = c.block.outer
    return {
      polys: [
        [
          { x, y },
          { x: x + w, y },
          { x: x + w, y: y + h },
          { x, y: y + h },
        ],
      ],
      width: c.width,
      height: c.height,
    }
  }
  const clampTo =
    (r: Rect | undefined) =>
    (p: Vec): Vec =>
      r ? { x: clamp(p.x, r.x, r.x + r.w), y: clamp(p.y, r.y, r.y + r.h) } : p
  const polys: Vec[][] = []
  for (const pl of c.placements) {
    const group = pl.group === undefined ? undefined : c.groups?.[pl.group]
    const clip = clampTo(group ? group.clip : c.clip)
    for (const s of c.parts[pl.part].shapes) {
      if (s.hole) continue
      polys.push(
        s.poly.map(p => {
          const q = clip(apply(pl.m, p))
          return group ? apply(group.m, q) : q
        }),
      )
    }
  }
  return { polys, width: c.width, height: c.height }
}

/* ------------------------------------------------------------------ frames */

export {
  FRAMES,
  frameInset,
  FRAME_SCALE_MIN,
  FRAME_SCALE_MAX,
  type FrameClear,
  type FrameInfo,
  type FramePreset,
} from './internal/frames.js'

export interface FrameParams {
  preset: FramePreset
  seed: number
  /** How heavy it's set: 1 is its natural size (frames stored before this have none) */
  scale?: number
}

export interface Frame {
  params: Required<FrameParams>
  svg: string
  width: number
  height: number
}

const isFrame = (v: unknown): v is FramePreset => FRAMES.some(f => f.id === v)

/** Clamped to the range, in steps of 0.05 so nearby values share a drawing */
export function normalizeFrameScale(scale: unknown): number {
  const s = Number(scale)
  if (!Number.isFinite(s) || s <= 0) return 1
  return Math.round(Math.min(FRAME_SCALE_MAX, Math.max(FRAME_SCALE_MIN, s)) * 20) / 20
}

export function normalizeFrame(
  p: Partial<FrameParams> | null | undefined,
): Required<FrameParams> | null {
  if (!p || !isFrame(p.preset)) return null
  return {
    preset: p.preset,
    seed: Math.floor(Math.abs(Number(p.seed))) || 0,
    scale: normalizeFrameScale(p.scale),
  }
}

/**
 * Draws a frame for a box of the given width ÷ height. Same seed, same
 * flowers and leaves as the presets; only the layout follows the box. The
 * scale changes the weight, not the variation.
 */
export function generateFrame(input: FrameParams, aspect: number): Frame {
  const params = normalizeFrame(input) ?? { preset: 'border' as const, seed: 0, scale: 1 }
  return drawFrame(params, aspect)
}

function drawFrame(params: Required<FrameParams>, aspect: number): Frame {
  const a = Math.round(aspect * 1000) / 1000
  const key = `frame:${params.preset}:${params.seed}:${a}:${params.scale}`
  return cached(key, () => {
    const rng = createRng(`frame:${params.preset}:${params.seed}`)
    return { params, ...render(composeFrame(params.preset, rng, a, params.scale), key) }
  })
}

/* ------------------------------------------------------------ fitted frames */

export interface FitFrameParams {
  preset: FramePreset
  seed: number
  /** Room the frame takes on each side, in the same units as the box (px, usually) */
  size: number
  /** What `size` keeps content clear of: all the ink (default), or only the edges (see frameInset) */
  clear?: FrameClear
}

export interface FittedFrame extends Frame {
  /**
   * Room the frame actually takes on each side, in the box's units: a hair
   * under the size asked for, or 30% of the box's shorter side if that's
   * less, so a small box keeps some room inside. (A frame too light to
   * draw, like 4px round a 20000px box, comes out at the lightest it can.)
   */
  inset: number
}

/** Lightest a fitted frame is drawn: past it the motifs get finer than a pixel on any screen */
const FIT_SCALE_MIN = 0.04
/** Most of the box's shorter side a fitted frame takes on each side, so some box is left inside */
const FIT_MAX_SHARE = 0.3

/**
 * Draws a frame for a box of width × height that takes `size` on each
 * side, however big the box is: a 300px card and a 1200px hero get the
 * same weight of border, with more vine along the bigger one.
 *
 * The weight snaps down to steps of 2%, so a box being resized reuses
 * drawings instead of growing a new one every pixel; the frame lands a
 * hair inside `size`, never outside it.
 */
export function fitFrame(input: FitFrameParams, width: number, height: number): FittedFrame | null {
  if (!(width > 0 && height > 0)) return null
  const base = normalizeFrame({ preset: input.preset, seed: input.seed }) ?? {
    preset: 'border' as const,
    seed: 0,
    scale: 1,
  }
  const size = Number(input.size)
  if (!(size > 0)) return null
  const aspect = width / height
  // The inset grows in proportion to the weight, so one measurement at weight 1 gives the rest.
  const unit = frameInset(base.preset, aspect, 1, input.clear).x * width
  const room = Math.min(size, FIT_MAX_SHARE * Math.min(width, height))
  const stepped = Math.round(exp(Math.floor(log(room / unit) * 50) / 50) * 1e6) / 1e6
  const scale = Math.max(FIT_SCALE_MIN, stepped)
  return { ...drawFrame({ ...base, scale }, aspect), inset: unit * scale }
}
