import type { Grade } from '../../../util/scene-grade'
import type { SceneReveal } from '../scene-reveal'
import type { Matrix } from './affine-geometry'

export type Layer = {
  texture: WebGLTexture | null
  width: number
  height: number
  // Maps the owner's user space onto texels at raster time.
  base: Matrix
  scale: number
  palette: number
  revision: number
  dirty: boolean
  // When it first needed a repaint; the oldest repaint goes first, so a busy texture cannot starve the rest.
  waiting: number
}
export type Run = Layer & {
  kind: 'run'
  owner: Group
  nodes: Element[]
  box: Box | null
  hits: number[]
  hot: Set<Element>
  placed: Float32Array | null
}
export type Surface = {
  kind: 'surface'
  frame: SVGForeignObjectElement
  canvas: HTMLCanvasElement
  style: CSSStyleDeclaration
  offset: Matrix
  shown: boolean
  placed: Float32Array | null
  texture: WebGLTexture | null
  dirty: boolean
  // Repainted rects in canvas texels since the last upload; null when the whole surface changed.
  damage: Box[] | null
}
export type Group = {
  kind: 'group'
  element: SVGGraphicsElement
  items: (Run | Surface | Group)[]
  structure: boolean
  style: CSSStyleDeclaration
  mask: (Layer & { source: SVGMaskElement; placed: Float32Array | null }) | null
  clip: SVGClipPathElement | null
  // Cached between the mutations, animations and resizes that can change them.
  moved: boolean
  fresh: boolean
  matrix: Matrix
  screen: Matrix
  opacity: number
  shown: boolean
}
export type Clip = { owner: Group; source: SVGClipPathElement }
export type Box = { left: number; top: number; right: number; bottom: number }

// Strokes and glyphs reach this far past the geometry box, in user units.
export const padding = 3
export const ambientLayers =
  '.trail-weather, .trail-hiker-tracks, .trail-reader-tracks, .trail-camp-tracks, .trail-visitor-tracks, .trail-ranger-tracks'
export const motionProperties = new Set(['opacity', 'transform', 'translate', 'rotate', 'scale'])
export const motionVariables = new Set([
  '--trail-tree-wind-scale',
  '--trail-weather-delay',
  '--trail-wind-period',
  '--trail-rain-strength',
  '--trail-wind-strength',
  '--trail-gust-strength',
  '--trail-wind-x',
  '--trail-wind-y',
  '--trail-rain-x',
  '--trail-leaf-push-x',
  '--trail-leaf-push-y',
  '--trail-leaf-rest-x',
  '--trail-leaf-rest-y',
  '--trail-leaf-flip',
  '--trail-leaf-lag-x',
  '--trail-leaf-lag-y',
  '--trail-leaf-travel-x',
  '--trail-leaf-travel-y',
])
export const unpainted = new Set([
  'defs',
  'mask',
  'clipPath',
  'linearGradient',
  'radialGradient',
  'pattern',
  'marker',
  'symbol',
  'title',
  'desc',
  'metadata',
  'style',
  'script',
])
export const shapes = new Set(['path', 'ellipse', 'circle', 'rect', 'line', 'polyline', 'polygon'])
export const reference = (value: string | null) => value?.match(/url\(["']?#([^"')]+)["']?\)/)?.[1]

export type Classes = {
  CSSTransition: CSSTransition
  Element: Element
  KeyframeEffect: KeyframeEffect
  SVGAElement: SVGAElement
  SVGClipPathElement: SVGClipPathElement
  SVGElement: SVGElement
  SVGForeignObjectElement: SVGForeignObjectElement
  SVGGElement: SVGGElement
  SVGGeometryElement: SVGGeometryElement
  SVGGradientElement: SVGGradientElement
  SVGGraphicsElement: SVGGraphicsElement
  SVGMaskElement: SVGMaskElement
  SVGRadialGradientElement: SVGRadialGradientElement
  SVGSVGElement: SVGSVGElement
  SVGTextElement: SVGTextElement
}

type Fade = {
  texture: WebGLTexture
  started: number
  updated: number
  progress: number
  origin: number
  scrollX: number
  scrollY: number
}
type ResizeFrame = { texture: WebGLTexture; width: number; height: number; started: number | null }
type Coverage = {
  texture: WebGLTexture | null
  cells: Uint8Array
  width: number
  height: number
  active: boolean
}
// The locals that more than one part of the compositor reads or writes.
type Shared = {
  resolution: number
  // The current frame's time; one frame counts once however many records it brings.
  clock: number
  palette: number
  revision: number
  // Hold the complete canvas while a whole-scene change fills within the normal raster budget.
  flush: boolean
  sizeDirty: boolean
  // Bumps when the canvas size changes, so every clip texture is rebuilt.
  clipVersion: number
  ready: boolean
  firstPaint: WebGLSync | null
  surfaceExposed: boolean
  generationPending: boolean
  palettePending: boolean
  paletteTransition: boolean
  // Each cell switches between complete palettes after uploads finish. The scene owns its
  // timing, so a completed body transition cannot cut a late palette reveal short.
  fade: Fade | null
  resizeFrame: ResizeFrame | null
  gradeFrom: Grade
  gradeTo: Grade
  gradeStarted: number
  initialAnimations: Animation[] | null
}
export type Scope = {
  scene: HTMLElement
  host: HTMLElement
  signal: AbortSignal
  opening: SceneReveal
  document: Document
  view: Window & typeof globalThis
  svg: SVGSVGElement
  canvas: HTMLCanvasElement
  gl: WebGL2RenderingContext
  is: <K extends keyof Classes>(value: unknown, name: K) => value is Classes[K]
  tree: () => Document | ShadowRoot
  active: () => boolean
  reducedMotion: MediaQueryList
  size: { width: number; height: number }
  coverage: Coverage
  // Surfaces uploaded since the last paint, with the canvas-texel rects they repainted.
  repainted: Map<Surface, Box[] | null>
  shared: Shared
}

export type Draw = {
  item: Run | Surface
  texture: WebGLTexture
  placed: Float32Array
  alpha: number
  mask: Float32Array
  maskTexture: WebGLTexture | null
  clipTexture: WebGLTexture | null
}
// What a draw last showed, and when that changed.
export type Trace = {
  texture: WebGLTexture
  content: number
  placed: Float32Array
  alpha: number
  mask: Float32Array
  maskTexture: WebGLTexture | null
  maskContent: number
  clipTexture: WebGLTexture | null
  clipContent: number
  changed: number
  // Whether the last change repainted the texture in place, with nothing else altered.
  repaint: boolean
  // How long the draw must stay unchanged before it joins a cached segment.
  hold: number
}
// Canvas pixels: left, top, right, bottom.
export type Rect = [number, number, number, number]
// A run of unchanged draws composited once into a texture covering their canvas extent.
export type Segment = {
  items: (Run | Surface)[]
  rect: Rect
  texture: WebGLTexture | null
  framebuffer: WebGLFramebuffer | null
  placed: Float32Array
  built: number
}
export type Unit = { draw: Draw; rect: Rect } | { segment: Segment; rect: Rect }
