import type { Point } from './scene-model'
import { type Box, viewBox } from './svg'

export type LayoutMode = 'wide' | 'narrow'

const WIDE_MIN = 600
const OWN_HEIGHT = 520
const MIN_SCALE = 0.55
/** Room kept between the stage's left edge and a pulled book. */
const LIFT_PAD = 8
/** Track height while the scroller is found, so an element that only grows with its content fails the test. */
const PROBE_HEIGHT = 100_000
const SVG_NS = 'http://www.w3.org/2000/svg'

/** Screen units to world px: the origin is the scene's top-left bound, then `scale`. */
export interface Frame {
  minX: number
  minY: number
  scale: number
}

export const toPx = (f: Frame, [x, y]: Point): Point => [
  (x - f.minX) * f.scale,
  (y - f.minY) * f.scale,
]

/**
 * Positions an `<svg>` over a screen box: viewBox in screen units, size in world px. The box is
 * grown to whole pixels and the viewBox keeps full precision, so every layer maps screen units to
 * the same pixels: the browser snaps a fractional box, and layers would meet a fraction apart.
 */
export function placeSvg(el: SVGSVGElement, box: Box, f: Frame): void {
  const left = Math.floor((box.x - f.minX) * f.scale)
  const top = Math.floor((box.y - f.minY) * f.scale)
  const width = Math.max(0, Math.ceil((box.x + box.w - f.minX) * f.scale) - left)
  const height = Math.max(0, Math.ceil((box.y + box.h - f.minY) * f.scale) - top)
  const unit = (px: number) => px / f.scale
  el.setAttribute(
    'viewBox',
    `${f.minX + unit(left)} ${f.minY + unit(top)} ${unit(width)} ${unit(height)}`,
  )
  el.setAttribute('width', String(width))
  el.setAttribute('height', String(height))
  el.style.left = `${left}px`
  el.style.top = `${top}px`
}

export function createSvgLayer(parent: HTMLElement, cls: string): SVGSVGElement {
  const el = parent.createSvg('svg', { cls, attr: { 'aria-hidden': 'true', focusable: 'false' } })
  el.setAttribute('xmlns', SVG_NS)
  return el
}

export interface Metrics {
  mode: LayoutMode
  /** Height of the sticky area (stage plus page in narrow mode). */
  stageHeight: number
  stageWidth: number
  /** Height of the stage drawing itself. */
  sceneHeight: number
  viewportHeight: number
  /** Scroll offset at which the track starts. */
  trackTop: number
  /** Scroll offset when measured. */
  scrollTop: number
  containerWidth: number
  /** World px per screen unit (SPEC 4.3). */
  scale: number
  /** Left edge and width of the drawn case in world px. */
  caseLeft: number
  caseWidth: number
  /** Width of the case with every pulled position in world px; the world starts at its left. */
  boxWidth: number
}

/** The scroll container, the scroll track and the camera. */
export class Stage {
  scroller: HTMLElement | null = null
  own = false
  /** Scroll offset the camera last used. */
  private seen = 0
  metrics: Metrics = {
    mode: 'wide',
    stageHeight: OWN_HEIGHT,
    stageWidth: 0,
    sceneHeight: OWN_HEIGHT,
    viewportHeight: OWN_HEIGHT,
    trackTop: 0,
    scrollTop: 0,
    containerWidth: -1,
    scale: 1,
    caseLeft: 0,
    caseWidth: 0,
    boxWidth: 0,
  }

  constructor(
    private container: HTMLElement,
    private root: HTMLElement,
    private track: HTMLElement,
    private stage: HTMLElement,
  ) {}

  /**
   * Finds the element that scrolls the view (SPEC 6.1): containerEl or its nearest ancestor that
   * has `overflow-y: auto | scroll` and a height of its own. An embed scrolls in the view itself.
   * Null while the view has no box; the resize observer calls again when it gets one.
   */
  discover(): HTMLElement | null {
    if (this.scroller) return this.scroller
    if (!this.container.isConnected || this.container.getClientRects().length === 0) return null
    const embedded = this.container.closest('.internal-embed, .markdown-embed, .bases-embed')
    let scroller: HTMLElement | null = null
    // In full screen the view is the whole screen and scrolls itself.
    if (!embedded && !this.fullscreen) {
      // Back from full screen the view still has its own height, which would hide the pane's scroll.
      this.root.removeClass('is-own-scroller')
      this.track.style.height = `${PROBE_HEIGHT}px`
      for (
        let el: HTMLElement | null = this.container;
        el && el !== document.body;
        el = el.parentElement
      ) {
        const overflow = getComputedStyle(el).overflowY
        if ((overflow === 'auto' || overflow === 'scroll') && el.scrollHeight > el.clientHeight) {
          scroller = el
          break
        }
      }
    }
    this.own = !scroller
    this.root.toggleClass('is-own-scroller', this.own)
    this.scroller = scroller ?? this.root
    return this.scroller
  }

  /** Forgets the scroller, for a view that moved in or out of full screen. */
  reset(): void {
    this.scroller = null
  }

  get fullscreen(): boolean {
    return document.fullscreenElement === this.root
  }

  /** The width the layout fits: the pane's, or the screen's in full screen. */
  private get width(): number {
    return (this.fullscreen ? this.root : this.container).clientWidth
  }

  /**
   * Lays out the stage and the track for `count` books, and records the measurements the camera
   * needs. The stage fills the scroller's height; the grid in styles.css gives it its width (80% of
   * a wide view), and the scale fits `fit` screen units to that width (SPEC 4.3, with the garden's
   * fit width). In a wide view the scale also fits `reach`, the case with every pulled book, beside
   * the `inset` px the control column always takes, as the garden fits its whole box. `span` is
   * the drawn case in screen units from the frame's left edge `minX`.
   */
  measure(
    count: number,
    step: number,
    fit: number,
    span: readonly [number, number],
    minX: number,
    reach: number,
    inset: number,
    paged: boolean,
  ): Metrics {
    const scroller = this.scroller
    if (!scroller) return this.metrics
    const width = this.width
    const mode: LayoutMode = width >= WIDE_MIN ? 'wide' : 'narrow'
    if (this.root.dataset.layout !== mode) this.root.dataset.layout = mode
    const viewportHeight = scroller.clientHeight || OWN_HEIGHT
    const stageHeight = viewportHeight
    // Narrow, the page panel sits under the stage; with the editor pane there is no page panel.
    const sceneHeight = mode === 'wide' || !paged ? stageHeight : Math.round(stageHeight * 0.58)
    // A layout read, once per measure: the grid follows the pane, and real themes may pad the view.
    const stageWidth = this.stage.clientWidth || width
    let fitted = stageWidth / (fit || 1)
    if (mode === 'wide')
      fitted = Math.min(fitted, (stageWidth - inset - 2 * LIFT_PAD) / (reach || 1))
    const scale = Math.min(1, Math.max(MIN_SCALE, fitted))
    const caseWidth = (span[1] - span[0]) * scale
    this.root.style.setProperty('--bc-stage-height', `${stageHeight}px`)
    this.root.style.setProperty('--bc-scene-height', `${sceneHeight}px`)
    // As SPEC 6.1, but never short of the room that brings the last book to the reading line.
    const height = Math.max(
      count * step + stageHeight,
      Math.max(0, count - 1) * step + viewportHeight,
    )
    this.track.style.height = `${height}px`
    const scrollTop = scroller.scrollTop
    const trackTop =
      this.track.getBoundingClientRect().top -
      scroller.getBoundingClientRect().top +
      scrollTop -
      (this.own ? 0 : scroller.clientTop)
    this.metrics = {
      mode,
      stageHeight,
      stageWidth,
      sceneHeight,
      viewportHeight,
      trackTop,
      scrollTop,
      containerWidth: width,
      scale,
      caseLeft: (span[0] - minX) * scale,
      caseWidth,
      boxWidth: reach * scale,
    }
    return this.metrics
  }

  /** True when the view's width or the scroller's height differs from the last measure. */
  resized(): boolean {
    const { containerWidth, viewportHeight } = this.metrics
    const height = this.scroller?.clientHeight || OWN_HEIGHT
    return this.width !== containerWidth || height !== viewportHeight
  }

  /**
   * Reading position in books: 0 when the track's top meets the top of the scroller. A caller
   * that wrote the layout since the last frame passes the offset it knows, as reading
   * `scrollTop` then forces a layout.
   */
  progress(count: number, step: number, scrollTop?: number): number {
    if (!this.scroller) return 0
    this.seen = scrollTop ?? this.scroller.scrollTop
    if (count === 0) return 0
    const t = (this.seen - this.metrics.trackTop) / step
    return Math.min(count - 1, Math.max(0, t))
  }

  /** True when the scroller moved since the camera last read it, and its scroll event is due. */
  moved(): boolean {
    return this.scroller !== null && this.scroller.scrollTop !== this.seen
  }

  scrollTo(index: number, step: number): void {
    if (!this.scroller) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    this.scroller.scrollTo({
      top: this.metrics.trackTop + index * step,
      behavior: reduced ? 'auto' : 'smooth',
    })
  }

  /** Stage point the camera holds the focused spine at. */
  anchor(): Point {
    const { mode, stageWidth, sceneHeight } = this.metrics
    return mode === 'wide'
      ? [stageWidth * 0.38, sceneHeight * 0.52]
      : [stageWidth * 0.5, sceneHeight * 0.45]
  }

  /**
   * World translation for a camera on `target`. Across, the case never leaves part of the stage
   * empty: a case narrower than the stage is centred, a wider one stops at its walls. The pulled
   * book (`lift`, its left and right edges in world px) gives way: near the right wall it stays
   * clear of the `reserve` px the control column takes, near the left wall it stays on the stage,
   * which wins when both cannot hold. In a wide view whose stage holds the whole box beside the
   * column, the case stays still across: centred, or moved left of the column, with every pulled
   * book on the stage.
   */
  offset(target: Point, lift: Point, reserve: number): Point {
    const { mode, stageWidth, caseLeft, caseWidth, boxWidth } = this.metrics
    const [ax, ay] = this.anchor()
    const room = stageWidth - reserve - LIFT_PAD - boxWidth
    if (mode === 'wide' && room >= LIFT_PAD) {
      const centred = (stageWidth - caseWidth) / 2 - caseLeft
      return [Math.min(room, Math.max(LIFT_PAD, centred)), ay - target[1]]
    }
    let x =
      caseWidth <= stageWidth
        ? (stageWidth - caseWidth) / 2 - caseLeft
        : Math.min(-caseLeft, Math.max(stageWidth - caseLeft - caseWidth, ax - target[0]))
    x = Math.min(x, stageWidth - reserve - lift[1])
    x = Math.max(x, LIFT_PAD - lift[0])
    return [x, ay - target[1]]
  }

  /** Visible band of the world, in world px, for the world translation `offset` and `zoom`. */
  band(offset: Point, zoom = 1): [number, number] {
    return [-offset[1] / zoom, (-offset[1] + this.metrics.sceneHeight) / zoom]
  }

  /** Visible columns of the world, in world px, for the world translation `offset` and `zoom`. */
  span(offset: Point, zoom = 1): [number, number] {
    return [-offset[0] / zoom, (-offset[0] + this.metrics.stageWidth) / zoom]
  }
}

/** Interpolated camera target between the spines either side of `t`, for `count > 0` books. */
export function cameraTarget(centre: (index: number) => Point, count: number, t: number): Point {
  const a = centre(Math.floor(t))
  const b = Math.ceil(t) < count ? centre(Math.ceil(t)) : a
  const k = t - Math.floor(t)
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k]
}

export type LayerKind = 'solids' | 'shelf'

interface Slot {
  kind: LayerKind
  el: SVGSVGElement
  /** Markup for the current signature; null until it is rendered. */
  markup: string | null
  signature: string
  /** viewBox, size and position as last written. */
  placed: string
  /** Top and bottom in world px. */
  band: [number, number]
  /** Left and right in world px. */
  span: [number, number]
  mounted: boolean
}

export interface LayerSource {
  /** Markup of a layer: geometry, classes and colour tokens only. */
  render(layer: number): string
  /** Last changes to the markup before it is written, such as the lifted book's hidden copy. */
  prepare(layer: number, markup: string): string
  /** Fills note text into freshly written markup. */
  mounted(layer: number, el: SVGSVGElement): void
}

/**
 * One `<svg>` per painter's-order layer. Case layers stay mounted. Shelf markup stays in memory
 * and is written into the DOM only near the camera, with a wider band for unmounting so the edge
 * does not churn (SPEC 5.5). Windowing changes only the children of a shelf, never its attributes.
 */
export class LayerStack {
  private slots: Slot[] = []
  private shape = ''

  constructor(
    private world: HTMLElement,
    private before: Element,
    private source: LayerSource,
  ) {}

  get length(): number {
    return this.slots.length
  }

  element(layer: number): SVGSVGElement | null {
    return this.slots[layer]?.el ?? null
  }

  isMounted(layer: number): boolean {
    return this.slots[layer]?.mounted ?? false
  }

  /** Recreates the elements when the sequence of layer kinds changed. */
  reshape(kinds: LayerKind[]): void {
    const shape = kinds.join(',')
    if (shape === this.shape) return
    this.clear()
    this.shape = shape
    let front = false
    for (const kind of kinds) {
      if (kind === 'shelf') front = true
      const cls =
        kind === 'shelf' ? 'bookcase-shelf' : front ? 'bookcase-case-front' : 'bookcase-case'
      const el = createSvgLayer(this.world, cls)
      this.world.insertBefore(el, this.before)
      this.slots.push({
        kind,
        el,
        markup: null,
        signature: '',
        placed: '',
        band: [0, 0],
        span: [0, 0],
        mounted: false,
      })
    }
  }

  /**
   * Places the layer and records its signature. A changed signature drops the stored markup and
   * redraws the layer when it is mounted. Returns true when the signature changed.
   */
  update(layer: number, signature: string, box: Box, frame: Frame): boolean {
    const slot = this.slots[layer]
    const placed = `${viewBox(box)}|${frame.scale}|${frame.minX}|${frame.minY}`
    if (placed !== slot.placed) {
      slot.placed = placed
      placeSvg(slot.el, box, frame)
    }
    slot.band = [(box.y - frame.minY) * frame.scale, (box.y + box.h - frame.minY) * frame.scale]
    slot.span = [(box.x - frame.minX) * frame.scale, (box.x + box.w - frame.minX) * frame.scale]
    if (slot.signature === signature) return false
    slot.signature = signature
    slot.markup = null
    if (slot.mounted || slot.kind === 'solids') this.write(layer)
    return true
  }

  /** Drops the stored markup after the caller patched a mounted layer in place. */
  invalidate(layer: number): void {
    const slot = this.slots[layer]
    if (slot) slot.markup = null
  }

  /**
   * Mounts the shelves in the visible band, or with `span` only those on the stage itself, as
   * after a rebuild. Returns how many it mounted.
   */
  check([top, bottom]: [number, number], span?: [number, number]): number {
    let mounted = 0
    for (const [i, slot] of this.slots.entries()) {
      if (slot.kind !== 'shelf' || slot.mounted) continue
      const [a, b] = slot.band
      if (b < top || a > bottom) continue
      if (span && (slot.span[1] < span[0] || slot.span[0] > span[1])) continue
      this.mount(i)
      mounted++
    }
    return mounted
  }

  /**
   * Mounts one shelf ahead of the camera: the first unmounted one of `order`, else the nearest
   * in the margin. Returns false when there was none.
   */
  mountNext(order: readonly number[], [top, bottom]: [number, number], margin: number): boolean {
    for (const i of order)
      if (this.slots[i]?.kind === 'shelf' && !this.slots[i].mounted) {
        this.mount(i)
        return true
      }
    let next = -1
    let gap = Infinity
    for (const [i, slot] of this.slots.entries()) {
      if (slot.kind !== 'shelf' || slot.mounted) continue
      const [a, b] = slot.band
      const d = a > bottom ? a - bottom : top - b
      if (d <= margin && d < gap) {
        gap = d
        next = i
      }
    }
    if (next < 0) return false
    this.mount(next)
    return true
  }

  /**
   * Idle work: mounts every shelf in the margin, unmounts the far ones that `keep` (the shelves
   * ahead of the camera) does not hold, and renders the markup of the rest.
   */
  settle(
    [top, bottom]: [number, number],
    margin: number,
    keep: ReadonlySet<number>,
    deadline: () => number,
  ): boolean {
    for (const [i, slot] of this.slots.entries()) {
      if (deadline() < 1) return true
      if (slot.kind !== 'shelf') continue
      const [a, b] = slot.band
      const near = b >= top - margin && a <= bottom + margin
      const far = b < top - 2 * margin || a > bottom + 2 * margin
      if (near && !slot.mounted) this.mount(i)
      else if (far && slot.mounted && !keep.has(i)) {
        slot.mounted = false
        slot.el.empty()
      }
      if (slot.markup === null) slot.markup = this.source.render(i)
    }
    return false
  }

  private mount(layer: number): void {
    this.slots[layer].mounted = true
    this.write(layer)
  }

  private write(layer: number): void {
    const slot = this.slots[layer]
    slot.markup ??= this.source.render(layer)
    // The markup holds geometry, classes and colour tokens only; `mounted` fills in the text.
    slot.el.innerHTML = this.source.prepare(layer, slot.markup)
    if (slot.kind === 'solids') slot.mounted = true
    this.source.mounted(layer, slot.el)
  }

  clear(): void {
    for (const slot of this.slots) slot.el.remove()
    this.slots = []
    this.shape = ''
  }
}
