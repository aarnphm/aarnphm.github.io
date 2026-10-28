import type { CardLayer, CardScene, LayerBounds } from '../scripts/press-kit-layers'
import type { RulerPoint } from '../scripts/press-kit-rulers'

interface Placement {
  x: number
  y: number
  scaleX: number
  scaleY: number
  hidden: boolean
}
export type ResizeAxis = 'x' | 'y' | 'xy'
export const resizeHandles: { axis: ResizeAxis; label: string }[] = [
  { axis: 'x', label: 'largeur' },
  { axis: 'y', label: 'hauteur' },
  { axis: 'xy', label: 'largeur et hauteur' },
]
interface Background {
  image: HTMLImageElement
  name: string
  fit: 'cover' | 'contain'
  zoom: number
  x: number
  y: number
  opacity: number
}
export interface Composition {
  placements: Record<string, Placement>
  order: string[]
}
export interface Layout {
  formats: Record<string, Composition>
  metricGap: number
  railTextScale: number
  textColor: 'auto' | 'black' | 'white'
  title?: string
  background?: Background
}
export interface Draft {
  value: Layout
  past: Layout[]
  future: Layout[]
}
interface LayerPlacement {
  layer: CardLayer
  placement: Placement
}
interface SelectionBand {
  start: number
  end: number
  items: LayerPlacement[]
}
interface PointerGesture {
  pointer: number
  x: number
  y: number
  selection: string[]
}
export type Gesture = PointerGesture &
  (
    | { kind: 'marquee'; additive: boolean; bounds: LayerBounds }
    | {
        kind: 'transform'
        initial: LayerPlacement[]
        bounds: LayerBounds
        before: Layout
        resize: ResizeAxis | null
        point: RulerPoint
        moveAxis?: 'x' | 'y'
        moved: boolean
        toggleOnClick?: string
      }
  )
export interface LayerRaster {
  width: number
  height: number
  image: CanvasImageSource
  pending?: Promise<void>
  failed?: { width: number; height: number }
}
export interface EditorState {
  canvas: HTMLCanvasElement
  signal: AbortSignal
  changed: () => void
  requestScene: () => void
  defaultTitle: () => string
  syncView: () => void
  details: HTMLDetailsElement
  controls: HTMLDivElement
  overlay: HTMLDivElement
  rulers: HTMLDivElement
  showRulers: boolean
  rulerToggle: HTMLButtonElement | null
  pointer?: RulerPoint
  file: HTMLInputElement
  drafts: Map<string, Draft>
  draft: Draft
  scene?: CardScene
  rasters: Map<CardLayer, LayerRaster>
  selected: Set<string>
  gesture?: Gesture
  frame: number
  revision: number
  pendingImage?: Promise<void>
  key: string
  formatKey: string
  railAnchors: Map<string, { right: number; bottom: number }>
  available: boolean
  enabled: boolean
  loading: boolean
  error: string
  observer: ResizeObserver
}

export const empty = (): Layout => ({
  formats: {},
  metricGap: 0,
  railTextScale: 1,
  textColor: 'auto',
})

export const copy = (layout: Layout): Layout => ({
  metricGap: layout.metricGap,
  railTextScale: layout.railTextScale,
  textColor: layout.textColor,
  title: layout.title,
  formats: Object.fromEntries(
    Object.entries(layout.formats).map(([key, composition]) => [
      key,
      {
        placements: Object.fromEntries(
          Object.entries(composition.placements).map(([id, value]) => [id, { ...value }]),
        ),
        order: [...composition.order],
      },
    ]),
  ),
  background: layout.background ? { ...layout.background } : undefined,
})

export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value))

export const composition = (state: EditorState): Composition =>
  (state.draft.value.formats[state.formatKey] ??= { placements: {}, order: [] })

export const placement = (state: EditorState, layer: CardLayer): Placement =>
  composition(state).placements[layer.id] ?? {
    x: layer.x,
    y: layer.y,
    scaleX: 1,
    scaleY: 1,
    hidden: false,
  }

export const layers = (state: EditorState): CardLayer[] => {
  const sceneLayers = state.scene?.layers ?? []
  const order = composition(state).order
  return [...sceneLayers].sort((a, b) => {
    const ai = order.indexOf(a.id),
      bi = order.indexOf(b.id)
    return (
      (ai < 0 ? sceneLayers.indexOf(a) : sceneLayers.length + ai) -
      (bi < 0 ? sceneLayers.indexOf(b) : sceneLayers.length + bi)
    )
  })
}

export const selection = (state: EditorState): LayerPlacement[] =>
  layers(state)
    .filter(layer => state.selected.has(layer.id))
    .map(layer => ({ layer, placement: { ...placement(state, layer) } }))

export const bounds = (items: LayerPlacement[]): LayerBounds | undefined => {
  if (!items.length) return
  const x = Math.min(...items.map(({ placement: p }) => p.x))
  const y = Math.min(...items.map(({ placement: p }) => p.y))
  return {
    x,
    y,
    width: Math.max(...items.map(({ layer, placement: p }) => p.x + layer.width * p.scaleX)) - x,
    height: Math.max(...items.map(({ layer, placement: p }) => p.y + layer.height * p.scaleY)) - y,
  }
}

const isMetricSelection = (items: LayerPlacement[]) =>
  items.length > 1 && items.every(({ layer }) => layer.id.includes('/stat/'))

const metricAxis = (items: LayerPlacement[], axis: 'x' | 'y') => {
  const dimension = axis === 'x' ? 'width' : 'height'
  const scale = axis === 'x' ? 'scaleX' : 'scaleY'
  const size = ({ layer, placement: p }: LayerPlacement) => layer[dimension] * p[scale]
  const center = (item: LayerPlacement) => item.placement[axis] + size(item) / 2
  const bands: SelectionBand[] = []
  for (const item of [...items].sort((a, b) => center(a) - center(b))) {
    const start = item.placement[axis]
    const end = start + size(item)
    const band = bands.at(-1)
    const centers = band?.items.map(center).sort((a, b) => a - b) ?? []
    const middle = centers[Math.floor(centers.length / 2)] ?? center(item)
    // Align a row by its center, so the small raster padding does not join adjacent rows.
    if (
      band &&
      Math.abs(center(item) - middle) <= Math.min(size(item), ...band.items.map(size)) / 2
    ) {
      band.start = Math.min(band.start, start)
      band.end = Math.max(band.end, end)
      band.items.push(item)
    } else bands.push({ start, end, items: [item] })
  }
  const gaps = bands.slice(1).map((band, index) => band.start - bands[index].end)
  const minimum =
    bands.reduce((total, band) => total + band.end - band.start, 0) +
    gaps.reduce((total, gap) => total + Math.min(0, gap), 0)
  return { bands, gaps, minimum }
}

export const resizeLimits = (state: EditorState, items: LayerPlacement[]) => {
  const itemBounds = bounds(items)
  if (isMetricSelection(items) && itemBounds) {
    const x = metricAxis(items, 'x')
    const y = metricAxis(items, 'y')
    return {
      minX: x.bands.length > 1 ? x.minimum / itemBounds.width : 1,
      minY: y.bands.length > 1 ? y.minimum / itemBounds.height : 1,
      maxX:
        x.bands.length > 1
          ? Math.max(1, ((state.scene?.width ?? itemBounds.width) * 5) / itemBounds.width)
          : 1,
      maxY:
        y.bands.length > 1
          ? Math.max(1, ((state.scene?.height ?? itemBounds.height) * 5) / itemBounds.height)
          : 1,
    }
  }
  return {
    minX: Math.max(
      ...items.map(
        ({ layer, placement: p }) =>
          Math.max(0.1, (layer.renderer?.minWidth ?? 0) / layer.width) / p.scaleX,
      ),
    ),
    minY: Math.max(
      ...items.map(
        ({ layer, placement: p }) =>
          Math.max(0.1, (layer.renderer?.minHeight ?? 0) / layer.height) / p.scaleY,
      ),
    ),
    maxX: Math.min(...items.map(({ placement: p }) => 5 / p.scaleX)),
    maxY: Math.min(...items.map(({ placement: p }) => 5 / p.scaleY)),
  }
}

export const resizedBounds = (
  state: EditorState,
  items: LayerPlacement[],
  itemBounds: LayerBounds,
  dx: number,
  dy: number,
  axis: ResizeAxis,
  lock = false,
) => {
  const { minX, minY, maxX, maxY } = resizeLimits(state, items)
  const { width, height } = itemBounds
  if (axis === 'xy' && lock) {
    const factor = clamp(
      1 + (dx * width + dy * height) / (width ** 2 + height ** 2),
      Math.max(minX, minY),
      Math.min(maxX, maxY),
    )
    return { ...itemBounds, width: width * factor, height: height * factor }
  }
  return {
    ...itemBounds,
    width: axis === 'y' ? width : width * clamp(1 + dx / width, minX, maxX),
    height: axis === 'x' ? height : height * clamp(1 + dy / height, minY, maxY),
  }
}

export const placeSelection = (
  comp: Composition,
  items: LayerPlacement[],
  before: LayerBounds,
  after: LayerBounds,
) => {
  const sx = after.width / before.width,
    sy = after.height / before.height
  if (isMetricSelection(items) && (sx !== 1 || sy !== 1)) {
    const distribute = (axis: 'x' | 'y', start: number, span: number) => {
      const { bands, gaps, minimum } = metricAxis(items, axis)
      const free = gaps.reduce((total, gap) => total + Math.max(0, gap), 0)
      const available = Math.max(0, span - minimum)
      let position = start
      for (const [index, band] of bands.entries()) {
        for (const { layer, placement: p } of band.items)
          comp.placements[layer.id] = {
            ...(comp.placements[layer.id] ?? p),
            [axis]: p[axis] + position - band.start,
            scaleX: p.scaleX,
            scaleY: p.scaleY,
          }
        position += band.end - band.start
        const gap = gaps[index]
        if (gap !== undefined)
          position +=
            Math.min(0, gap) + available * (free > 0 ? Math.max(0, gap) / free : 1 / gaps.length)
      }
    }
    distribute('x', after.x, after.width)
    distribute('y', after.y, after.height)
    return
  }
  for (const { layer, placement: p } of items)
    comp.placements[layer.id] = {
      ...p,
      x: after.x + (p.x - before.x) * sx,
      y: after.y + (p.y - before.y) * sy,
      scaleX: p.scaleX * sx,
      scaleY: p.scaleY * sy,
    }
}

export const selectLayer = (state: EditorState, id: string, additive: boolean) => {
  if (additive) {
    if (state.selected.has(id)) state.selected.delete(id)
    else state.selected.add(id)
  } else if (!state.selected.has(id)) state.selected = new Set([id])
}

export const remember = (state: EditorState, before: Layout) => {
  state.draft.past.push(before)
  if (state.draft.past.length > 30) state.draft.past.shift()
  state.draft.future = []
}
