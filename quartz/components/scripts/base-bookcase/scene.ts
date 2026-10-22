import type { TopicShelf } from '../../../util/base/bookcase/styles'
import type { Cover } from './covers'
import type { HardcoverBook } from './preview'
import type { Camera, Drag, LiftGeometry } from './state'

// Every binding of one mounted bookcase; the phases below share it.
export interface BookcaseScene {
  root: HTMLElement
  covers: Map<string, Cover>
  scene: SVGSVGElement
  world: HTMLElement
  viewport: HTMLElement
  indexPane: HTMLElement
  dialog: HTMLDialogElement
  scrollSpace: HTMLDivElement
  pan: HTMLDivElement
  viewportWidth: number
  viewportHeight: number
  signal: AbortSignal
  loadTextures: () => void
  reducedMotion: MediaQueryList
  entries: HTMLLIElement[]
  groups: HTMLDetailsElement[]
  books: Map<number, SVGGElement>
  bookBounds: Map<number, DOMRect>
  spineLengths: Map<number, number>
  spineBounds: Map<number, { x: number; y: number; width: number; height: number }>
  cacheBookGeometry: (index: number, book: SVGGElement) => void
  entryByIndex: Map<number, HTMLLIElement>
  residentCovers: Map<string, SVGImageElement>
  prefetched: Map<string, { image: HTMLImageElement; settled: boolean }>
  prefetching: number
  prefetchTask: number | undefined
  idle: boolean
  coverWindow: string
  originX: number
  originY: number
  box: DOMRect
  inventory: Map<
    number,
    {
      path: string
      fullCover: string | undefined
      item: {
        index: number
        slug: string
        title: string
        author: string | undefined
        spine: string
        ink: string
        cover: string | undefined
        coverSlug: string | undefined
        sampleSpine: boolean
        spineArt: string | undefined
        spineArtDetail: string | undefined
        spineArtDetailWidth: number | undefined
        spineArtDetailHeight: number | undefined
      }
    }
  >
  topic: string
  topicShelves: Record<string, TopicShelf>
  arrangement: string
  arrange: (visible: HTMLLIElement[]) => boolean
  readout: HTMLElement | null
  shelfLabel: HTMLElement | null
  previewArt: HTMLElement | null
  hardcover:
    | { show(value: HardcoverBook): Promise<boolean>; reveal(): void; clear(): void }
    | undefined
  previousButton: HTMLButtonElement | null
  nextButton: HTMLButtonElement | null
  zoomOut: HTMLButtonElement | null
  zoomIn: HTMLButtonElement | null
  preloaded: Map<string, HTMLImageElement>
  camera: Camera
  fitScale: number
  minScale: number
  fitWidth: number
  fitted: boolean
  focus: number
  lifted: SVGSVGElement | undefined
  exposedCover: SVGImageElement | undefined
  liftGeometry: LiftGeometry | undefined
  pull: Animation | undefined
  reveal: Animation | undefined
  settling: Set<{ svg: SVGSVGElement; cover?: SVGImageElement; done: () => void }>
  aim: Camera | undefined
  steppedAt: number
  painted: { x: number; y: number; scale: number; at: number }
  velocity: { x: number; y: number }
  drag: Drag | undefined
  frame: number
  wheelTimer: number | undefined
  cancelWheel: () => void
  scrollOrigin: { x: number; y: number }
  scrollLayout: string
  bakedScale: number
  scaleTimer: number | undefined
  paintedScroll: { left: number; top: number }
  cameraDirty: boolean
  previewWheelDelta: number
  previewWheelAt: number
  previewWheelStep: number
  previewIndex: number
  returnFocus: HTMLElement | undefined
  searchOpen: Map<HTMLDetailsElement, boolean> | undefined
  visibleEntries: HTMLLIElement[] | undefined
  matches: () => HTMLLIElement[]
  maxScale: () => number
  clamp: (n: number, min: number, max: number) => number
  bound: (view: Camera) => void
  paint: (now?: number) => void
  refreshShelfCovers: () => void
  schedule: () => void
  settlePrefetch: (url: string) => void
  dropPrefetch: (url: string) => void
  planPrefetch: () => void
  schedulePrefetch: () => void
  fit: () => void
  zoom: (factor: number, x?: number, y?: number) => void
  cloneBook: (index: number) => SVGSVGElement | undefined
  liftedIndex: (svg: SVGSVGElement | undefined) => number
  dropLift: () => void
  sendHome: (
    previous: SVGSVGElement,
    { px, py }: LiftGeometry,
    cover: SVGImageElement | undefined,
  ) => void
  lower: () => void
  lift: (index: number) => void
  placeLift: () => void
  applyWorldScale: () => void
  bakeScale: (scale: number) => void
  copyDetail: (row: HTMLLIElement, target: Element | null) => void
  preloadCovers: (visible: HTMLLIElement[], position: number) => void
  select: (index: number, reveal?: boolean) => void
  deselect: () => void
  focusCamera: (index: number) => void
  renderPreview: (enter?: boolean) => void
  openPreview: (index: number) => void
  browsePreview: (delta: number) => void
  wheelUnit: (event: WheelEvent, height: number) => number
  finishWheel: () => void
  hoverPoint: { x: number; y: number }
  cancelDrag: () => void
  selectFromIndex: (event: Event) => void
  hoverFromIndex: (event: PointerEvent) => void
  groupEntries: Map<HTMLDetailsElement, { entries: HTMLLIElement[]; count: HTMLElement | null }>
  disposeEscape: () => void
  refresh: () => void
  entryTopics: Map<HTMLLIElement, Set<string>>
  topicButtons: HTMLButtonElement[]
  showTopic: (next: string) => void
  topicList: HTMLElement | null
  nextTab: (key: string, active: number) => number | undefined
  updateFit: () => void
  resize: () => void
  observer: ResizeObserver
  initialTopic: string | null
}
