import { BasesView, type HoverPopover, type WorkspaceLeaf } from 'obsidian'
import type { BookcaseConfig } from '../options'
import type { Point, Scene, SceneBook, ShelfGroup } from '../scene-model'
import type { SpineFonts } from '../spine-fit'
import type { BookPaint } from '../svg'
import type { TopicTab } from '../topics'
import BookcasePlugin from '../../main'
import { Preload, type ViewFilter } from '../covers'
import { Drag } from '../drag'
import { EditorPane } from '../editor-pane'
import { Lift, type LiftRequest } from '../lift'
import { ModelReader, type BookModel } from '../model'
import { Page } from '../page'
import { Panel, type CameraAction } from '../panel'
import { LayerStack, Stage, type Frame } from '../stage'

export interface ViewHost extends BasesView {
  containerEl: HTMLElement
  plugin: BookcasePlugin
  type: string
  hoverPopover: HoverPopover | null
  root: HTMLElement
  track: HTMLElement
  stageEl: HTMLElement
  world: HTMLElement
  pageEl: HTMLElement
  indexEl: HTMLOListElement
  emptyEl: HTMLElement
  panel: Panel
  drag: Drag
  tabs: TopicTab[]
  filter: ViewFilter | null
  filterChanged: boolean
  filterKey: string | null
  observer: ResizeObserver | null
  stage: Stage
  layers: LayerStack
  lift: Lift
  page: Page
  reader: ModelReader
  preload: Preload
  settings: BookcaseConfig | null
  scene: Scene<BookModel> | null
  fonts: SpineFonts | null
  paints: (BookPaint | undefined)[]
  centres: (Point | undefined)[]
  reaches: (Point | undefined)[]
  reserve: number
  frame: Frame
  offset: Point
  base: Point
  size: Point
  zoom: number
  pan: Point
  releasing: boolean
  releaseAt: number
  placeRaf: number | null
  fullscreen: boolean
  restoreFocus: boolean
  scrollers: Set<HTMLElement>
  dark: boolean
  defsEl: SVGSVGElement
  defsKey: string
  focus: number
  selected: number
  selectedPath: string | null
  lifted: boolean
  near: boolean
  swept: { index: number; version: number; area: Point[] } | null
  hovering: number
  editor: EditorPane
  linked: boolean
  userScroll: number
  openTimer: number | null
  version: number
  heading: number
  roving: number
  marked: number
  pointerAt: Point
  pressed: Element | null
  dataChanged: boolean
  focusPath: string | null
  hidden: Set<number>
  buttons: HTMLButtonElement[]
  shelfLayer: Map<number, number>
  fromEnd: number[]
  coverBooks: Map<string, number[]>
  recolour: Set<number>
  rebuildRaf: number | null
  scrollRaf: number | null
  lastScroll: number
  settleTimer: number | null
  settleIdle: (() => void) | null
  preloadIdle: (() => void) | null
  focusMounts: number
  indexPending: boolean
  onload(): void
  onunload(): void
  onDataUpdated(): void
  schedule(): void
  rebuild(): void
  filtered(settings: BookcaseConfig): {
    groups: ShelfGroup<BookModel>[]
    style: any
    material:
      | 'paper'
      | 'walnut'
      | 'birch-ply'
      | 'black-lacquer'
      | 'steel'
      | 'copper'
      | 'concrete'
      | null
    continuous: boolean
    showTags: boolean
  }
  baseKey(): string
  setFilter(filter: ViewFilter): void
  selectTopic(id: string): void
  onCamera(action: CameraAction): void
  maxZoom(): number
  zoomBy(factor: number): void
  setControls(): void
  onFullscreen(): void
  place(): void
  band(): [number, number]
  requestPlace(): void
  releasePan(): void
  render(index: number): string
  prepare(index: number, markup: string): string
  paintLayers(scene: Scene<BookModel>, previous: (BookPaint | undefined)[]): void
  patchBook(layer: number, index: number): void
  paint(index: number): BookPaint
  centre(index: number): Point
  reach(index: number): Point
  bookElement(index: number): Element | null
  applyColours(): void
  requestSettle(): void
  cancelSettle(): void
  settle(remaining: () => number): boolean
  text(key: string): string | string[]
  setHidden(index: number, hidden: boolean): void
  onScroll(evt: Event): void
  camera(sync?: boolean, scrollTop?: number, user?: boolean): void
  ahead(): number[]
  emptyTarget(): Point
  setFocus(focus: number): void
  syncSelection(): void
  select(index: number, scrolled?: boolean): void
  deselect(): void
  clearSelection(): void
  resolveSelection(scene: Scene<BookModel>): void
  putBack(): void
  summary(): string
  hostLeaf(): WorkspaceLeaf | null
  openEditor(path: string): boolean
  setLinked(on: boolean, rebuild?: boolean): void
  closeHover(): void
  syncLift(index: number): void
  liftBook(index: number): LiftRequest | null
  preloadAround(): void
  rove(index: number): void
  markSelected(index: number): void
  updateIndex(scene: Scene<BookModel>): void
  buttonIndex(target: EventTarget | null): number | null
  onKey(evt: KeyboardEvent): void
  moveTo(index: number, focusButton: boolean): void
  bookAt(target: EventTarget | null): SceneBook<BookModel> | null
  onStageClick(evt: MouseEvent): void
  openNote(path: string, evt: MouseEvent | KeyboardEvent): void
  onPointerMove(evt: PointerEvent): void
  overLift(x: number, y: number): boolean
  hover(event: MouseEvent, targetEl: HTMLElement, linktext: string): void
}
