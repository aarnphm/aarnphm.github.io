import { BasesView, type HoverPopover, type QueryController, type WorkspaceLeaf } from 'obsidian'
import type { BookcaseConfig } from './options'
import type { Point, Scene, SceneBook, ShelfGroup } from './scene-model'
import type { SpineFonts } from './spine-fit'
import type { TopicTab } from './topics'
import type { ViewHost } from './view/view-host'
import BookcasePlugin from '../main'
import { Preload, type ViewFilter } from './covers'
import { Drag } from './drag'
import { EditorPane } from './editor-pane'
import { Lift, type LiftRequest } from './lift'
import { ModelReader, type BookModel } from './model'
import { Page } from './page'
import { Panel, type CameraAction } from './panel'
import { LayerStack, Stage, type Frame } from './stage'
import { fillText, type BookPaint } from './svg'
import {
  ahead,
  band,
  camera,
  centre,
  emptyTarget,
  maxZoom,
  onCamera,
  onFullscreen,
  onScroll,
  place,
  reach,
  releasePan,
  requestPlace,
  setControls,
  setFocus,
  zoomBy,
} from './view/camera'
import {
  bookAt,
  buttonIndex,
  hover,
  onKey,
  onPointerMove,
  onStageClick,
  openNote,
  overLift,
} from './view/gestures'
import {
  applyColours,
  bookElement,
  cancelSettle,
  liftBook,
  paint,
  paintLayers,
  patchBook,
  preloadAround,
  prepare,
  render,
  requestSettle,
  setHidden,
  settle,
  syncLift,
  text,
} from './view/layers'
import { onload, onunload } from './view/lifecycle'
import { baseKey, filtered, rebuild, selectTopic, setFilter } from './view/rebuild'
import {
  clearSelection,
  closeHover,
  deselect,
  hostLeaf,
  markSelected,
  moveTo,
  openEditor,
  putBack,
  resolveSelection,
  rove,
  select,
  setLinked,
  summary,
  syncSelection,
  updateIndex,
} from './view/selection'

export const BOOKCASE_VIEW = 'bookcase'

export class BookcaseView extends BasesView implements ViewHost {
  type = BOOKCASE_VIEW
  hoverPopover: HoverPopover | null = null

  root: HTMLElement
  track: HTMLElement
  stageEl: HTMLElement
  world: HTMLElement
  pageEl: HTMLElement
  indexEl: HTMLOListElement
  emptyEl: HTMLElement
  panel: Panel
  drag: Drag
  tabs: TopicTab[] = []
  /** Topic on show; null until read from plugin data. */
  filter: ViewFilter | null = null
  /** Set by a new filter: the rebuild starts at the first book it shows. */
  filterChanged = false
  filterKey: string | null = null
  observer: ResizeObserver | null = null
  stage: Stage
  layers: LayerStack
  lift: Lift
  page: Page
  reader: ModelReader
  preload = new Preload()

  settings: BookcaseConfig | null = null
  scene: Scene<BookModel> | null = null
  /** Resolved spine label fonts; cleared when the theme or fonts change. */
  fonts: SpineFonts | null = null
  /** Paint and camera point of each book, filled on first use. */
  paints: (BookPaint | undefined)[] = []
  centres: (Point | undefined)[] = []
  /** Left and right edges of each book when pulled out, in world px. */
  reaches: (Point | undefined)[] = []
  /** Stage px the control column takes from the right edge. */
  reserve = 0
  frame: Frame = { minX: 0, minY: 0, scale: 1 }
  /** World translation the camera last wrote. */
  offset: Point = [0, 0]
  /** Translation the scroll camera asks for, before zoom and pan. */
  base: Point = [0, 0]
  /** World size in px at zoom 1. */
  size: Point = [0, 0]
  /** Scale on top of the fitted one, from 1 to `maxZoom()`; zoom keeps the anchor in place. */
  zoom = 1
  /** Drag offset in stage px; it eases back to zero on the next focus change. */
  pan: Point = [0, 0]
  releasing = false
  releaseAt = 0
  placeRaf: number | null = null
  fullscreen = false
  /** Set when the scroller changes: the next rebuild scrolls the new one to the focused book. */
  restoreFocus = false
  scrollers = new Set<HTMLElement>()
  dark = false
  /** Holds the finish defs and material tokens every layer references; rewritten when the set changes. */
  defsEl: SVGSVGElement
  defsKey = ''
  /** Book at the reading position: it moves the camera and the roving tabindex, not the selection. */
  focus = -1
  /**
   * Book chosen by a click, a key, keyboard focus, or a wheel or touch scroll (the book at the
   * reading position); -1 for none. A scroll the view makes itself never selects.
   */
  selected = -1
  selectedPath: string | null = null
  /** The selected book is pulled out. A pointer leaving it, a blur or Escape puts it back. */
  lifted = false
  /** A moving pointer was over the lifted book or the slot it left; leaving both puts it back. */
  near = false
  /** The area the lifted book sweeps from its slot, in screen units. */
  swept: { index: number; version: number; area: Point[] } | null = null
  /** Lifted book hover-link fired for, once per lift; -1 for none. */
  hovering = -1
  editor: EditorPane
  /** The editor pane shows the selection, so the page panel is hidden. */
  linked = false
  /** Time of the last wheel or touch move over the view. */
  userScroll = Number.NEGATIVE_INFINITY
  openTimer: number | null = null
  /** Rebuild count, part of each lift key. */
  version = 0
  /** Direction of the last focus change, for the book that lifts next. */
  heading = 1
  /** Index of the button that holds tabindex 0. */
  roving = -1
  /** Index of the button that holds aria-current. */
  marked = -1
  /** Last pointer position over the world, to tell a real move from a scroll's. */
  pointerAt: Point = [Number.NaN, Number.NaN]
  /** Where the last press on the stage went down; a click acts on it. */
  pressed: Element | null = null
  dataChanged = true
  focusPath: string | null = null
  hidden = new Set<number>()
  buttons: HTMLButtonElement[] = []
  /** Layer index of each shelf. */
  shelfLayer = new Map<number, number>()
  /** Place of each book's `<g>` counted from the end of its shelf `<svg>`, where books come last. */
  fromEnd: number[] = []
  /** Books that show each vault cover. */
  coverBooks = new Map<string, number[]>()
  /** Books whose cover finished since the last idle pass. */
  recolour = new Set<number>()
  rebuildRaf: number | null = null
  scrollRaf: number | null = null
  lastScroll = 0
  settleTimer: number | null = null
  settleIdle: (() => void) | null = null
  preloadIdle: (() => void) | null = null
  /** Focus for which windowing last spent its one mount. */
  focusMounts = -1
  indexPending = false

  constructor(
    controller: QueryController,
    public containerEl: HTMLElement,
    public plugin: BookcasePlugin,
  ) {
    super(controller)
    this.reader = new ModelReader(this.app)
    this.editor = new EditorPane(this.app)
    this.root = containerEl.createDiv({ cls: 'bookcase-view', attr: { 'data-layout': 'wide' } })
    this.track = this.root.createDiv({ cls: 'bookcase-track' })
    this.stageEl = this.track.createDiv({ cls: 'bookcase-stage' })
    this.world = this.stageEl.createDiv({ cls: 'bookcase-world' })
    this.emptyEl = this.stageEl.createDiv({
      cls: 'bookcase-empty',
      text: 'No books match this view.',
    })
    this.panel = this.addChild(
      new Panel(this.stageEl, {
        camera: action => this.onCamera(action),
        topic: id => this.selectTopic(id),
      }),
    )
    this.drag = this.addChild(
      new Drag(this.stageEl, '.bookcase-controls', {
        pan: () => this.pan,
        start: () => (this.releasing = false),
        move: pan => {
          this.pan = pan
          this.requestPlace()
        },
      }),
    )
    this.pageEl = this.root.createEl('aside', { cls: 'bookcase-page' })
    this.indexEl = this.root.createEl('ol', {
      cls: 'bookcase-index',
      attr: { 'aria-label': 'Books' },
    })
    this.stage = new Stage(containerEl, this.root, this.track, this.stageEl)
    this.lift = this.addChild(
      new Lift(this.world, (index, hidden) => this.setHidden(index, hidden)),
    )
    // Lifts are created first, so every layer is inserted before them and they paint last.
    this.layers = new LayerStack(this.world, this.world.firstElementChild!, {
      render: layer => this.render(layer),
      prepare: (layer, markup) => this.prepare(layer, markup),
      mounted: (_, el) => fillText(el, key => this.text(key)),
    })
    this.defsEl = this.world.createSvg('svg', {
      cls: 'bookcase-defs',
      attr: { 'aria-hidden': 'true', focusable: 'false', width: 0, height: 0 },
    })
    this.page = new Page(this, this.pageEl, this.reader)
    // The ink pair and fallback hues are literals shared with the garden, so they live in JS.
    this.root.style.setProperty('--bc-ink-dark', '#100F0F')
    this.root.style.setProperty('--bc-ink-light', '#FFFCF0')
  }

  onload(): void {
    return onload.call(this)
  }

  onunload(): void {
    return onunload.call(this)
  }

  onDataUpdated(): void {
    this.dataChanged = true
    this.schedule()
  }

  schedule(): void {
    return schedule.call(this)
  }

  rebuild(): void {
    return rebuild.call(this)
  }

  /**
   * The books on show and their shelf layout. "all" is the view as Bases groups it, in the view's
   * own style; a topic keeps its books, shelved by `topicGroupBy`.
   */
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
  } {
    return filtered.call(this, settings)
  }

  /** Where the filter choice is kept: the .base file and the view name. */
  baseKey(): string {
    return baseKey.call(this)
  }

  setFilter(filter: ViewFilter): void {
    return setFilter.call(this, filter)
  }

  selectTopic(id: string): void {
    return selectTopic.call(this, id)
  }

  onCamera(action: CameraAction): void {
    return onCamera.call(this, action)
  }

  /** Zoom limits as the garden's: from the fitted scale to max(2, 4 × fitted). */
  maxZoom(): number {
    return maxZoom.call(this)
  }

  zoomBy(factor: number): void {
    return zoomBy.call(this, factor)
  }

  setControls(): void {
    return setControls.call(this)
  }

  onFullscreen(): void {
    return onFullscreen.call(this)
  }

  /**
   * Writes the world transform: the scroll camera's `base`, zoomed about the anchor, then the pan.
   * Zoomed or panned, the case keeps at least 30% of the stage, as in the garden.
   */
  place(): void {
    return place.call(this)
  }

  band(): [number, number] {
    return band.call(this)
  }

  /** Pan and zoom frames: a drag, a zoom step, or a released pan easing home. */
  requestPlace(): void {
    return requestPlace.call(this)
  }

  /** A new focus brings the camera back onto it: the pan eases to zero, the zoom stays. */
  releasePan(): void {
    return releasePan.call(this)
  }

  render(index: number): string {
    return render.call(this, index)
  }

  /** Writes the hidden state of a lifted book into its shelf's markup, so a mount sets no attribute. */
  prepare(index: number, markup: string): string {
    return prepare.call(this, index, markup)
  }

  /**
   * Signatures as SPEC 5.6, without the colours: a book whose colour alone changed is patched in
   * place, so a theme switch or a finished cover never rewrites a shelf.
   */
  paintLayers(scene: Scene<BookModel>, previous: (BookPaint | undefined)[]): void {
    return paintLayers.call(this, scene, previous)
  }

  /** Writes a book's new colour into its mounted shelf; an unmounted shelf renders it on mount. */
  patchBook(layer: number, index: number): void {
    return patchBook.call(this, layer, index)
  }

  paint(index: number): BookPaint {
    return paint.call(this, index)
  }

  centre(index: number): Point {
    return centre.call(this, index)
  }

  reach(index: number): Point {
    return reach.call(this, index)
  }

  /** The shelf copy of a book, when its shelf is mounted. */
  bookElement(index: number): Element | null {
    return bookElement.call(this, index)
  }

  /** Colours of books whose cover finished, and the lifted book's cover URL. */
  applyColours(): void {
    return applyColours.call(this)
  }

  requestSettle(): void {
    return requestSettle.call(this)
  }

  cancelSettle(): void {
    return cancelSettle.call(this)
  }

  /** Idle pass: finished covers, then windowing and markup. Returns true when work is left. */
  settle(remaining: () => number): boolean {
    return settle.call(this, remaining)
  }

  text(key: string): string | string[] {
    return text.call(this, key)
  }

  setHidden(index: number, hidden: boolean): void {
    return setHidden.call(this, index, hidden)
  }

  onScroll(evt: Event): void {
    return onScroll.call(this, evt)
  }

  /**
   * Camera, focus and windowing for the current scroll position (SPEC 6.2). After a rebuild
   * (`sync`) the focused book is brought up to date even when the focus did not move.
   */
  camera(sync = false, scrollTop?: number, user = false): void {
    return camera.call(this, sync, scrollTop, user)
  }

  /** Layers of the books within LOOKAHEAD of the focus, nearest first, the heading first. */
  ahead(): number[] {
    return ahead.call(this)
  }

  /** Middle of the case, for a view with no books. */
  emptyTarget(): Point {
    return emptyTarget.call(this)
  }

  /** The reading position moved: the roving tabindex, the covers to decode next, the heading. */
  setFocus(focus: number): void {
    return setFocus.call(this, focus)
  }

  /** After a rebuild: the reading position's tabindex, and the selection's lift and page. */
  syncSelection(): void {
    return syncSelection.call(this)
  }

  /**
   * Selects a book: it lifts, and the page panel or the editor pane shows it. A selection the
   * scroll made (`scrolled`) does not split off the editor pane, and moves an open one only once
   * the scroll rests.
   */
  select(index: number, scrolled = false): void {
    return select.call(this, index, scrolled)
  }

  /** Clears the selection: the book goes back and the page shows the topic and its count. */
  deselect(): void {
    return deselect.call(this)
  }

  /** Forgets the selection; the caller updates the page. */
  clearSelection(): void {
    return clearSelection.call(this)
  }

  /** The selection follows its path through a rebuild; a book gone from the view clears it. */
  resolveSelection(scene: Scene<BookModel>): void {
    return resolveSelection.call(this, scene)
  }

  /** The lifted book goes back to its shelf and its preview closes; the selection stays. */
  putBack(): void {
    return putBack.call(this)
  }

  /** The page panel with nothing selected: the topic on show and its count. */
  summary(): string {
    return summary.call(this)
  }

  /** This view's own leaf in the main area or a popout; null in an embed or a sidebar. */
  hostLeaf(): WorkspaceLeaf | null {
    return hostLeaf.call(this)
  }

  /** Opens `path` in the editor pane, splitting this view's leaf the first time (wide layout). */
  openEditor(path: string): boolean {
    return openEditor.call(this, path)
  }

  /** While the editor pane lives, the page panel is hidden and the stage takes the width. */
  setLinked(on: boolean, rebuild = true): void {
    return setLinked.call(this, on, rebuild)
  }

  /** Closes the page preview the view opened, so none is left behind. */
  closeHover(): void {
    return closeHover.call(this)
  }

  syncLift(index: number): void {
    return syncLift.call(this, index)
  }

  liftBook(index: number): LiftRequest | null {
    return liftBook.call(this, index)
  }

  /**
   * In idle time, decodes the thumbnails of focus ± 3 and writes the book a key selects next into
   * the spare lift.
   */
  preloadAround(): void {
    return preloadAround.call(this)
  }

  /** Roving tabindex on the index: the button of the book at the reading position. */
  rove(index: number): void {
    return rove.call(this, index)
  }

  /** aria-current on the index button of the selected book. */
  markSelected(index: number): void {
    return markSelected.call(this, index)
  }

  updateIndex(scene: Scene<BookModel>): void {
    return updateIndex.call(this, scene)
  }

  buttonIndex(target: EventTarget | null): number | null {
    return buttonIndex.call(this, target)
  }

  onKey(evt: KeyboardEvent): void {
    return onKey.call(this, evt)
  }

  moveTo(index: number, focusButton: boolean): void {
    return moveTo.call(this, index, focusButton)
  }

  bookAt(target: EventTarget | null): SceneBook<BookModel> | null {
    return bookAt.call(this, target)
  }

  /** A book selects it, the lifted book opens its note, the empty stage clears the selection. */
  onStageClick(evt: MouseEvent): void {
    return onStageClick.call(this, evt)
  }

  /** Opens the note: in the editor pane when there is one, else as Obsidian opens a link. */
  openNote(path: string, evt: MouseEvent | KeyboardEvent): void {
    return openNote.call(this, path, evt)
  }

  /**
   * A pointer moving onto the lifted book opens its page preview, once per lift. A pointer that
   * was over the lifted book or its slot and leaves both puts the book back and closes the
   * preview; the selection stays. A scroll that slides the book under a still pointer makes the
   * browser send moves at the same point; those do not count.
   */
  onPointerMove(evt: PointerEvent): void {
    return onPointerMove.call(this, evt)
  }

  /**
   * True when a client point is on the lifted book: pulled out, in the slot it left, or on the
   * way between them, so a pointer can follow the book out.
   */
  overLift(x: number, y: number): boolean {
    return overLift.call(this, x, y)
  }

  hover(event: MouseEvent, targetEl: HTMLElement, linktext: string): void {
    return hover.call(this, event, targetEl, linktext)
  }
}

function schedule(this: ViewHost): void {
  if (this.rebuildRaf !== null) return
  this.rebuildRaf = requestAnimationFrame(() => {
    this.rebuildRaf = null
    this.rebuild()
  })
}
