import {
  type BasesPropertyId,
  BasesView,
  type HoverPopover,
  Keymap,
  type QueryController,
  type WorkspaceLeaf,
} from 'obsidian'
import type BookcasePlugin from '../main'
import { Preload, type ViewFilter } from './covers'
import { Drag } from './drag'
import { EditorPane } from './editor-pane'
import { Lift, type LiftRequest, coverLines, liftContent } from './lift'
import { type BookModel, ModelReader, authorLabel } from './model'
import { type BookcaseConfig, readConfig } from './options'
import { Page } from './page'
import { type CameraAction, Panel } from './panel'
import { type Point, type Scene, type ShelfMaterial, buildScene } from './scene-model'
import { type SpineFonts, fitSpine, spineFonts } from './spine-fit'
import { type Frame, LayerStack, Stage, cameraTarget, toPx } from './stage'
import {
  type BookPaint,
  bookStyle,
  fillText,
  materialMarkup,
  paintBook,
  partsSignature,
  spineLabels,
  shelfBox,
  shelfMarkup,
  solidsBox,
  solidsMarkup,
} from './svg'
import { ALL, type TopicTab, lookupTopic, readTopicConfig, topicMembers, topicTabs } from './topics'

export const BOOKCASE_VIEW = 'bookcase'

const PRELOAD_RADIUS = 3
/** The tags a book carries, for tag topics. */
const TAGS = 'file.tags' as BasesPropertyId
/** Books either side of the focus whose shelves mount ahead of the camera. */
const LOOKAHEAD = 12
/** Idle work waits until the scroll has rested this long. */
const QUIET_MS = 160
/** One zoom step, as the garden's buttons. */
const ZOOM = 1.35
/** A released pan closes about 95% of the way back in three of these. */
const GLIDE_MS = 90
/** A scroll this soon after a wheel or touch move is the user's, and selects. */
const USER_SCROLL_MS = 250
/** A book the scroll selected opens in the editor pane once the scroll rests this long. */
const SCROLL_OPEN_MS = 300

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n))

/** The area a convex outline sweeps when it moves by `d`: the hull of both positions. */
function sweep(points: readonly Point[], [dx, dy]: Point): Point[] {
  const all = [...points, ...points.map(([x, y]): Point => [x + dx, y + dy])].sort(
    (a, b) => a[0] - b[0] || a[1] - b[1],
  )
  const cross = (o: Point, a: Point, b: Point) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const half = (list: Point[]) => {
    const out: Point[] = []
    for (const p of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop()
      out.push(p)
    }
    out.pop()
    return out
  }
  return [...half(all), ...half([...all].reverse())]
}

/** Even-odd test of a point against a closed polygon. */
function contains(points: readonly Point[], x: number, y: number): boolean {
  let inside = false
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i]
    const [xj, yj] = points[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** Runs `work` in idle time with the time it has left; returns a cancel function. */
function idle(work: (remaining: () => number) => void, timeout: number): () => void {
  if (typeof window.requestIdleCallback === 'function') {
    const id = window.requestIdleCallback(deadline => work(() => deadline.timeRemaining()), {
      timeout,
    })
    return () => window.cancelIdleCallback(id)
  }
  const id = window.setTimeout(() => work(() => 8), 0)
  return () => window.clearTimeout(id)
}

export class BookcaseView extends BasesView {
  type = BOOKCASE_VIEW
  hoverPopover: HoverPopover | null = null

  private root: HTMLElement
  private track: HTMLElement
  private stageEl: HTMLElement
  private world: HTMLElement
  private pageEl: HTMLElement
  private indexEl: HTMLOListElement
  private emptyEl: HTMLElement
  private panel: Panel
  private drag: Drag
  private tabs: TopicTab[] = []
  /** Topic on show; null until read from plugin data. */
  private filter: ViewFilter | null = null
  /** Set by a new filter: the rebuild starts at the first book it shows. */
  private filterChanged = false
  private filterKey: string | null = null
  private observer: ResizeObserver | null = null
  private stage: Stage
  private layers: LayerStack
  private lift: Lift
  private page: Page
  private reader: ModelReader
  private preload = new Preload()

  private settings: BookcaseConfig | null = null
  private scene: Scene<BookModel> | null = null
  /** Resolved spine label fonts; cleared when the theme or fonts change. */
  private fonts: SpineFonts | null = null
  /** Paint and camera point of each book, filled on first use. */
  private paints: (BookPaint | undefined)[] = []
  private centres: (Point | undefined)[] = []
  /** Left and right edges of each book when pulled out, in world px. */
  private reaches: (Point | undefined)[] = []
  /** Stage px the control column takes from the right edge. */
  private reserve = 0
  private frame: Frame = { minX: 0, minY: 0, scale: 1 }
  /** World translation the camera last wrote. */
  private offset: Point = [0, 0]
  /** Translation the scroll camera asks for, before zoom and pan. */
  private base: Point = [0, 0]
  /** World size in px at zoom 1. */
  private size: Point = [0, 0]
  /** Scale on top of the fitted one, from 1 to `maxZoom()`; zoom keeps the anchor in place. */
  private zoom = 1
  /** Drag offset in stage px; it eases back to zero on the next focus change. */
  private pan: Point = [0, 0]
  private releasing = false
  private releaseAt = 0
  private placeRaf: number | null = null
  private fullscreen = false
  /** Set when the scroller changes: the next rebuild scrolls the new one to the focused book. */
  private restoreFocus = false
  private scrollers = new Set<HTMLElement>()
  private dark = false
  /** Holds the finish defs and material tokens every layer references; rewritten when the set changes. */
  private defsEl: SVGSVGElement
  private defsKey = ''
  /** Book at the reading position: it moves the camera and the roving tabindex, not the selection. */
  private focus = -1
  /**
   * Book chosen by a click, a key, keyboard focus, or a wheel or touch scroll (the book at the
   * reading position); -1 for none. A scroll the view makes itself never selects.
   */
  private selected = -1
  private selectedPath: string | null = null
  /** The selected book is pulled out. A pointer leaving it, a blur or Escape puts it back. */
  private lifted = false
  /** A moving pointer was over the lifted book or the slot it left; leaving both puts it back. */
  private near = false
  /** The area the lifted book sweeps from its slot, in screen units. */
  private swept: { index: number; version: number; area: Point[] } | null = null
  /** Lifted book hover-link fired for, once per lift; -1 for none. */
  private hovering = -1
  private editor: EditorPane
  /** The editor pane shows the selection, so the page panel is hidden. */
  private linked = false
  /** Time of the last wheel or touch move over the view. */
  private userScroll = Number.NEGATIVE_INFINITY
  private openTimer: number | null = null
  /** Rebuild count, part of each lift key. */
  private version = 0
  /** Direction of the last focus change, for the book that lifts next. */
  private heading = 1
  /** Index of the button that holds tabindex 0. */
  private roving = -1
  /** Index of the button that holds aria-current. */
  private marked = -1
  /** Last pointer position over the world, to tell a real move from a scroll's. */
  private pointerAt: Point = [Number.NaN, Number.NaN]
  /** Where the last press on the stage went down; a click acts on it. */
  private pressed: Element | null = null
  private dataChanged = true
  private focusPath: string | null = null
  private hidden = new Set<number>()
  private buttons: HTMLButtonElement[] = []
  /** Layer index of each shelf. */
  private shelfLayer = new Map<number, number>()
  /** Place of each book's `<g>` counted from the end of its shelf `<svg>`, where books come last. */
  private fromEnd: number[] = []
  /** Books that show each vault cover. */
  private coverBooks = new Map<string, number[]>()
  /** Books whose cover finished since the last idle pass. */
  private recolour = new Set<number>()
  private rebuildRaf: number | null = null
  private scrollRaf: number | null = null
  private lastScroll = 0
  private settleTimer: number | null = null
  private settleIdle: (() => void) | null = null
  private preloadIdle: (() => void) | null = null
  /** Focus for which windowing last spent its one mount. */
  private focusMounts = -1
  private indexPending = false

  constructor(
    controller: QueryController,
    private containerEl: HTMLElement,
    private plugin: BookcasePlugin,
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
    this.registerEvent(
      this.app.workspace.on('css-change', () => {
        this.fonts = null
        this.schedule()
      }),
    )
    this.registerEvent(
      // A finished cover patches only its books, in the next idle pass (SPEC 7).
      this.plugin.covers.on('cover', path => {
        const books = this.coverBooks.get(path)
        if (!books) return
        for (const i of books) this.recolour.add(i)
        this.requestSettle()
      }),
    )
    // The track height set by a rebuild resizes the container too; only a new width, or a new
    // scroller height, rebuilds. The scroller is observed once it is found.
    const observer = new ResizeObserver(() => {
      if (this.stage.resized()) this.schedule()
    })
    observer.observe(this.containerEl)
    this.observer = observer
    this.register(() => observer.disconnect())
    // A closed editor pane hands the selection back to the page panel.
    this.registerEvent(
      this.app.workspace.on('layout-change', () => {
        if (this.linked && !this.editor.alive()) this.setLinked(false)
      }),
    )
    // Plus and minus zoom while focus is inside the view.
    this.registerDomEvent(this.root, 'keydown', evt => {
      if (evt.defaultPrevented || evt.ctrlKey || evt.metaKey || evt.altKey) return
      if (evt.key === '+' || evt.key === '=') this.zoomBy(ZOOM)
      else if (evt.key === '-') this.zoomBy(1 / ZOOM)
      else return
      evt.preventDefault()
    })
    // Escape clears the selection while focus is in the view, or the pointer is over it: a wheel
    // selects without moving focus.
    const doc = this.root.ownerDocument
    this.registerDomEvent(doc, 'keydown', evt => {
      if (evt.key !== 'Escape' || evt.defaultPrevented || this.selected < 0) return
      if (!this.root.contains(doc.activeElement) && !this.root.matches(':hover')) return
      evt.preventDefault()
      this.deselect()
    })
    // Focus that leaves the view puts the lifted book back; the selection stays. A press on the
    // stage drops focus to nothing, with the pointer still over the view: that is not a blur.
    this.registerDomEvent(this.root, 'focusout', evt => {
      const next = evt.relatedTarget
      if (next instanceof Node ? this.root.contains(next) : this.root.matches(':hover')) return
      this.putBack()
    })
    this.registerDomEvent(document, 'fullscreenchange', () => this.onFullscreen())
    // Only a scroll the user makes selects: one that follows a wheel or a touch move.
    const user = () => (this.userScroll = performance.now())
    this.registerDomEvent(this.root, 'wheel', user, { passive: true })
    this.registerDomEvent(this.root, 'touchmove', user, { passive: true })

    this.registerDomEvent(
      this.stageEl,
      'pointerdown',
      evt => (this.pressed = evt.target as Element),
    )
    this.registerDomEvent(this.stageEl, 'click', evt => this.onStageClick(evt))
    this.registerDomEvent(this.stageEl, 'pointermove', evt => this.onPointerMove(evt))
    this.registerDomEvent(this.stageEl, 'pointerleave', evt => {
      if (evt.pointerType !== 'touch' && this.near) this.putBack()
    })
    this.registerDomEvent(this.pageEl, 'click', evt => {
      const link = (evt.target as Element).closest<HTMLElement>('a.internal-link')
      const href = link?.dataset.href
      if (!href) return
      evt.preventDefault()
      void this.app.workspace.openLinkText(href, this.page.sourcePath, Keymap.isModEvent(evt))
    })
    this.registerDomEvent(this.pageEl, 'mouseover', evt => {
      const link = (evt.target as Element).closest<HTMLElement>('a.internal-link')
      const href = link?.dataset.href
      if (link && href) this.hover(evt, link, href)
    })
    this.registerDomEvent(this.indexEl, 'keydown', evt => this.onKey(evt))
    this.registerDomEvent(this.indexEl, 'click', evt => {
      const i = this.buttonIndex(evt.target)
      if (i === null) return
      this.select(i)
      this.moveTo(i, false)
    })
    this.registerDomEvent(this.indexEl, 'focusin', evt => {
      const visible = (evt.target as Element).matches(':focus-visible')
      this.lift.setFocusVisible(visible)
      // Keyboard focus is a choice; the focus a click gives its button selected already.
      const i = this.buttonIndex(evt.target)
      if (!visible || i === null || i === this.selected) return
      this.select(i)
      this.moveTo(i, false)
    })
    this.registerDomEvent(this.indexEl, 'focusout', () => this.lift.setFocusVisible(false))
    this.schedule()
  }

  onunload(): void {
    if (this.rebuildRaf !== null) cancelAnimationFrame(this.rebuildRaf)
    if (this.scrollRaf !== null) cancelAnimationFrame(this.scrollRaf)
    if (this.placeRaf !== null) cancelAnimationFrame(this.placeRaf)
    this.rebuildRaf = this.scrollRaf = this.placeRaf = null
    if (this.openTimer !== null) window.clearTimeout(this.openTimer)
    this.openTimer = null
    this.closeHover()
    // The pane may hold unsaved edits: the view lets go of it and never closes it.
    this.editor.forget()
    if (this.stage.fullscreen) void document.exitFullscreen().catch(() => undefined)
    this.cancelSettle()
    this.page.unload()
    this.plugin.covers.release(this)
    this.preload.clear()
    this.layers.clear()
    this.containerEl.empty()
  }

  onDataUpdated(): void {
    this.dataChanged = true
    this.schedule()
  }

  private schedule(): void {
    if (this.rebuildRaf !== null) return
    this.rebuildRaf = requestAnimationFrame(() => {
      this.rebuildRaf = null
      this.rebuild()
    })
  }

  private rebuild(): void {
    if (!this.data) return
    // Until the view has a box there is nothing to measure; the resize observer calls again.
    const scroller = this.stage.scroller ?? this.stage.discover()
    if (!scroller) return
    if (!this.scrollers.has(scroller)) {
      // A view that went full screen and back has two scrollers; only the current one counts.
      this.scrollers.add(scroller)
      this.registerDomEvent(scroller, 'scroll', evt => this.onScroll(evt), { passive: true })
      if (scroller !== this.root) this.observer?.observe(scroller)
    }

    const settings = readConfig(this.config)
    this.settings = settings
    if (this.linked && !settings.editorPane) {
      this.editor.forget()
      this.setLinked(false, false)
    }
    this.dark = document.body.hasClass('theme-dark')
    const { groups, style, material, continuous, showTags } = this.filtered(settings)
    const scene = buildScene(
      groups,
      {
        bookHeight: settings.bookHeight,
        shelfWidth: settings.shelfWidth,
        continuous,
        style,
        material,
        showTags,
        rows: settings.shelfRows,
      },
      model => ({ title: model.spine, author: authorLabel(model.authors) || undefined }),
    )
    const count = scene.books.length
    this.root.toggleClass('is-empty', count === 0)
    const materials = new Set<ShelfMaterial>()
    for (const layer of scene.layers) {
      for (const solid of layer.solids) materials.add(solid.material)
      for (const halo of layer.halos) materials.add(halo.material)
    }
    const defsKey = [...materials].sort().join(' ')
    if (defsKey !== this.defsKey) {
      this.defsKey = defsKey
      this.defsEl.innerHTML = materialMarkup(materials, this.plugin.grainAssets)
    }

    const [minX, minY, maxX, maxY] = scene.bounds
    const metrics = this.stage.measure(
      count,
      settings.scrollStep,
      scene.fit,
      scene.span,
      minX,
      maxX - minX,
      maxY - minY,
      this.panel.inset(),
      !this.linked && this.marked >= 0,
    )
    const { scale } = metrics
    this.frame = { minX, minY, scale }
    this.size = [(maxX - minX) * scale, (maxY - minY) * scale]
    this.world.style.width = `${this.size[0]}px`
    this.world.style.height = `${this.size[1]}px`
    this.zoom = Math.min(this.zoom, this.maxZoom())

    const previous = this.paints
    this.scene = scene
    this.version++
    this.paints = []
    this.centres = []
    this.reaches = []
    this.shelfLayer.clear()
    this.fromEnd = []
    for (const [i, layer] of scene.layers.entries()) {
      if (layer.kind !== 'shelf') continue
      this.shelfLayer.set(layer.shelf, i)
      for (const [p, index] of layer.books.entries()) this.fromEnd[index] = layer.books.length - p
    }
    this.coverBooks.clear()
    this.recolour.clear()
    const files = scene.books.map(book => {
      const cover = book.item.cover
      if (cover?.kind !== 'vault') return null
      const list = this.coverBooks.get(cover.file.path)
      if (list) list.push(book.index)
      else this.coverBooks.set(cover.file.path, [book.index])
      return cover.file
    })
    this.paintLayers(scene, previous)
    // The first index waits for the idle pass; later ones follow the data at once.
    if (this.buttons.length > 0) this.updateIndex(scene)
    else this.indexPending = true

    // Keep the focused book by path; when it is gone, keep its index. Only a shift moves the
    // scroll position, so a rebuild never cancels a smooth scroll in progress.
    let focus = this.focus
    let top: number | undefined = metrics.scrollTop
    if (this.filterChanged) {
      // A new topic starts at its first book, from the top, with the whole case in view.
      this.filterChanged = false
      this.focus = focus = -1
      this.focusPath = null
      this.clearSelection()
      this.zoom = 1
      this.pan = [0, 0]
      this.releasing = false
      scroller.scrollTop = 0
      top = 0
    }
    if (this.restoreFocus) {
      this.restoreFocus = false
      if (focus >= 0) {
        scroller.scrollTop = metrics.trackTop + Math.min(focus, count - 1) * settings.scrollStep
        top = undefined
      }
    }
    if (count > 0 && focus >= 0) {
      focus = Math.min(focus, count - 1)
      if (this.focusPath && scene.books[focus]?.path !== this.focusPath) {
        let best = -1
        for (const book of scene.books)
          if (
            book.path === this.focusPath &&
            (best < 0 || Math.abs(book.index - this.focus) < Math.abs(best - this.focus))
          )
            best = book.index
        if (best >= 0) focus = best
      }
      if (focus !== this.focus) {
        // The browser may clamp the offset, so the camera reads it back (a rare layout read).
        scroller.scrollTop = metrics.trackTop + focus * settings.scrollStep
        top = undefined
      }
    }
    this.plugin.covers.need(this, files, Math.max(0, focus))
    this.resolveSelection(scene)
    this.camera(true, top)
    this.setControls()
    this.dataChanged = false
    // Shelves outside the stage, and the markup of the others, are left to idle time.
    this.requestSettle()
  }

  /**
   * The books on show and their shelf layout. "all" is the view as Bases groups it, in the view's
   * own style; a topic keeps its books, shelved by `topicGroupBy`.
   */
  private filtered(settings: BookcaseConfig) {
    const data = this.data
    const config = readTopicConfig(this.config)
    const entries = data.data
    const tagsOf = new Map(
      entries.map(e => [
        e.file.path,
        this.reader.strings(e, TAGS).map(tag => tag.replace(/^#/, '')),
      ]),
    )
    const tags = (path: string) => tagsOf.get(path) ?? []
    const pairs: [string | null, string][] = []
    let total = 0
    for (const group of data.groupedData) {
      const key = group.hasKey() && group.key ? group.key.toString() : null
      for (const entry of group.entries) pairs.push([key, entry.file.path])
      total += group.entries.length
    }
    const members = topicMembers(pairs, tags, config)
    const tabs = topicTabs(members, total, config)
    this.filterKey ??= this.baseKey()
    const saved = this.filter ?? this.plugin.covers.filter(this.filterKey)
    const topic = saved && tabs.some(tab => tab.id === saved.topic) ? saved.topic : ALL
    this.filter = { topic }
    this.tabs = tabs
    this.panel.render(tabs, topic)
    this.reserve = this.panel.reserve()

    const shelf = lookupTopic(topic, config).shelf
    const style = shelf?.style ?? settings.shelfStyle
    // As the garden: the topic's shelf, else the view's options, which also style "all".
    const material = shelf?.material ?? settings.shelfMaterial
    const showTags = shelf?.showTags ?? settings.showTags
    if (this.root.dataset.topic !== topic) this.root.dataset.topic = topic
    if (this.root.dataset.shelfStyle !== style) this.root.dataset.shelfStyle = style
    // A topic's shelves group by small subcategories, so they share runs, as in the garden.
    const continuous = topic !== ALL
    const set = members.get(topic)
    const groups = continuous
      ? this.reader.topicGroups(
          entries.filter(entry => !set || set.has(entry.file.path)),
          config.shelvesBy,
          settings,
        )
      : this.reader.groups(data.groupedData, settings)
    return { groups, style, material, continuous, showTags }
  }

  /** Where the filter choice is kept: the .base file and the view name. */
  private baseKey(): string {
    let path = ''
    const embed = this.containerEl.closest('.internal-embed, .bases-embed')
    const src = embed?.getAttribute('src')
    if (src) {
      const link = src.split('#')[0]
      path = this.app.metadataCache.getFirstLinkpathDest(link, '')?.path ?? link
    } else
      for (const leaf of this.app.workspace.getLeavesOfType('bases')) {
        const view = leaf.view as { containerEl?: HTMLElement; file?: { path: string } | null }
        if (view.containerEl?.contains(this.containerEl)) path = view.file?.path ?? ''
      }
    return `${path}#${this.config.name}`
  }

  private setFilter(filter: ViewFilter): void {
    this.filter = filter
    this.filterChanged = true
    if (this.filterKey !== null) this.plugin.covers.setFilter(this.filterKey, filter)
    this.lift.clear()
    this.dataChanged = true
    this.schedule()
  }

  private selectTopic(id: string): void {
    if (this.filter && id !== this.filter.topic) this.setFilter({ topic: id })
  }

  private onCamera(action: CameraAction): void {
    if (action === 'in') this.zoomBy(ZOOM)
    else if (action === 'out') this.zoomBy(1 / ZOOM)
    else if (action === 'fullscreen') {
      if (this.stage.fullscreen) void document.exitFullscreen().catch(() => undefined)
      else void this.root.requestFullscreen().catch(() => undefined)
    } else {
      this.zoom = 1
      this.pan = [0, 0]
      this.releasing = false
      this.requestPlace()
    }
  }

  /** Zoom limits as the garden's: from the fitted scale to max(2, 4 × fitted). */
  private maxZoom(): number {
    const { scale } = this.frame
    return Math.max(2, scale * 4) / scale
  }

  private zoomBy(factor: number): void {
    const zoom = clamp(this.zoom * factor, 1, this.maxZoom())
    if (zoom === this.zoom) return
    const ratio = zoom / this.zoom
    this.zoom = zoom
    this.pan = [this.pan[0] * ratio, this.pan[1] * ratio]
    this.requestPlace()
  }

  private setControls(): void {
    this.panel.setCamera({
      canIn: this.zoom < this.maxZoom() - 0.001,
      canOut: this.zoom > 1.001,
      fullscreen: this.stage.fullscreen,
    })
  }

  private onFullscreen(): void {
    const full = this.stage.fullscreen
    if (full === this.fullscreen) return
    this.fullscreen = full
    this.root.toggleClass('is-fullscreen', full)
    if (full) this.observer?.observe(this.root)
    else this.observer?.unobserve(this.root)
    this.stage.reset()
    this.restoreFocus = true
    this.setControls()
    this.schedule()
  }

  /**
   * Writes the world transform: the scroll camera's `base`, zoomed about the anchor, then the pan.
   * Zoomed or panned, the case keeps at least 30% of the stage, as in the garden.
   */
  private place(): void {
    const [ax, ay] = this.stage.anchor()
    const z = this.zoom
    const x = ax + z * (this.base[0] - ax)
    const y = ay + z * (this.base[1] - ay)
    if (z !== 1 || this.pan[0] !== 0 || this.pan[1] !== 0) {
      const { stageWidth, sceneHeight } = this.stage.metrics
      const w = this.size[0] * z
      const h = this.size[1] * z
      const ix = Math.min(stageWidth, w) * 0.3
      const iy = Math.min(sceneHeight, h) * 0.3
      this.pan = [
        clamp(x + this.pan[0], ix - w, stageWidth - ix) - x,
        clamp(y + this.pan[1], iy - h, sceneHeight - iy) - y,
      ]
    }
    const offset: Point = [x + this.pan[0], y + this.pan[1]]
    this.offset = offset
    const move = `translate3d(${offset[0]}px, ${offset[1]}px, 0)`
    this.world.style.transform = z === 1 ? move : `${move} scale(${z})`
    this.root.toggleClass('is-zoomed', z !== 1)
  }

  private band(): [number, number] {
    return this.stage.band(this.offset, this.zoom)
  }

  /** Pan and zoom frames: a drag, a zoom step, or a released pan easing home. */
  private requestPlace(): void {
    this.placeRaf ??= requestAnimationFrame(now => {
      this.placeRaf = null
      if (this.releasing) {
        const k = 1 - Math.exp(-Math.min(64, Math.max(0, now - this.releaseAt)) / GLIDE_MS)
        this.releaseAt = now
        this.pan = [this.pan[0] * (1 - k), this.pan[1] * (1 - k)]
        if (Math.hypot(this.pan[0], this.pan[1]) < 0.5) {
          this.pan = [0, 0]
          this.releasing = false
        }
      }
      this.place()
      this.layers.check(this.band(), this.stage.span(this.offset, this.zoom))
      this.setControls()
      this.requestSettle()
      if (this.releasing) this.requestPlace()
    })
  }

  /** A new focus brings the camera back onto it: the pan eases to zero, the zoom stays. */
  private releasePan(): void {
    if (this.drag.dragging || this.releasing || (this.pan[0] === 0 && this.pan[1] === 0)) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      this.pan = [0, 0]
      return
    }
    this.releasing = true
    this.releaseAt = performance.now()
    this.requestPlace()
  }

  private render(index: number): string {
    const scene = this.scene
    const layer = scene?.layers[index]
    if (!scene || !layer) return ''
    return layer.kind === 'solids'
      ? solidsMarkup(layer)
      : shelfMarkup(layer, scene.books, book => this.paint(book.index))
  }

  /** Writes the hidden state of a lifted book into its shelf's markup, so a mount sets no attribute. */
  private prepare(index: number, markup: string): string {
    const layer = this.scene?.layers[index]
    if (layer?.kind !== 'shelf' || this.hidden.size === 0) return markup
    for (const i of layer.books)
      if (this.hidden.has(i))
        markup = markup.replace(`data-book="${i}" `, `data-book="${i}" visibility="hidden" `)
    return markup
  }

  /**
   * Signatures as SPEC 5.6, without the colours: a book whose colour alone changed is patched in
   * place, so a theme switch or a finished cover never rewrites a shelf.
   */
  private paintLayers(scene: Scene<BookModel>, previous: (BookPaint | undefined)[]): void {
    this.layers.reshape(scene.layers.map(layer => layer.kind))
    const { scale } = this.frame
    for (const [i, layer] of scene.layers.entries()) {
      if (layer.kind === 'solids') {
        const signature = `${scale}|${partsSignature(layer.halos, layer.solids)}`
        this.layers.update(i, signature, solidsBox(layer), this.frame)
        continue
      }
      const parts: unknown[] = [
        scale,
        layer.origin,
        layer.plate?.label,
        layer.plate?.face[0],
        layer.contact.points[0],
        partsSignature(layer.halos, layer.solids),
      ]
      for (const index of layer.books) {
        const book = scene.books[index]
        parts.push(
          index,
          book.path,
          book.label.title,
          book.label.author,
          spineLabels(book, scale),
          book.thickness,
          book.height,
          book.depth,
          book.faces.edge[0],
        )
      }
      if (this.layers.update(i, parts.join('|'), shelfBox(layer, scene.books), this.frame)) continue
      for (const index of layer.books) {
        // A paint never computed was never drawn.
        const old = previous[index]
        if (!old) continue
        const paint = this.paint(index)
        if (old.spine !== paint.spine || old.ink !== paint.ink) this.patchBook(i, index)
      }
    }
  }

  /** Writes a book's new colour into its mounted shelf; an unmounted shelf renders it on mount. */
  private patchBook(layer: number, index: number): void {
    this.layers.invalidate(layer)
    this.bookElement(index)?.setAttribute('style', bookStyle(this.paint(index)))
  }

  private paint(index: number): BookPaint {
    let paint = this.paints[index]
    if (!paint) {
      const book = this.scene!.books[index]
      paint = paintBook(
        book,
        this.plugin.covers.spineColour(book.item, this.dark),
        this.frame.scale,
      )
      this.paints[index] = paint
    }
    return paint
  }

  private centre(index: number): Point {
    return (this.centres[index] ??= toPx(this.frame, this.scene!.books[index].centre))
  }

  private reach(index: number): Point {
    let reach = this.reaches[index]
    if (reach === undefined) {
      const book = this.scene!.books[index]
      let left = Infinity
      let right = -Infinity
      for (const [x] of book.faces.edge) {
        left = Math.min(left, x)
        right = Math.max(right, x)
      }
      const px = (x: number) => (x + book.pull[0] - this.frame.minX) * this.frame.scale
      reach = [px(left), px(right)]
      this.reaches[index] = reach
    }
    return reach
  }

  /** The shelf copy of a book, when its shelf is mounted. */
  private bookElement(index: number): Element | null {
    const book = this.scene?.books[index]
    const layer = book ? this.shelfLayer.get(book.shelf) : undefined
    if (layer === undefined || !this.layers.isMounted(layer)) return null
    const children = this.layers.element(layer)?.children
    return children?.[children.length - this.fromEnd[index]] ?? null
  }

  /** Colours of books whose cover finished, and the lifted book's cover URL. */
  private applyColours(): void {
    const scene = this.scene
    if (!scene) return
    for (const index of this.recolour) {
      const book = scene.books[index]
      if (!book) continue
      const old = this.paints[index]
      this.paints[index] = undefined
      if (!old) continue
      const paint = this.paint(index)
      const layer = this.shelfLayer.get(book.shelf)
      if (layer !== undefined && (old.spine !== paint.spine || old.ink !== paint.ink))
        this.patchBook(layer, index)
    }
    const lifted = this.lift.index
    if (lifted !== null && this.recolour.has(lifted)) this.syncLift(lifted)
    this.recolour.clear()
  }

  private requestSettle(): void {
    if (this.settleTimer !== null || this.settleIdle !== null) return
    const wait = this.lastScroll + QUIET_MS - performance.now()
    if (wait > 0) {
      this.settleTimer = window.setTimeout(() => {
        this.settleTimer = null
        this.requestSettle()
      }, wait)
      return
    }
    this.settleIdle = idle(remaining => {
      this.settleIdle = null
      if (this.settle(remaining)) this.requestSettle()
    }, 1000)
  }

  private cancelSettle(): void {
    if (this.settleTimer !== null) window.clearTimeout(this.settleTimer)
    this.settleIdle?.()
    this.preloadIdle?.()
    this.settleTimer = this.settleIdle = this.preloadIdle = null
  }

  /** Idle pass: finished covers, then windowing and markup. Returns true when work is left. */
  private settle(remaining: () => number): boolean {
    if (!this.scene) return false
    // A scroll whose event has not fired yet counts as motion too.
    if (performance.now() - this.lastScroll < QUIET_MS || this.stage.moved()) return true
    if (this.indexPending) {
      this.indexPending = false
      this.updateIndex(this.scene)
    }
    if (this.recolour.size > 0) this.applyColours()
    return this.layers.settle(
      this.band(),
      this.stage.metrics.stageHeight,
      new Set(this.ahead()),
      remaining,
    )
  }

  private text(key: string): string | string[] {
    const scene = this.scene
    if (!scene) return ''
    const n = Number(key.slice(1))
    switch (key[0]) {
      case 't':
      case 'a': {
        const book = scene.books[n]
        if (!book) return ''
        this.fonts ??= spineFonts(this.root)
        const labels = fitSpine(book, this.fonts)
        return key[0] === 't' ? labels.title : labels.author
      }
      case 'p':
        for (const layer of scene.layers)
          if (layer.kind === 'shelf' && layer.shelf === n) return layer.plate?.label ?? ''
        return ''
      case 'c': {
        const book = scene.books[n]
        return book ? coverLines(book) : ''
      }
    }
    return ''
  }

  private setHidden(index: number, hidden: boolean): void {
    if (this.hidden.has(index) === hidden) return
    if (hidden) this.hidden.add(index)
    else this.hidden.delete(index)
    const g = this.bookElement(index)
    if (hidden) g?.setAttribute('visibility', 'hidden')
    else g?.removeAttribute('visibility')
  }

  private onScroll(evt: Event): void {
    if (evt.currentTarget !== this.stage.scroller || this.scrollRaf !== null) return
    this.lastScroll = performance.now()
    const user = this.lastScroll - this.userScroll < USER_SCROLL_MS
    this.scrollRaf = requestAnimationFrame(() => {
      this.scrollRaf = null
      this.camera(false, undefined, user)
    })
  }

  /**
   * Camera, focus and windowing for the current scroll position (SPEC 6.2). After a rebuild
   * (`sync`) the focused book is brought up to date even when the focus did not move.
   */
  private camera(sync = false, scrollTop?: number, user = false): void {
    const scene = this.scene
    const settings = this.settings
    if (!scene || !settings) return
    const count = scene.books.length
    const t = this.stage.progress(count, settings.scrollStep, scrollTop)
    // An empty case sits in the middle of the stage; otherwise the focused spine holds the anchor.
    const target = count > 0 ? cameraTarget(i => this.centre(i), count, t) : this.emptyTarget()
    let offset: Point
    if (count > 0) {
      const a = this.reach(Math.floor(t))
      const b = Math.ceil(t) < count ? this.reach(Math.ceil(t)) : a
      const k = t - Math.floor(t)
      const lift: Point = [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k]
      offset = this.stage.offset(target, lift, this.reserve)
    } else {
      const { stageWidth, sceneHeight } = this.stage.metrics
      offset = [stageWidth / 2 - target[0], sceneHeight / 2 - target[1]]
    }
    this.base = offset
    const focus = count > 0 ? Math.round(t) : -1
    if (focus !== this.focus) this.releasePan()
    this.place()
    if (focus !== this.focus) this.setFocus(focus)
    if (sync) this.syncSelection()
    // A user's scroll selects the book at the reading position, and slides it out again.
    else if (user && focus >= 0 && (focus !== this.selected || !this.lifted))
      this.select(focus, true)
    // A rebuild mounts only the shelves on the stage; the idle pass mounts the rest.
    const urgent = this.layers.check(
      this.band(),
      sync ? this.stage.span(this.offset, this.zoom) : undefined,
    )
    // One shelf mount per focus change: the ones the camera needs now, else one ahead of it.
    // Shelf markup is rendered in idle time, so a mount here only parses it.
    if (focus !== this.focusMounts) {
      this.focusMounts = focus
      if (urgent === 0 && !sync)
        this.layers.mountNext(this.ahead(), this.band(), this.stage.metrics.stageHeight)
    }
    this.requestSettle()
  }

  /** Layers of the books within LOOKAHEAD of the focus, nearest first, the heading first. */
  private ahead(): number[] {
    const books = this.scene?.books ?? []
    const order: number[] = []
    for (let d = 1; d <= LOOKAHEAD; d++)
      for (const i of [this.focus + this.heading * d, this.focus - this.heading * d]) {
        const layer = books[i] ? this.shelfLayer.get(books[i].shelf) : undefined
        if (layer !== undefined) order.push(layer)
      }
    return order
  }

  /** Middle of the case, for a view with no books. */
  private emptyTarget(): Point {
    const [minX, minY, maxX, maxY] = this.scene?.bounds ?? [0, 0, 0, 0]
    return toPx(this.frame, [(minX + maxX) / 2, (minY + maxY) / 2])
  }

  /** The reading position moved: the roving tabindex, the covers to decode next, the heading. */
  private setFocus(focus: number): void {
    if (this.focus >= 0 && focus >= 0) this.heading = focus < this.focus ? -1 : 1
    this.focus = focus
    const book = this.scene?.books[focus]
    if (book) this.focusPath = book.path
    this.rove(focus)
    if (!book) {
      this.preload.clear()
      return
    }
    this.preloadAround()
    this.plugin.covers.focus(this, focus)
  }

  /** After a rebuild: the reading position's tabindex, and the selection's lift and page. */
  private syncSelection(): void {
    const settings = this.settings
    if (!settings) return
    this.rove(this.focus)
    if (this.focus >= 0) this.preloadAround()
    const book = this.scene?.books[this.selected]
    this.markSelected(book ? this.selected : -1)
    if (book && this.lifted) this.syncLift(this.selected)
    if (!this.linked)
      this.page.schedule(book?.item ?? null, settings, this.dataChanged, this.summary())
  }

  /**
   * Selects a book: it lifts, and the page panel or the editor pane shows it. A selection the
   * scroll made (`scrolled`) does not split off the editor pane, and moves an open one only once
   * the scroll rests.
   */
  private select(index: number, scrolled = false): void {
    const book = this.scene?.books[index]
    const settings = this.settings
    if (!book || !settings) return
    const changed = index !== this.selected
    this.selected = index
    this.selectedPath = book.path
    this.markSelected(index)
    if (!this.lifted || this.lift.index !== index) {
      this.lift.show(this.liftBook(index), this.frame)
      // A book that slides out from under a still pointer was not hovered.
      this.near = false
      this.hovering = -1
      this.closeHover()
    }
    this.lifted = true
    if (!changed) return
    this.preloadAround()
    if (this.openTimer !== null) window.clearTimeout(this.openTimer)
    this.openTimer = null
    if (scrolled && this.linked) {
      this.openTimer = window.setTimeout(() => {
        this.openTimer = null
        if (this.selectedPath === book.path) this.openEditor(book.path)
      }, SCROLL_OPEN_MS)
      return
    }
    const inEditor = !scrolled && this.openEditor(book.path)
    if (!inEditor && !this.linked) this.page.schedule(book.item, settings, false, this.summary())
  }

  /** Clears the selection: the book goes back and the page shows the topic and its count. */
  private deselect(): void {
    if (this.openTimer !== null) window.clearTimeout(this.openTimer)
    this.openTimer = null
    if (this.selected < 0) {
      this.putBack()
      return
    }
    this.clearSelection()
    const settings = this.settings
    if (settings && !this.linked) this.page.schedule(null, settings, false, this.summary())
  }

  /** Forgets the selection; the caller updates the page. */
  private clearSelection(): void {
    this.selected = -1
    this.selectedPath = null
    this.putBack()
    this.markSelected(-1)
  }

  /** The selection follows its path through a rebuild; a book gone from the view clears it. */
  private resolveSelection(scene: Scene<BookModel>): void {
    if (this.selectedPath === null) return
    let best = -1
    for (const book of scene.books)
      if (
        book.path === this.selectedPath &&
        (best < 0 || Math.abs(book.index - this.selected) < Math.abs(best - this.selected))
      )
        best = book.index
    if (best < 0) this.clearSelection()
    else this.selected = best
  }

  /** The lifted book goes back to its shelf and its preview closes; the selection stays. */
  private putBack(): void {
    this.near = false
    this.hovering = -1
    this.closeHover()
    if (!this.lifted) return
    this.lifted = false
    this.lift.show(null, this.frame)
  }

  /** The page panel with nothing selected: the topic on show and its count. */
  private summary(): string {
    const count = this.scene?.books.length ?? 0
    const books = `${count} ${count === 1 ? 'book' : 'books'}`
    const topic = this.tabs.find(tab => tab.id === this.filter?.topic)
    return topic ? `${topic.label} · ${books}` : books
  }

  /** This view's own leaf in the main area or a popout; null in an embed or a sidebar. */
  private hostLeaf(): WorkspaceLeaf | null {
    if (this.containerEl.closest('.internal-embed, .markdown-embed, .bases-embed')) return null
    const { workspace } = this.app
    const hosts: WorkspaceLeaf[] = []
    workspace.iterateAllLeaves(leaf => {
      if (leaf.view.containerEl.contains(this.containerEl)) hosts.push(leaf)
    })
    const host = hosts[0]
    if (!host) return null
    const root = host.getRoot()
    return root === workspace.leftSplit || root === workspace.rightSplit ? null : host
  }

  /** Opens `path` in the editor pane, splitting this view's leaf the first time (wide layout). */
  private openEditor(path: string): boolean {
    if (!this.settings?.editorPane) return false
    if (this.linked) {
      if (this.editor.show(null, path)) return true
      this.setLinked(false)
      return false
    }
    const host = this.stage.metrics.mode === 'wide' ? this.hostLeaf() : null
    if (!host || !this.editor.show(host, path)) return false
    this.setLinked(true)
    return true
  }

  /** While the editor pane lives, the page panel is hidden and the stage takes the width. */
  private setLinked(on: boolean, rebuild = true): void {
    if (this.linked === on) return
    this.linked = on
    this.root.toggleClass('is-linked', on)
    const settings = this.settings
    if (!on && settings) {
      const book = this.scene?.books[this.selected]
      this.page.schedule(book?.item ?? null, settings, true, this.summary())
    }
    if (rebuild) this.schedule()
  }

  /** Closes the page preview the view opened, so none is left behind. */
  private closeHover(): void {
    const popover = this.hoverPopover
    if (!popover) return
    this.hoverPopover = null
    popover.hoverEl.detach()
    popover.unload()
  }

  private syncLift(index: number): void {
    const lifted = this.liftBook(index)
    if (!lifted) return
    if (this.lift.index === index) this.lift.update(lifted, this.frame)
    else this.lift.show(lifted, this.frame)
  }

  private liftBook(index: number): LiftRequest | null {
    const book = this.scene?.books[index]
    if (!book) return null
    const paint = this.paint(index)
    const image = this.plugin.covers.image(book.item, this.app)
    return {
      index,
      key: [this.version, index, paint.spine, paint.ink, image, this.frame.scale].join('|'),
      build: () => liftContent(book, paint, image, key => this.text(key)),
    }
  }

  /**
   * In idle time, decodes the thumbnails of focus ± 3 and writes the book a key selects next into
   * the spare lift.
   */
  private preloadAround(): void {
    this.preloadIdle ??= idle(() => {
      this.preloadIdle = null
      const books = this.scene?.books ?? []
      const focus = this.focus
      const next = this.selected >= 0 ? this.liftBook(this.selected + this.heading) : null
      if (next) this.lift.prepare(next, this.frame)
      const urls: string[] = []
      for (let d = 0; d <= PRELOAD_RADIUS; d++)
        for (const i of d === 0 ? [focus] : [focus - d, focus + d]) {
          const near = books[i]
          const url =
            near?.item.cover?.kind === 'vault' ? this.plugin.covers.thumb(near.item) : null
          if (url) urls.push(url)
        }
      this.preload.set(urls)
    }, 500)
  }

  /** Roving tabindex on the index: the button of the book at the reading position. */
  private rove(index: number): void {
    if (index === this.roving) return
    this.buttons[this.roving]?.setAttribute('tabindex', '-1')
    this.roving = index
    this.buttons[index]?.setAttribute('tabindex', '0')
  }

  /** aria-current on the index button of the selected book. */
  private markSelected(index: number): void {
    if (index === this.marked) return
    this.buttons[this.marked]?.removeAttribute('aria-current')
    const had = this.marked >= 0
    this.marked = index
    this.buttons[index]?.setAttribute('aria-current', 'true')
    // With nothing selected there is no page panel: the stage takes the width until a pick.
    if (had !== index >= 0) {
      this.root.toggleClass('has-selection', index >= 0)
      this.schedule()
    }
  }

  private updateIndex(scene: Scene<BookModel>): void {
    const books = scene.books
    while (this.buttons.length > books.length) this.buttons.pop()?.parentElement?.remove()
    while (this.buttons.length < books.length) {
      const i = this.buttons.length
      const button = this.indexEl
        .createEl('li')
        .createEl('button', {
          attr: {
            type: 'button',
            tabindex: i === this.roving ? '0' : '-1',
            'data-index': String(i),
          },
        })
      if (i === this.marked) button.setAttribute('aria-current', 'true')
      this.buttons.push(button)
    }
    for (const [i, book] of books.entries()) {
      const button = this.buttons[i]
      const label = [book.item.title, book.item.authors.join(', ')].filter(Boolean).join(', ')
      if (button.getAttribute('aria-label') !== label) {
        button.setAttribute('aria-label', label)
        button.textContent = book.item.title
      }
    }
    if (this.roving >= books.length) this.roving = -1
    if (this.marked >= books.length) this.markSelected(-1)
  }

  private buttonIndex(target: EventTarget | null): number | null {
    const button = (target as Element | null)?.closest<HTMLElement>('button[data-index]')
    const i = Number(button?.dataset.index)
    return button && Number.isInteger(i) ? i : null
  }

  private onKey(evt: KeyboardEvent): void {
    const scene = this.scene
    const i = this.buttonIndex(evt.target)
    if (!scene || i === null) return
    const books = scene.books
    const last = books.length - 1
    const shelfStart = (shelf: number) => books.findIndex(b => b.shelf === shelf)
    let next: number | null = null
    switch (evt.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        next = Math.min(last, i + 1)
        break
      case 'ArrowLeft':
      case 'ArrowUp':
        next = Math.max(0, i - 1)
        break
      case 'PageDown': {
        const start = shelfStart(books[i].shelf + 1)
        next = start >= 0 ? start : last
        break
      }
      case 'PageUp': {
        const start = shelfStart(books[i].shelf - 1)
        next = start >= 0 ? start : 0
        break
      }
      case 'Home':
        next = 0
        break
      case 'End':
        next = last
        break
      case 'Enter':
        evt.preventDefault()
        this.select(i)
        this.openNote(books[i].path, evt)
        return
      default:
        return
    }
    evt.preventDefault()
    this.select(next)
    this.moveTo(next, true)
  }

  private moveTo(index: number, focusButton: boolean): void {
    const settings = this.settings
    if (!settings) return
    if (focusButton) {
      this.rove(index)
      // The index sits at the top of the view; focusing it must not scroll the track back.
      this.buttons[index]?.focus({ preventScroll: true })
    }
    this.stage.scrollTo(index, settings.scrollStep)
  }

  private bookAt(target: EventTarget | null) {
    const g = (target as Element | null)?.closest<SVGGElement>('[data-book]')
    const index = Number(g?.dataset.book)
    return g && Number.isInteger(index) ? (this.scene?.books[index] ?? null) : null
  }

  /** A book selects it, the lifted book opens its note, the empty stage clears the selection. */
  private onStageClick(evt: MouseEvent): void {
    // A press that moves a few px off a book ends on another element; the click then goes to
    // their common ancestor. It is still a click on what was pressed.
    const pressed = this.pressed
    this.pressed = null
    const target =
      pressed && (evt.target as Element).contains(pressed) ? pressed : (evt.target as Element)
    if (target.closest('.bookcase-controls')) return
    const book = this.bookAt(target)
    if (!book) {
      this.deselect()
      return
    }
    if (target.closest('.bookcase-lift')) {
      this.openNote(book.path, evt)
      return
    }
    this.select(book.index)
    // Keys go on from the clicked book, and Escape and a blur reach the view.
    this.moveTo(book.index, true)
  }

  /** Opens the note: in the editor pane when there is one, else as Obsidian opens a link. */
  private openNote(path: string, evt: MouseEvent | KeyboardEvent): void {
    const mod = Keymap.isModEvent(evt)
    if (!mod && this.linked && this.editor.focus()) return
    void this.app.workspace.openLinkText(path, '', mod)
  }

  /**
   * A pointer moving onto the lifted book opens its page preview, once per lift. A pointer that
   * was over the lifted book or its slot and leaves both puts the book back and closes the
   * preview; the selection stays. A scroll that slides the book under a still pointer makes the
   * browser send moves at the same point; those do not count.
   */
  private onPointerMove(evt: PointerEvent): void {
    // Touch has no hover; a held button is a press or a drag.
    if (evt.pointerType === 'touch' || evt.buttons !== 0) return
    const [x, y] = this.pointerAt
    this.pointerAt = [evt.clientX, evt.clientY]
    if (evt.clientX === x && evt.clientY === y) return
    if (!this.lifted) return
    if (this.overLift(evt.clientX, evt.clientY)) this.near = true
    else if (this.near) {
      this.putBack()
      return
    }
    const el = (evt.target as Element).closest<HTMLElement>('.bookcase-lift')
    const book = el ? this.bookAt(evt.target) : null
    if (!el || !book || book.index !== this.selected || book.index === this.hovering) return
    this.hovering = book.index
    this.hover(evt, el, book.path)
  }

  /**
   * True when a client point is on the lifted book: pulled out, in the slot it left, or on the
   * way between them, so a pointer can follow the book out.
   */
  private overLift(x: number, y: number): boolean {
    const book = this.lifted ? this.scene?.books[this.selected] : undefined
    if (!book) return false
    if (this.swept?.index !== book.index || this.swept.version !== this.version)
      this.swept = {
        index: book.index,
        version: this.version,
        area: sweep(book.faces.edge, book.pull),
      }
    const box = this.world.getBoundingClientRect()
    const { minX, minY, scale } = this.frame
    const u = (x - box.left) / (this.zoom * scale) + minX
    const v = (y - box.top) / (this.zoom * scale) + minY
    return contains(this.swept.area, u, v)
  }

  private hover(event: MouseEvent, targetEl: HTMLElement, linktext: string): void {
    this.app.workspace.trigger('hover-link', {
      event,
      source: 'bases',
      hoverParent: this,
      targetEl,
      linktext,
    })
  }
}
