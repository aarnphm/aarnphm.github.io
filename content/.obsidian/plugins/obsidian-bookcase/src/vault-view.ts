// The film vault: the garden's rack drawing with its camera (native scroll, drag to pan, drag a can
// out or back in, zoom, fit)
// and pull, and the bookcase's page panel and editor pane in place of the garden's vault card.
import {
  type BasesEntry,
  type BasesPropertyId,
  BasesView,
  Component,
  type HoverPopover,
  Keymap,
  MarkdownRenderer,
  type QueryController,
  type TFile,
  type WorkspaceLeaf,
} from 'obsidian'
import type BookcasePlugin from '../main'
import { EditorPane } from './editor-pane'
import { ModelReader } from './model'
import { type VaultEntry, vaultMarkup } from './vault-model'
import { type VaultConfig, readVaultConfig } from './vault-options'

export const VAULT_VIEW = 'vault'

const SVG_NS = 'http://www.w3.org/2000/svg'
const ZOOM = 1.25
const MAX_SCALE = 3
/** 1000 ft rim titles are 9.5 units tall; this scale shows them at about 9 px. */
const READ_SCALE = 0.95
const DRAG_PX = 5
const PULL_ALIGN = Math.cos((24 * Math.PI) / 180)
const PULL_MS = 220
const GLIDE_MS = 90
const BAKE_MS = 120
const MAX_SETTLING = 1
const EASE = 'cubic-bezier(0.23, 1, 0.32, 1)'
/** The page panel renders once a run of arrow keys rests this long. */
const PAGE_MS = 140
const WIDE_PX = 600
const EMBED_HEIGHT = 520
const FULLSCREEN_ICON = 'M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7'
const EXIT_ICON = 'M4 14h6v6M3 21l7-7M20 10h-6V4M21 3l-7 7'

/** Fields the garden's record already shows; the view's `order` adds the rest. */
const CARD_KEYS = new Set([
  'title',
  'director',
  'year',
  'rating',
  'status',
  'studio',
  'watched',
  'runtime',
  'posters',
  'description',
])

/** Two lines of at most `max` characters for the round lid label, as the garden's. */
function lidLines(title: string, max = 12): string[] {
  const lines = ['']
  for (const word of title.toUpperCase().split(/\s+/)) {
    const line = lines[lines.length - 1]
    if (!line || line.length + word.length + 1 <= max)
      lines[lines.length - 1] = line ? `${line} ${word}` : word
    else if (lines.length < 2) lines.push(word)
    else {
      lines[1] = `${lines[1]}…`
      break
    }
  }
  return lines.map(line => (line.length > max + 1 ? `${line.slice(0, max - 1)}…` : line))
}

/** Embossed rings and the round lid label, laid on the pulled can's lid. */
function lidDetail(can: SVGGElement, title: string): SVGGElement | null {
  const lid = can.querySelector('.cn-lid')
  const transform = lid?.getAttribute('transform')
  const r = Number(lid?.getAttribute('r'))
  if (!transform || !Number.isFinite(r)) return null
  const node = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string>) => {
    const element = document.createElementNS(SVG_NS, tag)
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value)
    return element
  }
  const group = node('g', { class: 'cn-lid-detail', transform })
  const size = Math.round(r * 0.075 * 10) / 10
  const text = node('text', { class: 'cn-lid-text', 'font-size': String(size) })
  const lines = lidLines(title)
  lines.forEach((line, i) => {
    const span = node('tspan', {
      x: '0',
      y: String((i - (lines.length - 1) / 2) * size * 1.25 + size * 0.35),
    })
    span.textContent = line
    text.append(span)
  })
  group.append(
    node('circle', { class: 'cn-rings', r: String(r) }),
    node('circle', { class: 'cn-lid-label', r: String(Math.round(r * 0.28)) }),
    text,
  )
  return group
}

interface Film {
  entry: BasesEntry
  file: TFile
  title: string
  cover: string | null
}

/** One rebuild's elements and listeners; a rebuild or unload aborts them. */
interface Rack {
  abort: AbortController
  viewport: HTMLElement
  canvas: HTMLElement
  svg: SVGSVGElement
  lift: SVGGElement
  cans: Map<number, SVGGElement>
  rows: Map<number, HTMLElement>
  box: DOMRect
  start: [number, number]
}

export class VaultView extends BasesView {
  type = VAULT_VIEW
  hoverPopover: HoverPopover | null = null

  private root: HTMLElement
  private host: HTMLElement
  private pageEl: HTMLElement
  private liveEl: HTMLElement
  private reader: ModelReader
  private editor: EditorPane
  private settings: VaultConfig | null = null
  private films: Film[] = []
  private rack: Rack | null = null
  private rebuildRaf: number | null = null
  private observer: ResizeObserver | null = null
  private scroller: HTMLElement | null = null

  private scale = 1
  private bakedScale = 1
  private fitScale = 1
  private viewW = 0
  private viewH = 0
  private bakeTimer: number | null = null
  private aim: { left: number; top: number } | null = null
  private glideFrame = 0
  private glideAt = 0
  private writtenScroll = { left: -1, top: -1 }
  /** The first frame reads the top-left of the first bay; later rebuilds keep the camera. */
  private framed = false

  private current = -1
  private currentPath: string | null = null
  private pulled = new Map<SVGGElement, SVGGElement>()
  private settling = new Map<SVGGElement, { finish: () => void }>()
  private activePullAnim: Animation | null = null
  private pulledAt = 0
  private pullFrom = 0
  private pullDuration = PULL_MS
  /** Lifted can hover-link fired for, once per lift; -1 for none. */
  private hovering = -1
  private linked = false
  private pageTimer: number | null = null
  private pageToken = 0
  private pullToken = 0
  private pageChild: Component | null = null
  private fullscreen = false

  constructor(
    controller: QueryController,
    private containerEl: HTMLElement,
    private plugin: BookcasePlugin,
  ) {
    super(controller)
    this.reader = new ModelReader(this.app)
    this.editor = new EditorPane(this.app)
    this.root = containerEl.createDiv({ cls: 'vault-view', attr: { 'data-layout': 'wide' } })
    this.host = this.root.createDiv({ cls: 'vault-host' })
    this.pageEl = this.root.createEl('aside', { cls: 'vault-page' })
    this.liveEl = this.root.createDiv({ cls: 'vault-live', attr: { 'aria-live': 'polite' } })
  }

  onload(): void {
    const observer = new ResizeObserver(() => this.fitHeight())
    observer.observe(this.containerEl)
    this.observer = observer
    this.register(() => observer.disconnect())
    this.registerEvent(
      this.app.workspace.on('layout-change', () => {
        if (this.linked && !this.editor.alive()) this.setLinked(false)
      }),
    )
    this.registerDomEvent(document, 'fullscreenchange', () => {
      const full = document.fullscreenElement === this.root
      if (full === this.fullscreen) return
      this.fullscreen = full
      this.root.toggleClass('is-fullscreen', full)
      this.updateFullscreenButton()
      this.fitHeight()
    })
    // Escape returns the can while focus is in the view, or the pointer is over it.
    const doc = this.root.ownerDocument
    this.registerDomEvent(doc, 'keydown', evt => {
      if (evt.key !== 'Escape' || evt.defaultPrevented || this.current < 0) return
      if (!this.root.contains(doc.activeElement) && !this.root.matches(':hover')) return
      evt.preventDefault()
      this.release()
    })
    this.registerDomEvent(this.pageEl, 'click', evt => {
      const link = (evt.target as Element).closest<HTMLElement>('a.internal-link')
      const href = link?.dataset.href
      if (!href) return
      evt.preventDefault()
      const source = this.films[this.current]?.file.path ?? ''
      void this.app.workspace.openLinkText(href, source, Keymap.isModEvent(evt))
    })
    this.registerDomEvent(this.pageEl, 'mouseover', evt => {
      const link = (evt.target as Element).closest<HTMLElement>('a.internal-link')
      const href = link?.dataset.href
      if (link && href) this.hover(evt, link, href)
    })
    this.schedule()
  }

  onunload(): void {
    if (this.rebuildRaf !== null) cancelAnimationFrame(this.rebuildRaf)
    if (this.pageTimer !== null) window.clearTimeout(this.pageTimer)
    if (this.bakeTimer !== null) window.clearTimeout(this.bakeTimer)
    this.cancelGlide()
    this.activePullAnim?.cancel()
    this.activePullAnim = null
    for (const s of this.settling.values()) s.finish()
    this.settling.clear()
    this.rebuildRaf = this.pageTimer = this.bakeTimer = null
    this.closeHover()
    // The pane may hold unsaved edits: the view lets go of it and never closes it.
    this.editor.forget()
    if (this.fullscreen) void document.exitFullscreen().catch(() => undefined)
    this.rack?.abort.abort()
    this.rack = null
    this.dropPage()
    this.containerEl.empty()
  }

  onDataUpdated(): void {
    this.schedule()
  }

  private schedule(): void {
    if (this.rebuildRaf !== null) return
    this.rebuildRaf = requestAnimationFrame(() => {
      this.rebuildRaf = null
      this.rebuild()
    })
  }

  // ---- data and markup

  private rebuild(): void {
    const settings = readVaultConfig(this.config, this.config.name)
    this.settings = settings
    if (this.linked && !settings.editorPane) {
      this.editor.forget()
      this.setLinked(false)
    }
    const groups = this.data.groupedData
    const grouped = groups.some(group => group.hasKey())
    const films: Film[] = []
    const entries: VaultEntry[] = []
    for (const group of groups) {
      const label = grouped && group.hasKey() && group.key ? group.key.toString() : null
      for (const entry of group.entries) {
        const film = this.film(entry, settings)
        films.push(film)
        entries.push(this.entry(film, settings, grouped ? (label ?? 'none') : null))
      }
    }
    this.films = films
    const previous = this.rack
    const keep = previous && this.framed
    const left = previous?.viewport.scrollLeft ?? 0
    const top = previous?.viewport.scrollTop ?? 0
    this.cancelGlide()
    if (this.bakeTimer !== null) {
      window.clearTimeout(this.bakeTimer)
      this.bakeTimer = null
    }
    this.activePullAnim?.cancel()
    this.activePullAnim = null
    for (const s of this.settling.values()) s.finish()
    this.settling.clear()
    previous?.abort.abort()
    for (const clone of this.pulled.values()) clone.remove()
    this.pulled.clear()
    this.current = -1
    this.hovering = -1

    this.root.toggleClass('is-empty', films.length === 0)
    if (films.length === 0) {
      this.rack = null
      this.host.empty()
      this.host.createDiv({ cls: 'vault-empty', text: 'No films match this view.' })
      this.showPage(-1)
      return
    }
    this.host.innerHTML = vaultMarkup(entries, settings.view, `${this.plugin.grainAssets}vault/`)
    this.rack = this.wire()
    if (!this.rack) return
    this.fitHeight()
    this.measure()
    if (keep) {
      this.setScale(Math.max(this.fitScale, this.scale), this.viewW / 2, this.viewH / 2, true)
      this.writeScroll(left, top)
    } else this.frameStart()
    // A rebuild keeps the pick when its note is still in the view.
    const again = this.currentPath ? films.findIndex(f => f.file.path === this.currentPath) : -1
    // An open editor pane keeps its note: a rebuild follows every edit made in it.
    if (again >= 0) {
      this.select(again, false, false, false)
      if (!this.linked) this.showPage(again)
    } else this.release()
  }

  private film(entry: BasesEntry, settings: VaultConfig): Film {
    const file = entry.file
    const title = this.text(entry, settings.props.labelTitle) || file.basename
    const cover = this.reader.cover(this.reader.read(entry, settings.props.image), file.path)
    return {
      entry,
      file,
      title,
      cover:
        cover?.kind === 'vault'
          ? this.app.vault.getResourcePath(cover.file)
          : cover?.kind === 'url'
            ? cover.url
            : null,
    }
  }

  private entry(film: Film, settings: VaultConfig, group: string | null): VaultEntry {
    const { entry, file } = film
    const { props } = settings
    const num = (id: BasesPropertyId) => {
      const value = Number(this.text(entry, id))
      return this.text(entry, id) && Number.isFinite(value) ? value : undefined
    }
    const description = this.text(entry, 'note.description' as BasesPropertyId)
    return {
      slug: file.path,
      path: file.path,
      href: file.path,
      title: film.title,
      directors: this.reader.strings(entry, props.labelDirector),
      year: num(props.labelYear),
      rating: num(props.labelRating),
      status: this.text(entry, props.labelStatus) || undefined,
      studio: this.reader.strings(entry, props.labelStudio)[0],
      watched: this.text(entry, props.labelWatched) || undefined,
      runtime: num(props.labelRuntime),
      description: description || undefined,
      cover: film.cover ?? undefined,
      group,
      rows: [],
    }
  }

  private text(entry: BasesEntry, id: BasesPropertyId): string {
    return this.reader.read(entry, id)?.toString().trim() ?? ''
  }

  // ---- layout and camera

  /**
   * In a pane, the view fills what the pane shows below its toolbar; the rack scrolls inside it.
   * An embed gets a fixed height, full screen the whole screen.
   */
  private fitHeight(): void {
    this.root.dataset.layout = this.root.clientWidth >= WIDE_PX ? 'wide' : 'narrow'
    if (this.fullscreen) {
      this.root.style.removeProperty('height')
      return
    }
    const embedded = this.containerEl.closest('.internal-embed, .markdown-embed, .bases-embed')
    let height = EMBED_HEIGHT
    if (!embedded) {
      if (!this.scroller?.isConnected) {
        this.scroller = null
        for (
          let el: HTMLElement | null = this.containerEl;
          el && el !== document.body;
          el = el.parentElement
        ) {
          const overflow = getComputedStyle(el).overflowY
          if ((overflow === 'auto' || overflow === 'scroll') && el.clientHeight > 0) {
            this.scroller = el
            this.observer?.observe(el)
            break
          }
        }
      }
      const scroller = this.scroller
      if (scroller) {
        const above =
          this.root.getBoundingClientRect().top -
          scroller.getBoundingClientRect().top +
          scroller.scrollTop
        height = Math.max(EMBED_HEIGHT * 0.6, scroller.clientHeight - above - 8)
      }
    }
    this.root.style.height = `${Math.round(height)}px`
  }

  private measure(): void {
    const rack = this.rack
    if (!rack) return
    this.viewW = rack.viewport.clientWidth
    this.viewH = rack.viewport.clientHeight
    const { width: W, height: H } = rack.box
    this.fitScale = Math.min(1, (this.viewW - 32) / W, (this.viewH - 32) / H)
  }

  private pad(k = this.scale): [number, number] {
    const { width: W, height: H } = this.rack!.box
    return [
      Math.round(this.viewW - Math.min(this.viewW, W * k) * 0.3),
      Math.round(this.viewH - Math.min(this.viewH, H * k) * 0.3),
    ]
  }

  private writeScroll(left: number, top: number): void {
    const rack = this.rack
    if (!rack) return
    const l = Math.max(0, Math.round(left))
    const t = Math.max(0, Math.round(top))
    this.writtenScroll = { left: l, top: t }
    rack.viewport.scrollTo({ left: l, top: t, behavior: 'instant' })
  }

  private cancelGlide(): void {
    this.aim = null
    if (this.glideFrame) {
      cancelAnimationFrame(this.glideFrame)
      this.glideFrame = 0
    }
  }

  private layoutCanvas(k = this.scale): void {
    const rack = this.rack
    if (!rack) return
    const { canvas, svg, box } = rack
    const [px, py] = this.pad(k)
    canvas.style.width = `${Math.round(box.width * k + px * 2)}px`
    canvas.style.height = `${Math.round(box.height * k + py * 2)}px`
    svg.style.left = `${px}px`
    svg.style.top = `${py}px`
  }

  private applySceneScale(): void {
    const svg = this.rack?.svg
    if (!svg) return
    const ratio = this.bakedScale > 0 ? this.scale / this.bakedScale : 1
    if (Math.abs(ratio - 1) < 0.001) {
      svg.style.transform = ''
      svg.style.willChange = 'auto'
    } else {
      svg.style.transform = `scale(${ratio})`
      svg.style.willChange = 'transform'
    }
  }

  private bakeScale(nextScale = this.scale): void {
    if (this.bakeTimer !== null) {
      window.clearTimeout(this.bakeTimer)
      this.bakeTimer = null
    }
    const rack = this.rack
    if (!rack) return
    this.bakedScale = nextScale
    rack.svg.setAttribute('width', String(Math.round(rack.box.width * this.bakedScale)))
    rack.svg.setAttribute('height', String(Math.round(rack.box.height * this.bakedScale)))
    this.layoutCanvas(this.bakedScale)
    this.applySceneScale()
  }

  /** Zooms about a viewport point, the centre by default, keeping it in place. */
  private setScale(
    value: number,
    ax = this.viewW / 2,
    ay = this.viewH / 2,
    immediate = false,
  ): void {
    const rack = this.rack
    if (!rack) return
    this.cancelGlide()
    const { viewport } = rack
    const prevScale = this.scale
    const [ox, oy] = this.pad(prevScale)
    const wx = (viewport.scrollLeft + ax - ox) / prevScale
    const wy = (viewport.scrollTop + ay - oy) / prevScale
    this.scale = Math.min(MAX_SCALE, Math.max(this.fitScale, value))
    const [nx, ny] = this.pad(this.scale)
    if (immediate) {
      this.bakeScale(this.scale)
    } else {
      this.layoutCanvas(this.scale)
      this.applySceneScale()
      if (this.bakeTimer !== null) window.clearTimeout(this.bakeTimer)
      this.bakeTimer = window.setTimeout(() => this.bakeScale(this.scale), BAKE_MS)
    }
    this.writeScroll(wx * this.scale + nx - ax, wy * this.scale + ny - ay)
    const button = (action: string) =>
      this.root.querySelector<HTMLButtonElement>(`[data-camera="${action}"]`)
    const zoomOut = button('out')
    const zoomIn = button('in')
    if (zoomOut) zoomOut.disabled = this.scale <= this.fitScale * 1.001
    if (zoomIn) zoomIn.disabled = this.scale >= MAX_SCALE * 0.999
  }

  private frameStart(): void {
    const rack = this.rack
    if (!rack) return
    const { box, start } = rack
    this.setScale(Math.max(this.fitScale, READ_SCALE), this.viewW / 2, this.viewH / 2, true)
    const [ox, oy] = this.pad(this.scale)
    this.writeScroll(
      ox + (start[0] - box.x) * this.scale - 24,
      oy + (start[1] - box.y) * this.scale - 48,
    )
    this.framed = true
  }

  private fit(): void {
    const rack = this.rack
    if (!rack) return
    this.setScale(this.fitScale, this.viewW / 2, this.viewH / 2, true)
    const [ox, oy] = this.pad(this.fitScale)
    this.writeScroll(
      ox + (rack.box.width * this.fitScale - this.viewW) / 2,
      oy + (rack.box.height * this.fitScale - this.viewH) / 2,
    )
  }

  private stepGlide(now = performance.now()): void {
    this.glideFrame = 0
    const rack = this.rack
    const aim = this.aim
    if (!rack || !aim) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const dt = now - this.glideAt
    this.glideAt = now
    const curLeft = rack.viewport.scrollLeft
    const curTop = rack.viewport.scrollTop
    const dx = aim.left - curLeft
    const dy = aim.top - curTop
    if (reduced || dt <= 0 || Math.hypot(dx, dy) < 1) {
      this.aim = null
      this.writeScroll(aim.left, aim.top)
      return
    }
    const k = 1 - Math.exp(-Math.min(64, Math.max(8, dt)) / GLIDE_MS)
    this.writeScroll(curLeft + dx * k, curTop + dy * k)
    if (Math.hypot(aim.left - rack.viewport.scrollLeft, aim.top - rack.viewport.scrollTop) < 1) {
      this.aim = null
      this.writeScroll(aim.left, aim.top)
      return
    }
    this.glideFrame = requestAnimationFrame(nextNow => this.stepGlide(nextNow))
  }

  private lookAt(x: number, y: number): void {
    const rack = this.rack
    if (!rack) return
    const [ox, oy] = this.pad(this.scale)
    const targetLeft = Math.max(0, Math.round((x - rack.box.x) * this.scale + ox - this.viewW / 2))
    const targetTop = Math.max(0, Math.round((y - rack.box.y) * this.scale + oy - this.viewH / 2))
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduced) {
      this.cancelGlide()
      this.writeScroll(targetLeft, targetTop)
      return
    }
    const firstStep = !this.aim
    this.aim = { left: targetLeft, top: targetTop }
    if (firstStep) {
      this.glideAt = performance.now()
      const curLeft = rack.viewport.scrollLeft
      const curTop = rack.viewport.scrollTop
      const k = 1 - Math.exp(-16 / GLIDE_MS)
      this.writeScroll(curLeft + (targetLeft - curLeft) * k, curTop + (targetTop - curTop) * k)
    }
    if (!this.glideFrame && this.aim) {
      this.glideFrame = requestAnimationFrame(nextNow => this.stepGlide(nextNow))
    }
  }

  // ---- selection

  /** The pick: the can slides out with its lid label, and the page panel or editor follows it. */
  private select(index: number, camera = true, open = false, follow = true): void {
    const rack = this.rack
    const can = rack?.cans.get(index)
    if (!rack || !can) return
    const [px, py] = this.pullOf(can)
    if (index !== this.current) {
      this.pullToken++
      this.putBack()
      this.current = index
      this.currentPath = this.films[index].file.path
      this.hovering = -1
      const clone = this.mountClone(index)
      if (clone) {
        clone.dataset.out = ''
        clone.dataset.progress = '1'
        this.activePullAnim?.cancel()
        this.activePullAnim = null
        this.pulledAt = performance.now()
        this.pullFrom = 0
        this.pullDuration = PULL_MS
        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
        if (!reduced && typeof clone.animate === 'function') {
          const anim = clone.animate(
            [{ transform: 'translate(0px, 0px)' }, { transform: `translate(${px}px, ${py}px)` }],
            { duration: PULL_MS, easing: EASE },
          )
          this.activePullAnim = anim
          anim.onfinish = () => {
            if (this.activePullAnim === anim) this.activePullAnim = null
          }
        }
      }
      const row = rack.rows.get(index)
      row?.querySelector('.vault-entry-select')?.setAttribute('aria-current', 'true')
      this.corners(index)
      this.liveEl.setText(`${this.films[index].title}, ${index + 1} of ${this.films.length}`)
    }
    if (camera) {
      this.lookAt(Number(can.dataset.cx) + px * 0.6, Number(can.dataset.cy) + py * 0.6)
    }
    this.root.toggleClass('has-selection', true)
    if (!follow) return
    if (open && this.openEditor(this.films[index].file.path)) return
    if (this.linked) this.editor.show(null, this.films[index].file.path)
    else this.showPage(index)
  }

  private sendHome(can: SVGGElement, clone: SVGGElement, from: number): void {
    this.pulled.delete(can)
    delete clone.dataset.out
    delete clone.dataset.pulling
    delete clone.dataset.progress
    clone.style.transition = ''
    clone.style.transform = ''
    clone.style.pointerEvents = 'none'
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduced || from < 0.08 || typeof clone.animate !== 'function') {
      clone.remove()
      if (!this.pulled.has(can)) delete can.dataset.lifted
      return
    }
    if (this.settling.size >= MAX_SETTLING) this.settling.values().next().value?.finish()
    const [px, py] = this.pullOf(can)
    const duration = Math.max(1, Math.round(PULL_MS * from))
    let done = false
    let timer: number | undefined
    const anim = clone.animate(
      [
        { transform: `translate(${px * from}px, ${py * from}px)` },
        { transform: 'translate(0px, 0px)' },
      ],
      { duration, easing: EASE, fill: 'both' },
    )
    const finish = () => {
      if (done) return
      done = true
      if (timer !== undefined) window.clearTimeout(timer)
      anim.onfinish = null
      anim.cancel()
      this.settling.delete(can)
      clone.remove()
      if (!this.pulled.has(can)) delete can.dataset.lifted
    }
    anim.onfinish = finish
    timer = window.setTimeout(finish, duration + 20)
    this.settling.set(can, { finish })
  }

  /** Returns the pulled can to its slot. */
  private putBack(from?: number): void {
    const rack = this.rack
    if (!rack || this.current < 0) return
    const can = rack.cans.get(this.current)
    rack.rows
      .get(this.current)
      ?.querySelector('.vault-entry-select')
      ?.removeAttribute('aria-current')
    this.closeHover()
    const clone = can && this.pulled.get(can)
    if (!can || !clone) {
      this.activePullAnim?.cancel()
      this.activePullAnim = null
      return
    }
    const [px, py] = this.pullOf(can)
    const start = from ?? this.pullProgress(clone, px, py)
    this.activePullAnim?.cancel()
    this.activePullAnim = null
    this.sendHome(can, clone, start)
  }

  /** Clears the pick: the can goes back, the page panel closes, an editor pane keeps its note. */
  private release(from?: number): void {
    this.pullToken++
    this.putBack(from)
    this.current = -1
    this.currentPath = null
    this.corners(-1)
    this.root.toggleClass('has-selection', false)
    this.showPage(-1)
  }

  private pullOf(can: SVGGElement): [number, number] {
    return [
      parseFloat(can.style.getPropertyValue('--px')) || 0,
      parseFloat(can.style.getPropertyValue('--py')) || 0,
    ]
  }

  private pullProgress(clone: SVGGElement, px: number, py: number): number {
    if (clone.dataset.pulling !== undefined && clone.dataset.progress !== undefined) {
      const p = Number(clone.dataset.progress)
      if (Number.isFinite(p)) return Math.min(1, Math.max(0, p))
    }
    if (clone.dataset.pulling === undefined && this.activePullAnim) {
      const elapsed = performance.now() - this.pulledAt
      if (elapsed < 16) return this.pullFrom
      if (elapsed >= this.pullDuration) return 1
      const u = 1 - elapsed / this.pullDuration
      return this.pullFrom + (1 - this.pullFrom) * (1 - u * u * u)
    }
    if (clone.dataset.progress !== undefined) {
      const p = Number(clone.dataset.progress)
      if (Number.isFinite(p)) return Math.min(1, Math.max(0, p))
    }
    const len2 = px * px + py * py
    const resting = clone.dataset.out !== undefined ? 1 : 0
    if (len2 < 1) return resting
    const raw = getComputedStyle(clone).transform
    if (!raw || raw === 'none') return resting
    const matrix = new DOMMatrix(raw)
    if (!Number.isFinite(matrix.e) || !Number.isFinite(matrix.f)) return resting
    return Math.min(1, Math.max(0, (matrix.e * px + matrix.f * py) / len2))
  }

  private sceneScale(): number {
    return this.scale
  }

  private mountClone(index: number): SVGGElement | undefined {
    const rack = this.rack
    const can = rack?.cans.get(index)
    const film = this.films[index]
    if (!rack || !can || !film) return
    const existing = this.pulled.get(can)
    if (existing) return existing
    this.settling.get(can)?.finish()
    const clone = can.cloneNode(true) as SVGGElement
    clone.querySelector('.cn-rings')?.remove()
    const detail = lidDetail(clone, film.title)
    if (detail) clone.append(detail)
    can.dataset.lifted = ''
    this.pulled.set(can, clone)
    rack.lift.append(clone)
    return clone
  }

  /** Snaps a dragged can out into the pick, or back into its slot. */
  private finishPull(index: number, t: number): void {
    const can = this.rack?.cans.get(index)
    const clone = can && this.pulled.get(can)
    if (!can || !clone) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const [px, py] = this.pullOf(can)
    const out = t >= 0.5
    this.pullToken++
    this.activePullAnim?.cancel()
    this.activePullAnim = null
    delete clone.dataset.pulling
    clone.style.transition = ''
    clone.style.transform = ''
    if (!out) {
      if (this.current === index) this.release(t)
      else this.sendHome(can, clone, t)
      return
    }
    clone.dataset.out = ''
    clone.dataset.progress = '1'
    if (this.current !== index) {
      this.select(index, false, false, true)
    }
    if (reduced || t >= 0.999 || typeof clone.animate !== 'function') return
    const duration = Math.max(1, Math.round(PULL_MS * (1 - t)))
    this.pulledAt = performance.now()
    this.pullFrom = t
    this.pullDuration = duration
    const anim = clone.animate(
      [
        { transform: `translate(${px * t}px, ${py * t}px)` },
        { transform: `translate(${px}px, ${py}px)` },
      ],
      { duration, easing: EASE },
    )
    this.activePullAnim = anim
    anim.onfinish = () => {
      if (this.activePullAnim === anim) this.activePullAnim = null
    }
  }

  private step(delta: number): void {
    const total = this.films.length
    const target = this.current < 0 ? (delta > 0 ? 0 : total - 1) : this.current + delta
    if (target >= 0 && target < total) this.select(target, true, true)
  }

  private corners(index: number): void {
    const total = this.films.length
    const readout = this.root.querySelector('[data-vault-readout]')
    const section = this.root.querySelector('[data-vault-section]')
    if (readout) readout.textContent = index < 0 ? `– / ${total}` : `${index + 1} / ${total}`
    if (section)
      section.textContent =
        index < 0
          ? ''
          : (this.rack?.rows.get(index)?.closest('details')?.querySelector('.vault-index-label')
              ?.textContent ?? '')
  }

  private openNote(index: number, evt: MouseEvent | KeyboardEvent): void {
    const mod = Keymap.isModEvent(evt)
    if (!mod && this.linked && this.editor.focus()) return
    void this.app.workspace.openLinkText(this.films[index].file.path, '', mod)
  }

  // ---- page panel and editor pane

  private showPage(index: number): void {
    if (this.pageTimer !== null) window.clearTimeout(this.pageTimer)
    this.pageTimer = window.setTimeout(
      () => {
        this.pageTimer = null
        void this.renderPage(index, ++this.pageToken)
      },
      index < 0 ? 0 : PAGE_MS,
    )
  }

  private dropPage(): void {
    if (this.pageChild) this.removeChild(this.pageChild)
    this.pageChild = null
    this.pageEl.empty()
  }

  private async renderPage(index: number, token: number): Promise<void> {
    this.dropPage()
    const film = this.films[index]
    const settings = this.settings
    if (!film || !settings || index !== this.current) return
    const el = this.pageEl
    if (film.cover)
      el.createEl('figure', { cls: 'vault-page-poster' }).createEl('img', {
        attr: { src: film.cover, alt: '', decoding: 'async' },
      })
    el.createEl('h2', { cls: 'vault-page-title' }).createEl('a', {
      cls: 'internal-link',
      text: film.title,
      attr: { 'data-href': film.file.path, href: film.file.path },
    })
    // The garden's record of the can: label line, its rows and the description.
    const record = this.rack?.rows.get(index)?.querySelector('.vault-entry-detail')
    const detail = el.createDiv({ cls: 'vault-page-record' })
    for (const child of Array.from(record?.childNodes ?? [])) detail.append(child.cloneNode(true))
    const list = detail.querySelector('.vault-props')
    if (list) {
      // The view's other `order` properties go after director and studio, as the garden's rows.
      const anchor = list.children[2] ?? null
      const skip = new Set<string>(Object.values(settings.props))
      for (const id of this.config.getOrder()) {
        const name = id.replace(/^(note|file|formula)\./, '')
        if (skip.has(id) || CARD_KEYS.has(name)) continue
        const value = this.reader.read(film.entry, id)
        if (!value) continue
        const row = createDiv({ cls: 'vault-prop' })
        row.createEl('dt', { text: this.config.getDisplayName(id) })
        // The view implements hoverPopover, which is all a RenderContext carries.
        value.renderTo(row.createEl('dd'), this)
        list.insertBefore(row, anchor)
      }
    }
    if (!settings.pageContent) return
    const text = await this.app.vault.cachedRead(film.file)
    if (token !== this.pageToken) return
    const end = this.app.metadataCache.getFileCache(film.file)?.frontmatterPosition?.end.offset ?? 0
    const body = text.slice(end).trim()
    if (!body) return
    const child = this.addChild(new Component())
    this.pageChild = child
    const bodyEl = el.createDiv({ cls: 'vault-page-body markdown-rendered' })
    await MarkdownRenderer.render(this.app, body, bodyEl, film.file.path, child)
  }

  private hostLeaf(): WorkspaceLeaf | null {
    const { workspace } = this.app
    let host: WorkspaceLeaf | null = null
    workspace.iterateAllLeaves(leaf => {
      if (!host && leaf.view.containerEl.contains(this.containerEl)) host = leaf
    })
    if (!host) return null
    const root = (host as WorkspaceLeaf).getRoot()
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
    const host = this.root.dataset.layout === 'wide' && !this.fullscreen ? this.hostLeaf() : null
    if (!host || !this.editor.show(host, path)) return false
    this.setLinked(true)
    return true
  }

  /** While the editor pane lives, the page panel is hidden and the stage takes the width. */
  private setLinked(on: boolean): void {
    if (this.linked === on) return
    this.linked = on
    this.root.toggleClass('is-linked', on)
    if (!on) this.showPage(this.current)
    requestAnimationFrame(() => this.measure())
  }

  private closeHover(): void {
    const popover = this.hoverPopover
    if (!popover) return
    this.hoverPopover = null
    popover.hoverEl.detach()
    popover.unload()
  }

  private hover(event: MouseEvent, targetEl: HTMLElement | SVGElement, linktext: string): void {
    this.app.workspace.trigger('hover-link', {
      event,
      source: 'bases',
      hoverParent: this,
      targetEl,
      linktext,
    })
  }

  // ---- events

  private updateFullscreenButton(): void {
    const button = this.root.querySelector<HTMLButtonElement>('[data-camera="fullscreen"]')
    if (!button) return
    const label = this.fullscreen ? 'Exit full screen' : 'Full screen'
    button.setAttribute('aria-pressed', String(this.fullscreen))
    button.setAttribute('aria-label', label)
    button.setAttribute('title', label)
    button.querySelector('path')?.setAttribute('d', this.fullscreen ? EXIT_ICON : FULLSCREEN_ICON)
  }

  private wire(): Rack | null {
    const viewport = this.host.querySelector<HTMLElement>('.vault-viewport')
    const svg = this.host.querySelector<SVGSVGElement>('svg.vault-scene')
    const lift = svg?.querySelector<SVGGElement>('.vault-lift')
    const controls = this.host.querySelector<HTMLElement>('.vault-controls')
    if (!viewport || !svg || !lift || !controls) return null
    const abort = new AbortController()
    const { signal } = abort
    const cans = new Map<number, SVGGElement>()
    for (const can of Array.from(svg.querySelectorAll<SVGGElement>(':scope > .cn-can')))
      cans.set(Number(can.dataset.index), can)
    const rows = new Map<number, HTMLElement>()
    for (const row of Array.from(this.host.querySelectorAll<HTMLElement>('.vault-entry')))
      rows.set(Number(row.dataset.index), row)
    const vb = svg.viewBox.baseVal
    const [sx, sy] = (svg.dataset.start ?? '0 0').split(' ').map(Number)
    // The index stays for screen readers; the arrow keys on the rack walk it for everyone else.
    for (const button of Array.from(this.host.querySelectorAll<HTMLElement>('.vault-index button')))
      button.tabIndex = -1

    const full = controls.createEl('button', {
      attr: { type: 'button', 'data-camera': 'fullscreen', 'aria-pressed': 'false' },
    })
    full
      .createSvg('svg', { attr: { viewBox: '0 0 24 24', 'aria-hidden': 'true' } })
      .createSvg('path', { attr: { d: FULLSCREEN_ICON } })
    const canvas = document.createElement('div')
    canvas.className = 'vault-canvas'
    svg.remove()
    canvas.append(svg)
    viewport.append(canvas)
    const rack: Rack = {
      abort,
      viewport,
      canvas,
      svg,
      lift,
      cans,
      rows,
      box: new DOMRect(vb.x, vb.y, vb.width, vb.height),
      start: [sx, sy],
    }
    this.rack = rack
    this.updateFullscreenButton()

    type PendingGesture = {
      mode: 'pending'
      id: number
      x: number
      y: number
      left: number
      top: number
      index: number
      px: number
      py: number
    }
    type PanGesture = {
      mode: 'pan'
      id: number
      x: number
      y: number
      left: number
      top: number
      moved: boolean
    }
    type PullGesture = {
      mode: 'pull'
      id: number
      x: number
      y: number
      index: number
      px: number
      py: number
      start: number
      t: number
    }
    let gesture: PendingGesture | PanGesture | PullGesture | undefined
    let dragged = false
    let draggedResetTimer: number | null = null
    const clearDragged = () => {
      dragged = false
      if (draggedResetTimer !== null) {
        window.clearTimeout(draggedResetTimer)
        draggedResetTimer = null
      }
    }
    const markDragged = () => {
      dragged = true
      if (draggedResetTimer !== null) window.clearTimeout(draggedResetTimer)
      draggedResetTimer = window.setTimeout(() => {
        draggedResetTimer = null
        dragged = false
      }, 0)
    }
    const capturePointer = (pointerId: number) => {
      try {
        viewport.setPointerCapture(pointerId)
      } catch {}
    }
    const releasePointer = (pointerId: number) => {
      try {
        if (viewport.hasPointerCapture(pointerId)) viewport.releasePointerCapture(pointerId)
      } catch {}
    }
    signal.addEventListener('abort', clearDragged, { once: true })
    const placePull = (index: number, t: number) => {
      const can = rack.cans.get(index)
      const clone = can && this.pulled.get(can)
      if (!can || !clone) return
      const [px, py] = this.pullOf(can)
      this.activePullAnim?.cancel()
      this.activePullAnim = null
      clone.dataset.pulling = ''
      clone.dataset.progress = String(t)
      clone.style.transition = 'none'
      clone.style.transform = `translate(${px * t}px, ${py * t}px)`
    }
    viewport.addEventListener(
      'scroll',
      () => {
        if (
          Math.abs(viewport.scrollLeft - this.writtenScroll.left) > 1 ||
          Math.abs(viewport.scrollTop - this.writtenScroll.top) > 1
        ) {
          this.cancelGlide()
        }
      },
      { passive: true, signal },
    )
    viewport.addEventListener(
      'pointerdown',
      evt => {
        clearDragged()
        if (evt.button !== 0 || evt.isPrimary === false) return
        this.cancelGlide()
        const node = (evt.target as Element).closest<SVGGElement>('.cn-can')
        const index = Number(node?.dataset.index ?? -1)
        const can = rack.cans.get(index)
        const [px, py] = can ? this.pullOf(can) : [0, 0]
        gesture = {
          mode: 'pending',
          id: evt.pointerId,
          x: evt.clientX,
          y: evt.clientY,
          left: viewport.scrollLeft,
          top: viewport.scrollTop,
          index: can && px * px + py * py > 1 ? index : -1,
          px,
          py,
        }
      },
      { signal },
    )
    viewport.addEventListener(
      'pointermove',
      evt => {
        if (gesture && evt.pointerId === gesture.id) {
          if (evt.buttons === 0) {
            endGesture(evt)
            return
          }
          const dx = evt.clientX - gesture.x
          const dy = evt.clientY - gesture.y
          const dist = Math.hypot(dx, dy)
          if (gesture.mode === 'pending') {
            if (dist < DRAG_PX) return
            const len = Math.hypot(gesture.px, gesture.py)
            const dot =
              gesture.index >= 0 && len > 0 ? (dx * gesture.px + dy * gesture.py) / (len * dist) : 0
            const can = gesture.index >= 0 ? rack.cans.get(gesture.index) : undefined
            const existing = can ? this.pulled.get(can) : undefined
            const isOut = Boolean(existing && existing.dataset.out !== undefined)
            const live = existing ? this.pullProgress(existing, gesture.px, gesture.py) : 0
            const canPullOut = !isOut && dot >= PULL_ALIGN
            const canPushIn = isOut && dot <= -PULL_ALIGN
            if (can && (canPullOut || canPushIn)) {
              const startT = canPushIn ? Math.max(0.5, live) : live
              if (this.current >= 0 && this.current !== gesture.index) this.release()
              this.activePullAnim?.cancel()
              this.activePullAnim = null
              const clone = existing ?? this.mountClone(gesture.index)
              if (!clone) {
                gesture = {
                  mode: 'pan',
                  id: gesture.id,
                  x: gesture.x,
                  y: gesture.y,
                  left: gesture.left,
                  top: gesture.top,
                  moved: false,
                }
              } else {
                placePull(gesture.index, startT)
                capturePointer(evt.pointerId)
                viewport.dataset.pulling = ''
                gesture = {
                  mode: 'pull',
                  id: gesture.id,
                  x: gesture.x,
                  y: gesture.y,
                  index: gesture.index,
                  px: gesture.px,
                  py: gesture.py,
                  start: startT,
                  t: startT,
                }
              }
            } else {
              gesture = {
                mode: 'pan',
                id: gesture.id,
                x: gesture.x,
                y: gesture.y,
                left: gesture.left,
                top: gesture.top,
                moved: false,
              }
            }
          }
          if (gesture.mode === 'pan') {
            const panDx = evt.clientX - gesture.x
            const panDy = evt.clientY - gesture.y
            if (!gesture.moved && Math.hypot(panDx, panDy) < DRAG_PX) return
            if (!gesture.moved) {
              gesture.moved = true
              capturePointer(evt.pointerId)
              viewport.dataset.dragging = ''
            }
            this.writeScroll(gesture.left - panDx, gesture.top - panDy)
            return
          }
          if (gesture.mode !== 'pull') return
          const k = this.sceneScale()
          const along =
            ((evt.clientX - gesture.x) / k) * gesture.px +
            ((evt.clientY - gesture.y) / k) * gesture.py
          const t = Math.min(
            1,
            Math.max(
              0,
              gesture.start + along / (gesture.px * gesture.px + gesture.py * gesture.py),
            ),
          )
          gesture.t = t
          placePull(gesture.index, t)
          return
        }
        // A pointer that moves onto the pulled can opens its page preview, once per lift.
        if (evt.pointerType === 'touch' || evt.buttons !== 0 || this.current < 0) return
        const can = (evt.target as Element).closest<SVGGElement>('.vault-lift .cn-can')
        if (!can || this.hovering === this.current) return
        this.hovering = this.current
        this.hover(evt, can, this.films[this.current].file.path)
      },
      { signal },
    )
    const endGesture = (evt?: PointerEvent) => {
      const active = gesture
      if (!active) return
      if (evt && evt.pointerId !== undefined && evt.pointerId !== active.id) return
      gesture = undefined
      delete viewport.dataset.dragging
      delete viewport.dataset.pulling
      releasePointer(active.id)
      if (active.mode === 'pan') {
        if (active.moved) markDragged()
        return
      }
      if (active.mode === 'pull') {
        markDragged()
        this.finishPull(active.index, active.t)
      }
    }
    viewport.addEventListener('pointerup', endGesture, { signal })
    viewport.addEventListener('pointercancel', endGesture, { signal })
    viewport.addEventListener('lostpointercapture', endGesture, { signal })
    viewport.addEventListener(
      'wheel',
      evt => {
        if (!evt.ctrlKey && !evt.metaKey) return
        evt.preventDefault()
        const rect = viewport.getBoundingClientRect()
        const factor = Math.exp(-evt.deltaY * 0.0025)
        this.setScale(
          this.scale * factor,
          (evt.clientX ?? rect.left + this.viewW / 2) - rect.left,
          (evt.clientY ?? rect.top + this.viewH / 2) - rect.top,
        )
      },
      { passive: false, signal },
    )
    viewport.addEventListener(
      'click',
      evt => {
        // After a drag the click is dropped, so no can lifts or opens.
        if (dragged) {
          clearDragged()
          return
        }
        const can = (evt.target as Element).closest<SVGGElement>('.cn-can')
        if (!can) return this.release()
        const index = Number(can.dataset.index)
        if (index === this.current) this.openNote(index, evt)
        else this.select(index, true, true)
      },
      { signal },
    )
    viewport.addEventListener(
      'keydown',
      evt => {
        clearDragged()
        if (evt.altKey || evt.ctrlKey || evt.metaKey) {
          if (evt.key === 'Enter' && this.current >= 0) this.openNote(this.current, evt)
          return
        }
        const key = evt.key
        if (key === 'ArrowRight' || key === 'ArrowDown') this.step(1)
        else if (key === 'ArrowLeft' || key === 'ArrowUp') this.step(-1)
        else if (key === 'Enter' && this.current >= 0) this.openNote(this.current, evt)
        else if (key === 'Enter') this.select(0, true, true)
        else if (key === ' ') this.select(this.current < 0 ? 0 : this.current, true, true)
        else if (key === '+' || key === '=') this.setScale(this.scale * ZOOM)
        else if (key === '-') this.setScale(this.scale / ZOOM)
        else if (key === '0') this.fit()
        else return
        evt.preventDefault()
      },
      { signal },
    )
    controls.addEventListener(
      'click',
      evt => {
        const action = (evt.target as Element).closest<HTMLElement>('[data-camera]')?.dataset.camera
        if (action === 'fit') this.fit()
        else if (action === 'in') this.setScale(this.scale * ZOOM)
        else if (action === 'out') this.setScale(this.scale / ZOOM)
        else if (action === 'fullscreen') {
          if (this.fullscreen) void document.exitFullscreen().catch(() => undefined)
          else void this.root.requestFullscreen().catch(() => undefined)
        }
      },
      { signal },
    )
    this.host.addEventListener(
      'click',
      evt => {
        const row = (evt.target as Element).closest<HTMLElement>('.vault-entry')
        if (row) this.select(Number(row.dataset.index), true, true)
      },
      { signal },
    )
    const observer = new ResizeObserver(() => {
      this.root.dataset.layout = this.root.clientWidth >= WIDE_PX ? 'wide' : 'narrow'
      this.measure()
      if (this.scale < this.fitScale)
        this.setScale(this.fitScale, this.viewW / 2, this.viewH / 2, true)
      else this.layoutCanvas(this.scale)
    })
    observer.observe(viewport)
    signal.addEventListener('abort', () => observer.disconnect(), { once: true })
    return rack
  }
}
