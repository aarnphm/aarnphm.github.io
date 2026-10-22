// The film vault: the garden's rack drawing with its camera (native scroll, drag to pan, drag a can
// out or back in, zoom, fit)
// and pull, and the bookcase's page panel and editor pane in place of the garden's vault card.
import {
  BasesView,
  type HoverPopover,
  Keymap,
  type QueryController,
  type WorkspaceLeaf,
} from 'obsidian'
import type BookcasePlugin from '../main'
import { EditorPane } from './editor-pane'
import { ModelReader } from './model'
import { type SavedVaultCamera, type VaultCamera, createVaultCamera } from './vault-camera'
import { bindVaultGestures } from './vault-gestures'
import { vaultMarkup } from './vault-model'
import { type VaultMotion, createVaultMotion } from './vault-motion'
import { type VaultConfig, readVaultConfig } from './vault-options'
import { type VaultFilm, VaultPage, readVaultFilms } from './vault-page'
import { createVaultPull, pullOf } from './vault-pull'
import { type VaultScene, prepareVaultScene } from './vault-scene'

export const VAULT_VIEW = 'vault'

const WIDE_PX = 600
const EMBED_HEIGHT = 520
const FULLSCREEN_ICON = 'M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7'
const EXIT_ICON = 'M4 14h6v6M3 21l7-7M20 10h-6V4M21 3l-7 7'

/** One rebuild's elements and listeners; a rebuild or unload aborts them. */
interface Rack {
  abort: AbortController
  viewport: HTMLElement
  svg: SVGSVGElement
  scene: VaultScene
  camera: VaultCamera
  motion: VaultMotion
  cans: Map<number, SVGGElement>
  rows: Map<number, HTMLElement>
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
  private page: VaultPage
  private reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
  private pull = createVaultPull(this.reduced, {
    cans: () => this.rack?.cans,
    motion: () => this.rack?.motion,
    scene: () => this.rack?.scene,
    titleOf: index => this.films[index]?.title ?? '',
    holding: () => Boolean(this.rack?.camera.holding),
    current: () => this.current,
    onPutBack: index => {
      this.rack?.rows
        .get(index)
        ?.querySelector('.vault-entry-select')
        ?.removeAttribute('aria-current')
      this.closeHover()
    },
    onSnapOut: index => {
      this.putBack()
      this.current = index
      this.currentPath = this.films[index].file.path
      this.hovering = -1
      this.markCurrent(index)
      this.root.toggleClass('has-selection', true)
      if (this.linked) this.editor.show(null, this.films[index].file.path)
      else this.showPage(index)
    },
    onSnapIn: (_index, from, speed) => this.release(from, speed),
  })
  private settings: VaultConfig | null = null
  private films: VaultFilm[] = []
  private rack: Rack | null = null
  private rebuildRaf: number | null = null
  private observer: ResizeObserver | null = null
  private scroller: HTMLElement | null = null
  /** The first frame reads the top-left of the first bay; later rebuilds keep the camera. */
  private framed = false

  private current = -1
  private currentPath: string | null = null
  /** Lifted can hover-link fired for, once per lift; -1 for none. */
  private hovering = -1
  private linked = false
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
    this.page = new VaultPage(this, this.pageEl, this.reader)
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
    this.rebuildRaf = null
    this.pull.clear(this.rack?.motion)
    this.closeHover()
    // The pane may hold unsaved edits: the view lets go of it and never closes it.
    this.editor.forget()
    if (this.fullscreen) void document.exitFullscreen().catch(() => undefined)
    this.rack?.abort.abort()
    this.rack = null
    this.page.unload()
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
    const { films, entries } = readVaultFilms(
      this.app,
      this.reader,
      this.data.groupedData,
      settings,
    )
    this.films = films
    const previous = this.rack
    const saved: SavedVaultCamera | undefined =
      previous && this.framed
        ? {
            scale: previous.camera.scale,
            left: previous.viewport.scrollLeft,
            top: previous.viewport.scrollTop,
          }
        : undefined
    this.pull.clear(previous?.motion)
    previous?.abort.abort()
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
    this.host.innerHTML = vaultMarkup(entries, settings.view)
    this.fitHeight()
    this.rack = this.wire(saved)
    if (!this.rack) return
    this.framed = true
    // A rebuild keeps the pick when its note is still in the view.
    const again = this.currentPath ? films.findIndex(f => f.file.path === this.currentPath) : -1
    // An open editor pane keeps its note: a rebuild follows every edit made in it.
    if (again >= 0) {
      this.select(again, false, false, false)
      if (!this.linked) this.showPage(again)
    } else this.release()
  }

  // ---- layout

  /**
   * In a pane, the view fills what the pane shows below its toolbar; the rack scrolls inside it.
   * An embed gets a fixed height, full screen the whole screen.
   */
  private fitHeight(): void {
    this.setLayout()
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

  /** Writes the layout only when it changes: any attribute write on the root restyles the rack. */
  private setLayout(): void {
    const layout = this.root.clientWidth >= WIDE_PX ? 'wide' : 'narrow'
    if (this.root.dataset.layout !== layout) this.root.dataset.layout = layout
  }

  // ---- selection

  /** The pick: the can slides out with its lid label, and the page panel or editor follows it. */
  private select(index: number, camera = true, open = false, follow = true): void {
    const rack = this.rack
    const can = rack?.cans.get(index)
    if (!rack || !can) return
    const [px, py] = pullOf(can)
    if (index !== this.current || !this.pull.has(can)) {
      this.putBack()
      this.current = index
      this.currentPath = this.films[index].file.path
      this.hovering = -1
      this.pull.pullOut(index, can)
      this.markCurrent(index)
    }
    if (camera) {
      rack.camera.lookAt(Number(can.dataset.cx) + px * 0.6, Number(can.dataset.cy) + py * 0.6)
    }
    this.root.toggleClass('has-selection', true)
    if (!follow) return
    if (open && this.openEditor(this.films[index].file.path)) return
    if (this.linked) this.editor.show(null, this.films[index].file.path)
    else this.showPage(index)
  }

  private markCurrent(index: number): void {
    const row = this.rack?.rows.get(index)
    row?.querySelector('.vault-entry-select')?.setAttribute('aria-current', 'true')
    this.corners(index)
    this.liveEl.setText(`${this.films[index].title}, ${index + 1} of ${this.films.length}`)
  }

  /** Returns the pulled can to its slot. */
  private putBack(from?: number, speed = 0): void {
    this.pull.putBack(from, speed)
  }

  /** Clears the pick: the can goes back, the page panel closes, an editor pane keeps its note. */
  private release(from?: number, speed = 0): void {
    this.putBack(from, speed)
    this.current = -1
    this.currentPath = null
    this.corners(-1)
    this.root.toggleClass('has-selection', false)
    this.showPage(-1)
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
    const film = index >= 0 && index === this.current ? this.films[index] : undefined
    const record = film ? this.rack?.rows.get(index)?.querySelector('.vault-entry-detail') : null
    this.page.show(film, record, this.settings)
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
    requestAnimationFrame(() => this.rack?.camera.syncSize())
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

  private wire(saved?: SavedVaultCamera): Rack | null {
    const viewport = this.host.querySelector<HTMLElement>('.vault-viewport')
    const controls = this.host.querySelector<HTMLElement>('.vault-controls')
    const scene = prepareVaultScene(this.host)
    if (!viewport || !controls || !scene) return null
    const { svg } = scene
    const abort = new AbortController()
    const { signal } = abort
    const cans = new Map<number, SVGGElement>()
    for (const can of Array.from(svg.querySelectorAll<SVGGElement>(':scope > .cn-can')))
      cans.set(Number(can.dataset.index), can)
    const rows = new Map<number, HTMLElement>()
    for (const row of Array.from(this.host.querySelectorAll<HTMLElement>('.vault-entry')))
      rows.set(Number(row.dataset.index), row)
    // The index stays for screen readers; the arrow keys on the rack walk it for everyone else.
    for (const button of Array.from(this.host.querySelectorAll<HTMLElement>('.vault-index button')))
      button.tabIndex = -1

    const full = controls.createEl('button', {
      attr: { type: 'button', 'data-camera': 'fullscreen', 'aria-pressed': 'false' },
    })
    full
      .createSvg('svg', { attr: { viewBox: '0 0 24 24', 'aria-hidden': 'true' } })
      .createSvg('path', { attr: { d: FULLSCREEN_ICON } })
    const status = controls.querySelector('.vault-status')
    if (status) controls.insertBefore(full, status)

    const camera = createVaultCamera(this.host, viewport, svg, this.reduced, signal, saved, () =>
      this.setLayout(),
    )
    const motion = createVaultMotion(svg)
    const rack: Rack = { abort, viewport, svg, scene, camera, motion, cans, rows }
    this.rack = rack
    this.updateFullscreenButton()

    bindVaultGestures({
      host: this.host,
      viewport,
      controls,
      signal,
      camera,
      cans,
      pulled: this.pull.pulled,
      pullOf,
      pullProgress: clone => this.pull.pullProgress(clone),
      current: () => this.current,
      release: () => this.release(),
      mountClone: index => this.pull.mountClone(index),
      placePull: (index, t) => this.pull.placePull(index, t),
      finishPull: (index, t, speed) => this.pull.finishPull(index, t, speed),
      cancelPull: () => this.pull.cancelPull(),
      onCanClick: (index, evt) => {
        if (index === this.current) this.openNote(index, evt)
        else this.select(index, true, true)
      },
      onPulledHover: (evt, can) => {
        if (this.hovering === this.current || this.current < 0) return
        this.hovering = this.current
        this.hover(evt, can, this.films[this.current].file.path)
      },
      onStep: delta => this.step(delta),
      onSelect: index => this.select(index, true, true),
      onOpenNote: (index, evt) => this.openNote(index, evt),
      onToggleFullscreen: () => {
        if (this.fullscreen) void document.exitFullscreen().catch(() => undefined)
        else void this.root.requestFullscreen().catch(() => undefined)
      },
    })
    signal.addEventListener(
      'abort',
      () => {
        this.pull.clear(motion)
        motion.dispose()
        scene.dispose()
      },
      { once: true },
    )
    scene.start(signal)
    return rack
  }
}
