import { Keymap } from 'obsidian'
import type { ViewHost } from './view-host'
import { ZOOM } from './camera'

export function onload(this: ViewHost): void {
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
  // Plus and minus zoom while focus is inside the view; category initials select and focus topics.
  this.registerDomEvent(this.root, 'keydown', evt => {
    if (evt.defaultPrevented || evt.ctrlKey || evt.metaKey || evt.altKey || evt.isComposing) return
    if (
      evt.target instanceof Element &&
      evt.target.closest('input, textarea, select, [contenteditable]')
    )
      return
    if (evt.key === '+' || evt.key === '=') this.zoomBy(ZOOM)
    else if (evt.key === '-') this.zoomBy(1 / ZOOM)
    else if (/^\p{L}$/u.test(evt.key) && this.panel.stepByLetter(evt.key)) {
      // Topic selected and focused by panel.
    } else return
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

  this.registerDomEvent(this.stageEl, 'pointerdown', evt => (this.pressed = evt.target as Element))
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

export function onunload(this: ViewHost): void {
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
