import type { WorkspaceLeaf } from 'obsidian'
import type { BookModel } from '../model'
import type { Scene } from '../scene-model'
import type { ViewHost } from './view-host'

/** A book the scroll selected opens in the editor pane once the scroll rests this long. */
const SCROLL_OPEN_MS = 300

export function syncSelection(this: ViewHost): void {
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

export function select(this: ViewHost, index: number, scrolled = false): void {
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

export function deselect(this: ViewHost): void {
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

export function clearSelection(this: ViewHost): void {
  this.selected = -1
  this.selectedPath = null
  this.putBack()
  this.markSelected(-1)
}

export function resolveSelection(this: ViewHost, scene: Scene<BookModel>): void {
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

export function putBack(this: ViewHost): void {
  this.near = false
  this.hovering = -1
  this.closeHover()
  if (!this.lifted) return
  this.lifted = false
  this.lift.show(null, this.frame)
}

export function summary(this: ViewHost): string {
  const count = this.scene?.books.length ?? 0
  const books = `${count} ${count === 1 ? 'book' : 'books'}`
  const topic = this.tabs.find(tab => tab.id === this.filter?.topic)
  return topic ? `${topic.label} · ${books}` : books
}

export function hostLeaf(this: ViewHost): WorkspaceLeaf | null {
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

export function openEditor(this: ViewHost, path: string): boolean {
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

export function setLinked(this: ViewHost, on: boolean, rebuild = true): void {
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

export function closeHover(this: ViewHost): void {
  const popover = this.hoverPopover
  if (!popover) return
  this.hoverPopover = null
  popover.hoverEl.detach()
  popover.unload()
}

export function rove(this: ViewHost, index: number): void {
  if (index === this.roving) return
  this.buttons[this.roving]?.setAttribute('tabindex', '-1')
  this.roving = index
  this.buttons[index]?.setAttribute('tabindex', '0')
}

export function markSelected(this: ViewHost, index: number): void {
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

export function updateIndex(this: ViewHost, scene: Scene<BookModel>): void {
  const books = scene.books
  while (this.buttons.length > books.length) this.buttons.pop()?.parentElement?.remove()
  while (this.buttons.length < books.length) {
    const i = this.buttons.length
    const button = this.indexEl
      .createEl('li')
      .createEl('button', {
        attr: { type: 'button', tabindex: i === this.roving ? '0' : '-1', 'data-index': String(i) },
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

export function moveTo(this: ViewHost, index: number, focusButton: boolean): void {
  const settings = this.settings
  if (!settings) return
  if (focusButton) {
    this.rove(index)
    // The index sits at the top of the view; focusing it must not scroll the track back.
    this.buttons[index]?.focus({ preventScroll: true })
  }
  this.stage.scrollTo(index, settings.scrollStep)
}
