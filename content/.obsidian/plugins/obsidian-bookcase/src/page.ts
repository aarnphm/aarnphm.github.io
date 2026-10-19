import {
  type App,
  type BasesPropertyId,
  type BasesViewConfig,
  Component,
  type HoverParent,
  MarkdownRenderer,
} from 'obsidian'
import type { BookModel, ModelReader } from './model'
import type { BookcaseConfig } from './options'

const DELAY_MS = 140

export interface PageHost extends HoverParent {
  app: App
  config: BasesViewConfig
  addChild<T extends Component>(component: T): T
  removeChild<T extends Component>(component: T): T
}

/** The sheet next to the case with the selected book's title, properties and body (SPEC 8). */
export class Page {
  private timer: number | null = null
  private token = 0
  private child: Component | null = null
  private shown: string | null = null
  private pending: {
    id: string
    book: BookModel | null
    config: BookcaseConfig
    note: string
    at: number
  } | null = null
  private source = ''

  constructor(
    private host: PageHost,
    private el: HTMLElement,
    private reader: ModelReader,
  ) {}

  /** Source path for links on the page. */
  get sourcePath(): string {
    return this.source
  }

  /**
   * Renders `book`, or with no book the quiet `note`, once the selection has rested for DELAY_MS.
   * The same book again keeps its pending render. One timer serves a run of changes: when it fires
   * early it waits out the rest.
   */
  schedule(book: BookModel | null, config: BookcaseConfig, force = false, note = ''): void {
    const id = book ? `${book.index}:${book.path}` : `-:${note}`
    if (!force && id === (this.timer === null ? this.shown : this.pending?.id)) return
    this.pending = { id, book, config, note, at: performance.now() }
    if (this.timer === null) this.arm(DELAY_MS)
  }

  private arm(delay: number): void {
    this.timer = window.setTimeout(() => {
      this.timer = null
      const pending = this.pending
      if (!pending) return
      const rest = pending.at + DELAY_MS - performance.now()
      if (rest > 1) {
        this.arm(rest)
        return
      }
      this.pending = null
      this.shown = pending.id
      void this.render(pending.book, pending.config, pending.note, ++this.token)
    }, delay)
  }

  unload(): void {
    if (this.timer !== null) window.clearTimeout(this.timer)
    this.timer = null
    this.pending = null
    this.token++
    this.dropChild()
    this.el.empty()
  }

  private dropChild(): void {
    if (this.child) this.host.removeChild(this.child)
    this.child = null
  }

  private async render(
    book: BookModel | null,
    config: BookcaseConfig,
    note: string,
    token: number,
  ): Promise<void> {
    const { app } = this.host
    this.dropChild()
    this.el.empty()
    this.source = ''
    if (!book) {
      if (note) this.el.createEl('p', { cls: 'bookcase-page-empty', text: note })
      return
    }
    const { entry, path } = book
    this.source = path

    // Clicks and hovers on this link and on links in the body are delegated by the view.
    this.el
      .createEl('h2', { cls: 'bookcase-page-title' })
      .createEl('a', {
        cls: 'internal-link',
        text: book.title,
        attr: { 'data-href': path, href: path },
      })

    const year = this.reader.read(entry, 'note.year')?.toString()
    const byline = [book.authors.join(', '), year].filter(Boolean).join(' · ')
    if (byline) this.el.createEl('p', { cls: 'bookcase-page-byline', text: byline })

    const skip = new Set<BasesPropertyId>([config.spineTitle, config.spineAuthor, config.image])
    const list = this.el.createEl('dl', { cls: 'bookcase-page-properties' })
    for (const id of this.host.config.getOrder()) {
      if (skip.has(id)) continue
      const value = this.reader.read(entry, id)
      if (!value) continue
      list.createEl('dt', { text: this.host.config.getDisplayName(id) })
      // The view implements hoverPopover, which is all a RenderContext carries.
      value.renderTo(list.createEl('dd'), this.host)
    }
    if (!list.hasChildNodes()) list.remove()

    const description = this.reader.read(entry, 'note.description')?.toString()
    if (description) this.el.createEl('p', { cls: 'bookcase-page-description', text: description })

    if (!config.pageContent) return
    const text = await app.vault.cachedRead(book.file)
    if (token !== this.token) return
    const end = app.metadataCache.getFileCache(book.file)?.frontmatterPosition?.end.offset ?? 0
    const body = text.slice(end).trim()
    if (!body) return
    const child = this.host.addChild(new Component())
    this.child = child
    const el = this.el.createDiv({ cls: 'bookcase-page-body markdown-rendered' })
    await MarkdownRenderer.render(app, body, el, path, child)
  }
}
