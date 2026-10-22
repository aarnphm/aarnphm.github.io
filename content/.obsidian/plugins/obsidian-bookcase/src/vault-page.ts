import {
  type App,
  type BasesEntry,
  type BasesEntryGroup,
  type BasesPropertyId,
  type BasesViewConfig,
  Component,
  type HoverParent,
  MarkdownRenderer,
  type TFile,
} from 'obsidian'
import type { ModelReader } from './model'
import type { VaultEntry } from './vault-model'
import type { VaultConfig } from './vault-options'

const PAGE_MS = 140

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

export interface VaultFilm {
  entry: BasesEntry
  file: TFile
  title: string
  cover: string | null
}

export function readVaultFilms(
  app: App,
  reader: ModelReader,
  groups: BasesEntryGroup[],
  settings: VaultConfig,
): { films: VaultFilm[]; entries: VaultEntry[] } {
  const text = (entry: BasesEntry, id: BasesPropertyId) =>
    reader.read(entry, id)?.toString().trim() ?? ''
  const num = (entry: BasesEntry, id: BasesPropertyId) => {
    const raw = text(entry, id)
    const value = Number(raw)
    return raw && Number.isFinite(value) ? value : undefined
  }
  const grouped = groups.some(group => group.hasKey())
  const films: VaultFilm[] = []
  const entries: VaultEntry[] = []
  const { props } = settings
  for (const group of groups) {
    const label = grouped && group.hasKey() && group.key ? group.key.toString() : null
    for (const entry of group.entries) {
      const { file } = entry
      const title = text(entry, props.labelTitle) || file.basename
      const rawCover = reader.cover(reader.read(entry, props.image), file.path)
      const cover =
        rawCover?.kind === 'vault'
          ? app.vault.getResourcePath(rawCover.file)
          : rawCover?.kind === 'url'
            ? rawCover.url
            : null
      const film: VaultFilm = { entry, file, title, cover }
      films.push(film)
      const description = text(entry, 'note.description' as BasesPropertyId)
      entries.push({
        slug: file.path,
        path: file.path,
        href: file.path,
        title,
        directors: reader.strings(entry, props.labelDirector),
        year: num(entry, props.labelYear),
        rating: num(entry, props.labelRating),
        status: text(entry, props.labelStatus) || undefined,
        studio: reader.strings(entry, props.labelStudio)[0],
        watched: text(entry, props.labelWatched) || undefined,
        runtime: num(entry, props.labelRuntime),
        description: description || undefined,
        cover: cover ?? undefined,
        group: grouped ? (label ?? 'none') : null,
        rows: [],
      })
    }
  }
  return { films, entries }
}

export interface VaultPageHost extends HoverParent {
  app: App
  config: BasesViewConfig
  addChild<T extends Component>(component: T): T
  removeChild<T extends Component>(component: T): T
}

export class VaultPage {
  private timer: number | null = null
  private token = 0
  private child: Component | null = null

  constructor(
    private host: VaultPageHost,
    private el: HTMLElement,
    private reader: ModelReader,
  ) {}

  show(
    film: VaultFilm | undefined,
    record: Element | null | undefined,
    settings: VaultConfig | null,
  ): void {
    if (this.timer !== null) window.clearTimeout(this.timer)
    this.timer = window.setTimeout(
      () => {
        this.timer = null
        void this.render(film, record, settings, ++this.token)
      },
      film ? PAGE_MS : 0,
    )
  }

  unload(): void {
    if (this.timer !== null) window.clearTimeout(this.timer)
    this.timer = null
    this.token++
    this.dropChild()
    this.el.empty()
  }

  private dropChild(): void {
    if (this.child) this.host.removeChild(this.child)
    this.child = null
    this.el.empty()
  }

  private async render(
    film: VaultFilm | undefined,
    record: Element | null | undefined,
    settings: VaultConfig | null,
    token: number,
  ): Promise<void> {
    this.dropChild()
    if (!film || !settings) return
    const el = this.el
    if (film.cover)
      el.createEl('figure', { cls: 'vault-page-poster' }).createEl('img', {
        attr: { src: film.cover, alt: '', decoding: 'async' },
      })
    el.createEl('h2', { cls: 'vault-page-title' }).createEl('a', {
      cls: 'internal-link',
      text: film.title,
      attr: { 'data-href': film.file.path, href: film.file.path },
    })
    const detail = el.createDiv({ cls: 'vault-page-record' })
    for (const child of Array.from(record?.childNodes ?? [])) detail.append(child.cloneNode(true))
    const list = detail.querySelector('.vault-props')
    if (list) {
      const anchor = list.children[2] ?? null
      const skip = new Set<string>(Object.values(settings.props))
      for (const id of this.host.config.getOrder()) {
        const name = id.replace(/^(note|file|formula)\./, '')
        if (skip.has(id) || CARD_KEYS.has(name)) continue
        const value = this.reader.read(film.entry, id)
        if (!value) continue
        const row = createDiv({ cls: 'vault-prop' })
        row.createEl('dt', { text: this.host.config.getDisplayName(id) })
        value.renderTo(row.createEl('dd'), this.host)
        list.insertBefore(row, anchor)
      }
    }
    if (!settings.pageContent) return
    const { app } = this.host
    const text = await app.vault.cachedRead(film.file)
    if (token !== this.token) return
    const end = app.metadataCache.getFileCache(film.file)?.frontmatterPosition?.end.offset ?? 0
    const body = text.slice(end).trim()
    if (!body) return
    const child = this.host.addChild(new Component())
    this.child = child
    const bodyEl = el.createDiv({ cls: 'vault-page-body markdown-rendered' })
    await MarkdownRenderer.render(app, body, bodyEl, film.file.path, child)
  }
}
