import {
  type App,
  type BasesEntry,
  type BasesEntryGroup,
  type BasesPropertyId,
  ListValue,
  type TFile,
  type Value,
} from 'obsidian'
import type { BookcaseConfig } from './options'
import type { ShelfGroup } from './scene-model'

export interface BookModel {
  /** Position in the flattened order. A note in several groups has one model per copy. */
  index: number
  group: number
  path: string
  file: TFile
  /** Entry of the current result set, for the page panel; replaced on every rebuild. */
  entry: BasesEntry
  /** Full value. */
  title: string
  /** Text before the first ':'. */
  spine: string
  authors: string[]
  cover: { kind: 'vault'; file: TFile } | { kind: 'url'; url: string } | null
  /** Validated `spineColor`, as hex when the colour is opaque. */
  colorOverride: string | null
}

const isList = (value: Value): value is ListValue =>
  typeof ListValue === 'function'
    ? value instanceof ListValue
    : typeof (value as ListValue).length === 'function'

export class ModelReader {
  private logged = new Set<string>()
  private colors = new Map<string, string | null>()
  private probe: CanvasRenderingContext2D | null = null

  constructor(private app: App) {}

  read(entry: BasesEntry, id: BasesPropertyId | null): Value | null {
    if (!id) return null
    let value: Value | null
    try {
      value = entry.getValue(id)
    } catch {
      value = null
    }
    if (!value || !value.isTruthy()) return null
    if (value.toString().startsWith('Error')) {
      if (!this.logged.has(id)) {
        this.logged.add(id)
        console.debug(`bookcase: ${id} evaluates to an error`, value.toString())
      }
      return null
    }
    return value
  }

  /** Text of each value of `id` on `entry`; a list gives one string per item. */
  strings(entry: BasesEntry, id: BasesPropertyId | null): string[] {
    return this.list(this.read(entry, id))
  }

  /** Shelves as Bases groups the entries. */
  groups(grouped: BasesEntryGroup[], config: BookcaseConfig): ShelfGroup<BookModel>[] {
    let index = 0
    const out: ShelfGroup<BookModel>[] = []
    for (const group of grouped) {
      const g = out.length
      out.push({
        label: group.hasKey() && group.key ? group.key.toString() : null,
        books: group.entries.map(entry => {
          const model = this.model(entry, config, index++, g)
          return { path: model.path, item: model }
        }),
      })
    }
    return out
  }

  /**
   * Shelves of one topic: `entries` in the view's order, grouped by the first value of `by`.
   * Groups follow their label; books without one come last, on an unlabeled shelf.
   */
  topicGroups(
    entries: BasesEntry[],
    by: BasesPropertyId,
    config: BookcaseConfig,
  ): ShelfGroup<BookModel>[] {
    const byLabel = new Map<string | null, BasesEntry[]>()
    for (const entry of entries) {
      const label = this.strings(entry, by)[0] ?? null
      const list = byLabel.get(label)
      if (list) list.push(entry)
      else byLabel.set(label, [entry])
    }
    const labels = [...byLabel.keys()].sort((a, b) =>
      a === null ? 1 : b === null ? -1 : a.localeCompare(b),
    )
    let index = 0
    return labels.map((label, g) => ({
      label,
      books: byLabel.get(label)!.map(entry => {
        const model = this.model(entry, config, index++, g)
        return { path: model.path, item: model }
      }),
    }))
  }

  private model(
    entry: BasesEntry,
    config: BookcaseConfig,
    index: number,
    group: number,
  ): BookModel {
    const file = entry.file
    const title = this.read(entry, config.spineTitle)?.toString().trim() || file.basename
    const colon = title.indexOf(':')
    return {
      index,
      group,
      path: file.path,
      file,
      entry,
      title,
      spine: (colon > 0 ? title.slice(0, colon) : title).trim(),
      authors: this.list(this.read(entry, config.spineAuthor)),
      cover: this.cover(this.read(entry, config.image), file.path),
      colorOverride: this.color(this.read(entry, config.spineColor)?.toString().trim() ?? null),
    }
  }

  private list(value: Value | null): string[] {
    if (!value) return []
    if (!isList(value)) return [value.toString()].filter(Boolean)
    const out: string[] = []
    for (let i = 0; i < value.length(); i++) {
      const item = value.get(i)
      if (item.isTruthy()) out.push(item.toString())
    }
    return out
  }

  cover(value: Value | null, source: string): BookModel['cover'] {
    const first = value ? (isList(value) ? value.get(0) : value) : null
    const raw = first?.isTruthy() ? first.toString().trim() : ''
    if (!raw) return null
    if (/^https?:\/\//i.test(raw)) return { kind: 'url', url: raw }
    const link = raw
      .replace(/^!/, '')
      .replace(/^\[\[/, '')
      .replace(/\]\]$/, '')
      .split('|')[0]
      .trim()
    const file = link ? this.app.metadataCache.getFirstLinkpathDest(link, source) : null
    return file ? { kind: 'vault', file } : null
  }

  private color(raw: string | null): string | null {
    if (!raw) return null
    const cached = this.colors.get(raw)
    if (cached !== undefined) return cached
    // The value lands in a style attribute, so the form is fixed before CSS judges it.
    const shaped =
      /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(raw) ||
      /^(rgb|hsl|oklch)\([0-9a-z.,%+\-\s/]*\)$/i.test(raw)
    let out: string | null = null
    if (shaped && CSS.supports('color', raw)) {
      this.probe ??= createEl('canvas').getContext('2d')
      out = raw
      if (this.probe) {
        this.probe.fillStyle = '#000000'
        this.probe.fillStyle = raw
        const hex = String(this.probe.fillStyle)
        if (/^#[0-9a-f]{6}$/i.test(hex)) out = hex.toUpperCase()
      }
    }
    this.colors.set(raw, out)
    return out
  }
}

/** Spine label for the author: two names, then "et al.". */
export function authorLabel(authors: string[]): string {
  if (authors.length <= 2) return authors.join(', ')
  return `${authors.slice(0, 2).join(', ')} et al.`
}
