import {
  type App,
  Component,
  type EventRef,
  Events,
  normalizePath,
  type Plugin,
  type TFile,
} from 'obsidian'
import type { BookModel } from './model'
import { fallbackSpine, hashPath, rgbToHex, spineInk } from './scene-model'

export interface CoverEntry {
  key: string
  color: string
  ink: string
  aspect: number
  /** Vault-relative path of the thumbnail. */
  thumb: string
}

export type FilterPanel = 'topics' | 'tags'

/** What one view shows to this viewer: a topic, the tags that narrow it, and the open panel tab. */
export interface ViewFilter {
  topic: string
  tags: string[]
  panel: FilterPanel
}

/** Plugin data: the cover index, and each view's filter for this viewer. */
interface CoverIndex {
  version: 1
  covers: Record<string, CoverEntry>
  /** Keyed by `<.base path>#<view name>`. Never written into the .base file. */
  views?: Record<string, ViewFilter>
}

const isFilter = (value: unknown): value is ViewFilter =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as ViewFilter).topic === 'string' &&
  Array.isArray((value as ViewFilter).tags) &&
  (value as ViewFilter).tags.every(tag => typeof tag === 'string')

/** A saved filter, with the panel tab that older data lacks. */
const readFilter = (value: unknown): ViewFilter | null =>
  isFilter(value)
    ? { topic: value.topic, tags: value.tags, panel: value.panel === 'tags' ? 'tags' : 'topics' }
    : null

interface Need {
  /** Cover of each book in reading order; null for books without a vault cover. */
  files: (TFile | null)[]
  focus: number
  /** False once a scan found nothing left to cache. */
  open: boolean
}

const THUMB_HEIGHT = 480
const STRIP = 0.06
const SAMPLES = 12
const SAVE_DELAY = 2000

const keys = new WeakMap<TFile, { stat: string; key: string }>()

function coverKey(file: TFile): string {
  const stat = `${file.path}:${file.stat.mtime}:${file.stat.size}`
  const known = keys.get(file)
  if (known?.stat === stat) return known.key
  const key = hashPath(stat).toString(16)
  keys.set(file, { stat, key })
  return key
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function toBlob(canvas: HTMLCanvasElement, type: string, quality?: number): Promise<Blob | null> {
  return new Promise(resolve => canvas.toBlob(resolve, type, quality))
}

/**
 * Decodes each vault cover once, in idle time, into a small thumbnail and a sampled spine colour.
 * The index lives in plugin data; the thumbnails in the plugin's cache folder.
 */
export class CoverCache extends Component {
  private index: CoverIndex = { version: 1, covers: {} }
  private events = new Events()
  private needs = new Map<object, Need>()
  /** Resource URL of each thumbnail; the adapter may hand out a new URL on every call. */
  private urls = new Map<string, string>()
  private failed = new Set<string>()
  private idle: number | null = null
  private idleKind: 'idle' | 'timeout' = 'timeout'
  private saveTimer: number | null = null
  private working = false
  private folderReady = false
  private stopped = false

  constructor(private plugin: Plugin) {
    super()
  }

  private get folder(): string {
    return normalizePath(`${this.plugin.manifest.dir ?? ''}/cache/covers`)
  }

  async load(): Promise<void> {
    const data = (await this.plugin.loadData()) as Partial<CoverIndex> | null
    if (data?.version !== 1) return
    const { covers, views } = data
    if (covers && typeof covers === 'object') this.index.covers = covers
    if (views && typeof views === 'object')
      this.index.views = Object.fromEntries(
        Object.entries(views).flatMap(([key, f]) => {
          const filter = readFilter(f)
          return filter ? [[key, filter]] : []
        }),
      )
  }

  /** The filter this viewer last chose in the view `key`. */
  filter(key: string): ViewFilter | null {
    return this.index.views?.[key] ?? null
  }

  setFilter(key: string, filter: ViewFilter): void {
    this.index.views = {
      ...this.index.views,
      [key]: { topic: filter.topic, tags: [...filter.tags], panel: filter.panel },
    }
    this.save()
  }

  onunload(): void {
    this.stopped = true
    this.cancelIdle()
    this.needs.clear()
    if (this.saveTimer !== null) {
      window.clearTimeout(this.saveTimer)
      this.saveTimer = null
      void this.plugin.saveData(this.index)
    }
  }

  on(name: 'cover', callback: (path: string) => void): EventRef {
    return this.events.on(name, callback as (...data: unknown[]) => unknown)
  }

  /** A fresh entry for the cover, or null while it is missing or stale. */
  get(file: TFile): CoverEntry | null {
    const entry = this.index.covers[file.path]
    return entry && entry.key === coverKey(file) ? entry : null
  }

  thumbUrl(entry: CoverEntry): string {
    let url = this.urls.get(entry.thumb)
    if (url === undefined) {
      url = this.plugin.app.vault.adapter.getResourcePath(entry.thumb)
      this.urls.set(entry.thumb, url)
    }
    return url
  }

  /** Covers an open view shows, nearest to `focus` first. */
  need(owner: object, files: (TFile | null)[], focus: number): void {
    this.needs.set(owner, { files, focus, open: true })
    this.schedule()
  }

  /** Moves the queue's centre for `owner` without rescanning its covers. */
  focus(owner: object, focus: number): void {
    const need = this.needs.get(owner)
    if (!need || need.focus === focus) return
    need.focus = focus
    if (need.open) this.schedule()
  }

  release(owner: object): void {
    this.needs.delete(owner)
  }

  /** Spine colour: the `spineColor` property, else the sampled colour, else the fallback hue. */
  spineColour(model: BookModel, dark: boolean): string {
    if (model.colorOverride) return model.colorOverride
    const entry = model.cover?.kind === 'vault' ? this.get(model.cover.file) : null
    if (!entry) return fallbackSpine(model.path)
    if (!dark) return entry.color
    const [r, g, b] = [1, 3, 5].map(i => parseInt(entry.color.slice(i, i + 2), 16) * 0.88)
    return rgbToHex(r, g, b)
  }

  /** Thumbnail URL when cached; an external cover's own URL. */
  thumb(model: BookModel): string | null {
    const cover = model.cover
    if (!cover) return null
    if (cover.kind === 'url') return cover.url
    const entry = this.get(cover.file)
    return entry ? this.thumbUrl(entry) : null
  }

  /** Image for the lifted book: the thumbnail, or the original until it is cached. */
  image(model: BookModel, app: App): string | null {
    const cover = model.cover
    if (cover?.kind === 'vault' && !this.get(cover.file))
      return app.vault.getResourcePath(cover.file)
    return this.thumb(model)
  }

  private next(): TFile | null {
    for (const need of this.needs.values()) {
      if (!need.open) continue
      const { files, focus } = need
      for (let d = 0; d < files.length; d++) {
        for (const i of d === 0 ? [focus] : [focus - d, focus + d]) {
          const file = files[i]
          if (file && !this.failed.has(file.path) && !this.get(file)) return file
        }
      }
      need.open = false
    }
    return null
  }

  private schedule(): void {
    if (this.stopped || this.working || this.idle !== null) return
    if (![...this.needs.values()].some(need => need.open)) return
    const run = () => {
      this.idle = null
      void this.work()
    }
    if (typeof window.requestIdleCallback === 'function') {
      this.idleKind = 'idle'
      this.idle = window.requestIdleCallback(run, { timeout: 2000 })
    } else {
      this.idleKind = 'timeout'
      this.idle = window.setTimeout(run, 16)
    }
  }

  private cancelIdle(): void {
    if (this.idle === null) return
    if (this.idleKind === 'idle') window.cancelIdleCallback(this.idle)
    else window.clearTimeout(this.idle)
    this.idle = null
  }

  private async work(): Promise<void> {
    const file = this.next()
    if (!file) return
    this.working = true
    try {
      const entry = await this.decode(file)
      if (this.stopped) return
      const old = this.index.covers[file.path]
      if (old && old.thumb !== entry.thumb) void this.removeFile(old.thumb)
      this.index.covers[file.path] = entry
      this.save()
      this.events.trigger('cover', file.path)
    } catch (error) {
      this.failed.add(file.path)
      console.debug('bookcase: cover not cached', file.path, error)
    } finally {
      this.working = false
    }
    if (!this.stopped) this.schedule()
  }

  private async decode(file: TFile): Promise<CoverEntry> {
    const { vault } = this.plugin.app
    const buffer = await vault.readBinary(file)
    // Decoding the bytes keeps the canvas untainted, unlike drawing from an app:// URL.
    const bitmap = await createImageBitmap(new Blob([buffer]))
    try {
      const { width, height } = bitmap
      const aspect = height / width

      const strip = createEl('canvas', { attr: { width: 1, height: SAMPLES } })
      const sctx = strip.getContext('2d', { willReadFrequently: true })
      if (!sctx) throw new Error('no 2d context')
      sctx.drawImage(bitmap, 0, 0, Math.max(1, Math.round(width * STRIP)), height, 0, 0, 1, SAMPLES)
      const pixels = sctx.getImageData(0, 0, 1, SAMPLES).data
      const channel = (c: number) =>
        median(Array.from({ length: SAMPLES }, (_, i) => pixels[i * 4 + c]))
      const color = rgbToHex(channel(0), channel(1), channel(2))

      const th = Math.min(THUMB_HEIGHT, height)
      const tw = Math.max(1, Math.round(th / aspect))
      const canvas = createEl('canvas', { attr: { width: tw, height: th } })
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('no 2d context')
      ctx.drawImage(bitmap, 0, 0, tw, th)
      bitmap.close()

      let blob = await toBlob(canvas, 'image/webp', 0.82)
      let ext = 'webp'
      if (!blob || blob.type !== 'image/webp') {
        blob = await toBlob(canvas, 'image/jpeg', 0.82)
        ext = 'jpg'
      }
      if (!blob) throw new Error('no thumbnail')
      const key = coverKey(file)
      const thumb = `${this.folder}/${key}.${ext}`
      await this.ensureFolder()
      await vault.adapter.writeBinary(thumb, await blob.arrayBuffer())
      return { key, color, ink: spineInk(color), aspect, thumb }
    } finally {
      bitmap.close()
    }
  }

  private async ensureFolder(): Promise<void> {
    if (this.folderReady) return
    const { adapter } = this.plugin.app.vault
    if (!(await adapter.exists(this.folder))) await adapter.mkdir(this.folder)
    this.folderReady = true
  }

  private async removeFile(path: string): Promise<void> {
    try {
      const { adapter } = this.plugin.app.vault
      if (await adapter.exists(path)) await adapter.remove(path)
    } catch (error) {
      console.debug('bookcase: stale thumbnail kept', path, error)
    }
  }

  private save(): void {
    if (this.saveTimer !== null) window.clearTimeout(this.saveTimer)
    this.saveTimer = window.setTimeout(() => {
      this.saveTimer = null
      void this.plugin.saveData(this.index)
    }, SAVE_DELAY)
  }
}

/** Decoded thumbnails around the focus; at most `limit` held at once. */
export class Preload {
  private images = new Map<string, HTMLImageElement>()

  constructor(private limit = 7) {}

  set(urls: string[]): void {
    const wanted = new Set(urls.slice(0, this.limit))
    for (const [url, image] of this.images)
      if (!wanted.has(url)) {
        image.src = ''
        this.images.delete(url)
      }
    for (const url of wanted) {
      if (this.images.has(url)) continue
      const image = new Image()
      image.decoding = 'async'
      image.src = url
      image.decode().catch(() => undefined)
      this.images.set(url, image)
    }
  }

  clear(): void {
    this.set([])
  }
}
