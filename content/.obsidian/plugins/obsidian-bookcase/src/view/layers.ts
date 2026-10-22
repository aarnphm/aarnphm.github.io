import type { BookModel } from '../model'
import type { Scene } from '../scene-model'
import type { ViewHost } from './view-host'
import { coverLines, liftContent, type LiftRequest } from '../lift'
import { fitSpine, spineFonts } from '../spine-fit'
import {
  bookStyle,
  paintBook,
  partsSignature,
  shelfBox,
  shelfMarkup,
  solidsBox,
  solidsMarkup,
  spineLabels,
  type BookPaint,
} from '../svg'

const PRELOAD_RADIUS = 3

/** Idle work waits until the scroll has rested this long. */
const QUIET_MS = 160

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

export function render(this: ViewHost, index: number): string {
  const scene = this.scene
  const layer = scene?.layers[index]
  if (!scene || !layer) return ''
  return layer.kind === 'solids'
    ? solidsMarkup(layer)
    : shelfMarkup(layer, scene.books, book => this.paint(book.index))
}

export function prepare(this: ViewHost, index: number, markup: string): string {
  const layer = this.scene?.layers[index]
  if (layer?.kind !== 'shelf' || this.hidden.size === 0) return markup
  for (const i of layer.books)
    if (this.hidden.has(i))
      markup = markup.replace(`data-book="${i}" `, `data-book="${i}" visibility="hidden" `)
  return markup
}

export function paintLayers(
  this: ViewHost,
  scene: Scene<BookModel>,
  previous: (BookPaint | undefined)[],
): void {
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

export function patchBook(this: ViewHost, layer: number, index: number): void {
  this.layers.invalidate(layer)
  this.bookElement(index)?.setAttribute('style', bookStyle(this.paint(index)))
}

export function paint(this: ViewHost, index: number): BookPaint {
  let paint = this.paints[index]
  if (!paint) {
    const book = this.scene!.books[index]
    paint = paintBook(book, this.plugin.covers.spineColour(book.item, this.dark), this.frame.scale)
    this.paints[index] = paint
  }
  return paint
}

export function bookElement(this: ViewHost, index: number): Element | null {
  const book = this.scene?.books[index]
  const layer = book ? this.shelfLayer.get(book.shelf) : undefined
  if (layer === undefined || !this.layers.isMounted(layer)) return null
  const children = this.layers.element(layer)?.children
  return children?.[children.length - this.fromEnd[index]] ?? null
}

export function applyColours(this: ViewHost): void {
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

export function requestSettle(this: ViewHost): void {
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

export function cancelSettle(this: ViewHost): void {
  if (this.settleTimer !== null) window.clearTimeout(this.settleTimer)
  this.settleIdle?.()
  this.preloadIdle?.()
  this.settleTimer = this.settleIdle = this.preloadIdle = null
}

export function settle(this: ViewHost, remaining: () => number): boolean {
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

export function text(this: ViewHost, key: string): string | string[] {
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

export function setHidden(this: ViewHost, index: number, hidden: boolean): void {
  if (this.hidden.has(index) === hidden) return
  if (hidden) this.hidden.add(index)
  else this.hidden.delete(index)
  const g = this.bookElement(index)
  if (hidden) g?.setAttribute('visibility', 'hidden')
  else g?.removeAttribute('visibility')
}

export function syncLift(this: ViewHost, index: number): void {
  const lifted = this.liftBook(index)
  if (!lifted) return
  if (this.lift.index === index) this.lift.update(lifted, this.frame)
  else this.lift.show(lifted, this.frame)
}

export function liftBook(this: ViewHost, index: number): LiftRequest | null {
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

export function preloadAround(this: ViewHost): void {
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
        const url = near?.item.cover?.kind === 'vault' ? this.plugin.covers.thumb(near.item) : null
        if (url) urls.push(url)
      }
    this.preload.set(urls)
  }, 500)
}
