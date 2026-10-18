import { Component } from 'obsidian'
import type { BookModel } from './model'
import { INK_DARK, type Point, type SceneBook, rgbToHex, spineInk } from './scene-model'
import { type Frame, createSvgLayer, placeSvg } from './stage'
import { type BookPaint, type Box, bookBox, fillText, liftMarkup } from './svg'

export interface LiftBook {
  index: number
  markup: string
  /** The book's box where it stands, in screen units. */
  box: Box
  /** Pull offset in screen units. */
  pull: Point
  /** Cover image URL, set on the `<image>` after the markup is written. */
  image: string | null
  fill: (el: SVGSVGElement) => void
}

/** A book to lift, built only when no element holds it yet. */
export interface LiftRequest {
  index: number
  /** Changes whenever the drawn book would: the data, its colours, its cover URL or the scale. */
  key: string
  build: () => LiftBook
}

interface Slot {
  el: SVGSVGElement
  index: number
  key: string
  pull: Point
  /** False while the slot holds a book written ahead of time and not yet shown. */
  shown: boolean
  /** Time at which a returning book gives its slot back; null while it is not returning. */
  until: number | null
  markup: string
  image: string | null
  /** The cover URL the `<image>` holds. */
  loaded: string | null
  scale: number
}

/**
 * A returning book gives its slot back after this fixed time, not on `transitionend`: a slide cut
 * short by a new focus ends early, and its end would then land on the next focus change too.
 */
const SETTLE_MS = 260
const COVER_SIZE = 12
const COVER_LINES = 4

function darken(hex: string, k: number): string | null {
  const a = /^#([0-9a-f]{6})$/i.exec(hex)
  const b = /^#([0-9a-f]{6})$/i.exec(INK_DARK)
  if (!a || !b) return null
  const ch = (v: string, i: number) => parseInt(v.slice(i, i + 2), 16)
  const mix = (i: number) => ch(a[1], i) * (1 - k) + ch(b[1], i) * k
  return rgbToHex(mix(0), mix(2), mix(4))
}

/** Front-cover title for a book without a cover image, wrapped to the cover's width. */
export function coverLines(book: SceneBook<BookModel>): string[] {
  const max = Math.max(4, Math.floor((book.depth - 16) / (COVER_SIZE * 0.5)))
  const out: string[] = []
  let line = ''
  for (const word of book.item.title.split(/\s+/).filter(Boolean)) {
    const next = line ? `${line} ${word}` : word
    if (next.length <= max || !line) line = next
    else {
      out.push(line)
      line = word
    }
  }
  if (line) out.push(line)
  const cut = (l: string) => (l.length > max ? `${l.slice(0, max - 1)}…` : l)
  const lines = out.slice(0, COVER_LINES).map(cut)
  if (out.length > COVER_LINES)
    lines[COVER_LINES - 1] = `${lines[COVER_LINES - 1].slice(0, max - 1)}…`
  return lines
}

/** The lifted copy of `book`: full colour, its cover image or title, and the `--bc-hi` edge. */
export function liftContent(
  book: SceneBook<BookModel>,
  paint: BookPaint,
  image: string | null,
  text: (key: string) => string | string[],
): LiftBook {
  const side = darken(paint.spine, 0.3)
  const markup = liftMarkup(book, paint, {
    image: image !== null && book.cover !== null,
    ink: side ? spineInk(side) : paint.ink,
    lines: image && book.cover ? 0 : coverLines(book).length,
  })
  return {
    index: book.index,
    markup,
    box: bookBox(book),
    pull: book.pull,
    image: book.cover ? image : null,
    fill: el => fillText(el, text),
  }
}

let motionQuery: MediaQueryList | null = null
// One query object for the session: it follows the OS setting, and a lookup per focus is costly.
const reducedMotion = () =>
  (motionQuery ??= window.matchMedia('(prefers-reduced-motion: reduce)')).matches

/**
 * The pulled-out book (SPEC 6.3). The incoming book slides out along its pull vector, the
 * outgoing one slides back and then gives its slot back to the shelf copy. A third element holds
 * the book that comes next, written in idle time, so a focus change only has to show it.
 */
export class Lift extends Component {
  private slots: [Slot, Slot, Slot]
  private current: Slot | null = null
  private outgoing: Slot | null = null
  private frame: Frame = { minX: 0, minY: 0, scale: 1 }
  private raf: number | null = null
  private tick: number | null = null

  constructor(
    world: HTMLElement,
    private setHidden: (index: number, hidden: boolean) => void,
  ) {
    super()
    const make = (): Slot => {
      const el = createSvgLayer(world, 'bookcase-lift')
      el.style.display = 'none'
      return {
        el,
        index: -1,
        key: '',
        pull: [0, 0],
        shown: false,
        until: null,
        markup: '',
        image: null,
        loaded: null,
        scale: 0,
      }
    }
    this.slots = [make(), make(), make()]
  }

  onload(): void {
    this.register(() => {
      if (this.raf !== null) cancelAnimationFrame(this.raf)
      if (this.tick !== null) cancelAnimationFrame(this.tick)
      for (const slot of this.slots) {
        slot.el.remove()
      }
    })
  }

  get index(): number | null {
    return this.current?.index ?? null
  }

  setFocusVisible(on: boolean): void {
    for (const slot of this.slots) slot.el.toggleClass('is-focus-visible', on)
  }

  /** Lifts the requested book, or puts every book back when null. */
  show(request: LiftRequest | null, frame: Frame): void {
    this.frame = frame
    if (request && this.current?.index === request.index) return
    if (request && this.outgoing?.index === request.index) {
      // Focus came back before the book settled: retarget the running transition.
      const back = this.outgoing
      back.until = null
      this.outgoing = null
      if (this.current) this.retire(this.current)
      this.current = back
      this.pull(back)
      return
    }
    if (this.current) this.retire(this.current)
    this.current = null
    if (!request) return
    const ready = this.slots.find(s => s !== this.outgoing && s.key === request.key)
    const slot = ready ?? this.slots.find(s => s !== this.outgoing && !s.shown) ?? this.slots[0]
    const reduced = reducedMotion()
    if (!ready) this.write(slot, request.key, request.build())
    this.reveal(slot, !reduced)
    this.current = slot
    if (reduced) {
      this.pull(slot)
      return
    }
    // Called from a frame callback: this frame paints the book at rest, the next one pulls it,
    // so the transition starts at rest without reading layout back. The cover, which shows
    // only once the book is out, loads on that second frame too.
    slot.el.style.transform = 'none'
    this.nextFrame(() => {
      if (this.current !== slot) return
      this.pull(slot)
      this.setImage(slot)
    })
  }

  /** Puts every book back at once, before a rebuild that replaces the books. */
  clear(): void {
    for (const slot of this.slots) if (slot.index >= 0) this.finish(slot)
  }

  /** Writes the requested book and its cover into a spare element, hidden, for a later `show`. */
  prepare(request: LiftRequest, frame: Frame): void {
    if (this.current?.index === request.index || this.outgoing?.index === request.index) return
    const spare = this.slots.filter(s => s !== this.current && s !== this.outgoing && !s.shown)
    if (spare.length === 0 || spare.some(s => s.key === request.key)) return
    this.frame = frame
    this.write(spare[0], request.key, request.build())
    this.setImage(spare[0])
  }

  /**
   * Brings the current book up to date after a rebuild or a finished cover: a new cover URL
   * only changes the image's href; new markup or a new scale redraws it in place.
   */
  update(request: LiftRequest, frame: Frame): void {
    const slot = this.current
    if (!slot || slot.index !== request.index || slot.key === request.key) return
    const book = request.build()
    if (slot.markup === book.markup && slot.scale === frame.scale) {
      slot.key = request.key
      slot.image = book.image
      this.setImage(slot)
      return
    }
    this.frame = frame
    this.write(slot, request.key, book)
    this.reveal(slot)
    slot.el.addClass('is-instant')
    this.pull(slot)
    this.nextFrame(() => slot.el.removeClass('is-instant'))
  }

  private nextFrame(callback: () => void): void {
    if (this.raf !== null) cancelAnimationFrame(this.raf)
    this.raf = requestAnimationFrame(() => {
      this.raf = null
      callback()
    })
  }

  /** Writes the book into the element, which stays hidden until `reveal`. */
  private write(slot: Slot, key: string, book: LiftBook): void {
    if (slot.shown && slot.index !== book.index) this.finish(slot)
    this.cancelReturn(slot)
    slot.index = book.index
    slot.key = key
    slot.pull = book.pull
    slot.markup = book.markup
    slot.image = book.image
    slot.scale = this.frame.scale
    placeSvg(slot.el, book.box, this.frame)
    // Geometry, classes and colours only; note text and the cover URL go in through the DOM.
    slot.el.innerHTML = book.markup
    book.fill(slot.el)
    slot.loaded = null
  }

  /** Shows the element and hides the shelf copy. The cover can wait for the pull frame. */
  private reveal(slot: Slot, later = false): void {
    this.cancelReturn(slot)
    slot.shown = true
    if (!later) this.setImage(slot)
    slot.el.style.display = ''
    this.setHidden(slot.index, true)
  }

  private setImage(slot: Slot): void {
    if (!slot.image || slot.loaded === slot.image) return
    slot.loaded = slot.image
    slot.el.querySelector('image')?.setAttribute('href', slot.image)
  }

  private pull(slot: Slot): void {
    const [x, y] = slot.pull
    const { scale } = this.frame
    slot.el.style.transform = `translate(${x * scale}px, ${y * scale}px)`
  }

  private retire(slot: Slot): void {
    if (this.outgoing && this.outgoing !== slot) this.finish(this.outgoing)
    this.outgoing = slot
    slot.el.style.transform = 'none'
    this.cancelReturn(slot)
    if (reducedMotion()) {
      this.finish(slot)
      return
    }
    slot.until = performance.now() + SETTLE_MS
    this.watch()
  }

  private finish(slot: Slot): void {
    this.cancelReturn(slot)
    if (slot.shown) this.setHidden(slot.index, false)
    slot.shown = false
    slot.index = -1
    slot.key = ''
    slot.markup = ''
    slot.image = null
    // Hidden, not emptied: the next write replaces the children in one step.
    slot.el.style.display = 'none'
    slot.el.style.transform = 'none'
    if (this.outgoing === slot) this.outgoing = null
    if (this.current === slot) this.current = null
  }

  /** Gives slots back on the first frame after their time, so the shelf copy shows in step. */
  private watch(): void {
    if (this.tick !== null) return
    this.tick = requestAnimationFrame(now => {
      this.tick = null
      let waiting = false
      for (const slot of this.slots) {
        if (slot.until === null) continue
        if (now >= slot.until) this.finish(slot)
        else waiting = true
      }
      if (waiting) this.watch()
    })
  }

  private cancelReturn(slot: Slot): void {
    slot.until = null
  }
}
