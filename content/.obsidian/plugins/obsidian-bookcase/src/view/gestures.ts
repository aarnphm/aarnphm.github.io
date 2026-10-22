import { Keymap } from 'obsidian'
import type { BookModel } from '../model'
import type { Point, SceneBook } from '../scene-model'
import type { ViewHost } from './view-host'

/** The area a convex outline sweeps when it moves by `d`: the hull of both positions. */
function sweep(points: readonly Point[], [dx, dy]: Point): Point[] {
  const all = [...points, ...points.map(([x, y]): Point => [x + dx, y + dy])].sort(
    (a, b) => a[0] - b[0] || a[1] - b[1],
  )
  const cross = (o: Point, a: Point, b: Point) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const half = (list: Point[]) => {
    const out: Point[] = []
    for (const p of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop()
      out.push(p)
    }
    out.pop()
    return out
  }
  return [...half(all), ...half([...all].reverse())]
}

/** Even-odd test of a point against a closed polygon. */
function contains(points: readonly Point[], x: number, y: number): boolean {
  let inside = false
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i]
    const [xj, yj] = points[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

export function buttonIndex(this: ViewHost, target: EventTarget | null): number | null {
  const button = (target as Element | null)?.closest<HTMLElement>('button[data-index]')
  const i = Number(button?.dataset.index)
  return button && Number.isInteger(i) ? i : null
}

export function onKey(this: ViewHost, evt: KeyboardEvent): void {
  const scene = this.scene
  const i = this.buttonIndex(evt.target)
  if (!scene || i === null) return
  const books = scene.books
  const last = books.length - 1
  const shelfStart = (shelf: number) => books.findIndex(b => b.shelf === shelf)
  let next: number | null = null
  switch (evt.key) {
    case 'ArrowRight':
    case 'ArrowDown':
      next = Math.min(last, i + 1)
      break
    case 'ArrowLeft':
    case 'ArrowUp':
      next = Math.max(0, i - 1)
      break
    case 'PageDown': {
      const start = shelfStart(books[i].shelf + 1)
      next = start >= 0 ? start : last
      break
    }
    case 'PageUp': {
      const start = shelfStart(books[i].shelf - 1)
      next = start >= 0 ? start : 0
      break
    }
    case 'Home':
      next = 0
      break
    case 'End':
      next = last
      break
    case 'Enter':
      evt.preventDefault()
      this.select(i)
      this.openNote(books[i].path, evt)
      return
    default:
      return
  }
  evt.preventDefault()
  this.select(next)
  this.moveTo(next, true)
}

export function bookAt(this: ViewHost, target: EventTarget | null): SceneBook<BookModel> | null {
  const g = (target as Element | null)?.closest<SVGGElement>('[data-book]')
  const index = Number(g?.dataset.book)
  return g && Number.isInteger(index) ? (this.scene?.books[index] ?? null) : null
}

export function onStageClick(this: ViewHost, evt: MouseEvent): void {
  // A press that moves a few px off a book ends on another element; the click then goes to
  // their common ancestor. It is still a click on what was pressed.
  const pressed = this.pressed
  this.pressed = null
  const target =
    pressed && (evt.target as Element).contains(pressed) ? pressed : (evt.target as Element)
  if (target.closest('.bookcase-controls')) return
  const book = this.bookAt(target)
  if (!book) {
    this.deselect()
    return
  }
  if (target.closest('.bookcase-lift')) {
    this.openNote(book.path, evt)
    return
  }
  this.select(book.index)
  // Keys go on from the clicked book, and Escape and a blur reach the view.
  this.moveTo(book.index, true)
}

export function openNote(this: ViewHost, path: string, evt: MouseEvent | KeyboardEvent): void {
  const mod = Keymap.isModEvent(evt)
  if (!mod && this.linked && this.editor.focus()) return
  void this.app.workspace.openLinkText(path, '', mod)
}

export function onPointerMove(this: ViewHost, evt: PointerEvent): void {
  // Touch has no hover; a held button is a press or a drag.
  if (evt.pointerType === 'touch' || evt.buttons !== 0) return
  const [x, y] = this.pointerAt
  this.pointerAt = [evt.clientX, evt.clientY]
  if (evt.clientX === x && evt.clientY === y) return
  if (!this.lifted) return
  if (this.overLift(evt.clientX, evt.clientY)) this.near = true
  else if (this.near) {
    this.putBack()
    return
  }
  const el = (evt.target as Element).closest<HTMLElement>('.bookcase-lift')
  const book = el ? this.bookAt(evt.target) : null
  if (!el || !book || book.index !== this.selected || book.index === this.hovering) return
  this.hovering = book.index
  this.hover(evt, el, book.path)
}

export function overLift(this: ViewHost, x: number, y: number): boolean {
  const book = this.lifted ? this.scene?.books[this.selected] : undefined
  if (!book) return false
  if (this.swept?.index !== book.index || this.swept.version !== this.version)
    this.swept = {
      index: book.index,
      version: this.version,
      area: sweep(book.faces.edge, book.pull),
    }
  const box = this.world.getBoundingClientRect()
  const { minX, minY, scale } = this.frame
  const u = (x - box.left) / (this.zoom * scale) + minX
  const v = (y - box.top) / (this.zoom * scale) + minY
  return contains(this.swept.area, u, v)
}

export function hover(
  this: ViewHost,
  event: MouseEvent,
  targetEl: HTMLElement,
  linktext: string,
): void {
  this.app.workspace.trigger('hover-link', {
    event,
    source: 'bases',
    hoverParent: this,
    targetEl,
    linktext,
  })
}
