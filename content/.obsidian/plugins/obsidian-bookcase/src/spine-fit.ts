// The garden cuts spine labels by an average glyph width, which runs short for wide note fonts:
// a monospace `--font-text` overruns the author. The plugin re-cuts them with the real font.
import type { SceneBook } from './scene-model'

export interface SpineFonts {
  title: string
  author: string
}

/** The families `.bc-spine-title` and `.bc-spine-author` resolve to. */
export function spineFonts(el: HTMLElement): SpineFonts {
  const css = getComputedStyle(el)
  return {
    title: css.getPropertyValue('--font-text').trim() || 'serif',
    author: css.getPropertyValue('--font-interface').trim() || 'sans-serif',
  }
}

let context: CanvasRenderingContext2D | null = null

function measurer(font: string): (text: string) => number {
  context ??= document.createElement('canvas').getContext('2d')
  const ctx = context
  if (!ctx) return () => 0
  ctx.font = font
  return text => ctx.measureText(text).width
}

/** Longest prefix of `text` that fits `room`, with an ellipsis when cut. */
function fit(text: string, room: number, width: (text: string) => number): string {
  if (width(text) <= room) return text
  const base = text.endsWith('…') ? text.slice(0, -1) : text
  const cut = (n: number) => `${base.slice(0, n).trimEnd()}…`
  let lo = 0
  let hi = base.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (width(cut(mid)) <= room) lo = mid
    else hi = mid - 1
  }
  return lo === 0 ? '' : cut(lo)
}

export interface SpineLabels {
  title: string
  author: string
}

const fitted = new WeakMap<object, { fonts: SpineFonts; labels: SpineLabels }>()

/** Spine title and author cut to the room `spineTextW` gives them, measured in `fonts`. */
export function fitSpine(book: SceneBook<unknown>, fonts: SpineFonts): SpineLabels {
  const hit = fitted.get(book)
  if (hit?.fonts === fonts) return hit.labels
  const { title, author } = book.text
  // spineTextW: labels run from the head frame to `foot`, where the author ends.
  const foot = author?.x ?? book.height - 34
  let room = foot
  let authorText = ''
  if (author) {
    const width = measurer(`${author.size}px ${fonts.author}`)
    authorText = fit(author.text, foot * 0.35, width)
    if (authorText) room -= width(authorText) + 10
  }
  const titleText = title ? fit(title.text, room, measurer(`${title.size}px ${fonts.title}`)) : ''
  const labels = { title: titleText, author: authorText }
  fitted.set(book, { fonts, labels })
  return labels
}
