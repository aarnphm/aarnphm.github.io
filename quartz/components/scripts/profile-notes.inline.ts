import { debounce } from '../../util/debounce'

// Profile notes hang in the margin (CSS: position absolute). Each one is set level with the word
// that cites it, then pushed down just enough to clear the note above.
const GAP = 16

function anchorOf(note: HTMLElement): Element | null {
  const id = CSS.escape(note.id)
  return document.querySelector(`[data-note="${id}"], a[href="#${id}"]`)
}

// Top of the text line that holds the citation. A footnote mark is a raised square, so measure
// the character just before it rather than the mark.
function lineTop(anchor: Element): number {
  const mark = anchor.closest('sup') ?? anchor
  const before = mark.previousSibling
  if (before?.nodeType === Node.TEXT_NODE && before.textContent) {
    const range = document.createRange()
    const end = before.textContent.length
    range.setStart(before, Math.max(0, end - 1))
    range.setEnd(before, end)
    const rects = range.getClientRects()
    if (rects.length > 0) return rects[rects.length - 1].top
  }
  if (before instanceof Element) {
    const rects = before.getClientRects()
    if (rects.length > 0) return rects[rects.length - 1].top
  }
  return anchor.getBoundingClientRect().top
}

function settle() {
  const notes = [
    ...document.querySelectorAll<HTMLElement>('article .profile-note:not(.profile-note-boxed)'),
  ]
  if (notes.length === 0) return
  for (const note of notes) note.style.top = ''
  const hanging = notes.filter(note => getComputedStyle(note).position === 'absolute')
  if (hanging.length === 0) return

  // One read pass, then one write pass.
  const placed = hanging.map(note => {
    const parent = note.offsetParent as HTMLElement | null
    const anchor = anchorOf(note)
    const rect = note.getBoundingClientRect()
    return {
      note,
      want: anchor ? lineTop(anchor) : rect.top,
      height: rect.height,
      origin: parent ? parent.getBoundingClientRect().top + parent.clientTop : 0,
    }
  })
  placed.sort((a, b) => a.want - b.want)
  let floor = -Infinity
  for (const { note, want, height, origin } of placed) {
    const top = Math.max(want, floor + GAP)
    floor = top + height
    note.style.top = `${top - origin}px`
  }

  // Absolute boxes inside multicol (the tldr) can land off their computed offset; measure once
  // more and correct by the residue.
  const residue = placed.map(({ note }, i) => {
    const want = parseFloat(note.style.top) + placed[i].origin
    return want - note.getBoundingClientRect().top
  })
  placed.forEach(({ note }, i) => {
    if (Math.abs(residue[i]) >= 1) note.style.top = `${parseFloat(note.style.top) + residue[i]}px`
  })
}

document.addEventListener('nav', () => {
  if (!document.querySelector('article .profile-note')) return
  const controller = new AbortController()
  const relayout = debounce(settle, 100)
  settle()
  void document.fonts?.ready.then(settle)
  window.addEventListener('resize', relayout, { passive: true, signal: controller.signal })
  window.addEventListener('load', settle, { once: true, signal: controller.signal })
  window.addCleanup(() => controller.abort())
})
