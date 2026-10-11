import type { LectureNote } from './store'
import { PDF_QUOTE_LIMIT, type PdfQuote } from '../../util/pdf-marks'
import { rangesForQuotes } from '../arena-feed/quote-highlights'
import { findClosestHeading, getRangeOffsets } from '../multiplayer/anchor'

export interface LectureSelection {
  range: Range
  quote: PdfQuote
  text: string
  heading: string | null
}

function quotedMarkdown(range: Range): string {
  const fragment = range.cloneContents()
  for (const math of fragment.querySelectorAll('.katex')) {
    const tex = math.querySelector('annotation[encoding="application/x-tex"]')?.textContent
    if (tex) {
      const text = math.closest('.katex-display') ? `\n\n$$\n${tex}\n$$\n\n` : `$${tex}$`
      math.replaceWith(document.createTextNode(text))
    } else if (math.querySelector('.katex-html')) math.querySelector('.katex-mathml')?.remove()
  }
  return fragment.textContent?.trim() ?? ''
}

export function lectureSelection(article: HTMLElement): LectureSelection | null {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || !selection.rangeCount) return null
  const range = selection.getRangeAt(0)
  if (!article.contains(range.startContainer) || !article.contains(range.endContainer)) return null
  const offsets = getRangeOffsets(range, article)
  if (!offsets || !offsets.text.trim() || offsets.text.length > PDF_QUOTE_LIMIT) return null
  const text = article.textContent ?? ''
  return {
    range: range.cloneRange(),
    text: quotedMarkdown(range),
    heading: findClosestHeading(range.startContainer),
    quote: {
      exact: offsets.text,
      prefix: text.slice(Math.max(0, offsets.startOffset - 64), offsets.startOffset),
      suffix: text.slice(offsets.endOffset, offsets.endOffset + 64),
    },
  }
}

export function noteRange(article: HTMLElement, note: LectureNote): Range | null {
  if (!note.quote) return null
  const section = note.heading
    ? article.querySelector<HTMLElement>(`#${CSS.escape(note.heading)}`)
    : null
  return rangesForQuotes(section ?? article, [note.quote])[0] ?? null
}

export function showNoteRange(article: HTMLElement, range: Range): void {
  let element = range.startContainer.parentElement
  while (element && element !== article) {
    if (element.classList.contains('collapsible-header')) {
      const toggle = element.querySelector<HTMLElement>(
        ':scope > .collapse-shell > :is(h1, h2, h3, h4, h5, h6) [data-collapse-toggle][aria-expanded="false"]',
      )
      toggle?.click()
    }
    element = element.parentElement
  }
  range.startContainer.parentElement?.scrollIntoView({ block: 'center', behavior: 'auto' })
  article.focus({ preventScroll: true })
}
