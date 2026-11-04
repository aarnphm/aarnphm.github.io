import { type PdfMarkTarget, type PdfQuote } from '../../../util/pdf-marks'
import { rangesForQuotes } from '../../arena-feed/quote-highlights'
import { boxQuad, selectionOnPage } from '../geometry'
import { notePath, type OutlineEntry } from '../parts'
import { PdfViewer, type PdfOutlineNode, type ViewerPage } from '../viewer'

export function toast(
  message: string,
  action?: { label: string; onClick: () => void },
  durationMs = 2400,
) {
  document.dispatchEvent(new CustomEvent('toast', { detail: { message, action, durationMs } }))
}

export async function copyText(text: string, message: string) {
  try {
    await navigator.clipboard.writeText(text)
    toast(message)
    return true
  } catch {
    toast('Copy failed. The clipboard is unavailable here.')
    return false
  }
}

export function flattenOutline(
  nodes: PdfOutlineNode[],
  depth = 0,
  out: OutlineEntry[] = [],
): OutlineEntry[] {
  for (const node of nodes) {
    out.push({
      title: node.title.replace(/\s+/g, ' ').trim(),
      dest: node.dest,
      url: node.url ?? null,
      depth,
    })
    if (node.items?.length && out.length < 800) flattenOutline(node.items, depth + 1, out)
  }
  return out
}

export function isEditable(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target.isContentEditable)
  )
}

/** R1: a dev checkout without `git lfs pull` serves the pointer text instead of the bytes. */
export async function describeLoadFailure(pdfUrl: string, message: string): Promise<string> {
  try {
    const response = await fetch(pdfUrl, { headers: { Range: 'bytes=0-63' } })
    const head = await response.text()
    if (head.startsWith('version https://git-lfs')) {
      return `the file is a Git LFS pointer, not the PDF. Run git lfs pull --include "${pdfUrl.slice(1)}"`
    }
    if (response.status === 404) return 'the file is missing from the build (404)'
  } catch {
    /* keep the viewer's message */
  }
  return message
}

/** Clicks the strip link the keymap names (`prev`, `next`, `up`); true when one exists. */
export function followCourseKey(key: string): boolean {
  const link = document.querySelector<HTMLAnchorElement>(`a[data-course-key="${key}"]`)
  if (!link) return false
  link.click()
  return true
}

/** Finds a stale mark's quote near its old page and rebuilds its quads from the new text layer. */
export async function reanchor(
  viewer: PdfViewer,
  target: PdfMarkTarget,
): Promise<PdfMarkTarget | null> {
  if (target.type !== 'text') return target.page <= viewer.pageCount ? target : null
  const quote: PdfQuote = target.quote
  for (const offset of [0, -1, 1, -2, 2]) {
    const number = target.page + offset
    if (number < 1 || number > viewer.pageCount) continue
    const layer = await viewer.ensureTextLayer(number)
    const page: ViewerPage | undefined = viewer.pages[number - 1]
    if (!layer || !page?.base) continue
    const [range] = rangesForQuotes(layer, [quote])
    if (!range) continue
    const found = selectionOnPage(range)
    if (!found || found.page !== number) continue
    return {
      type: 'text',
      page: number,
      quads: found.boxes.map(box => boxQuad(box, page.base!)),
      quote: found.quote,
    }
  }
  return null
}

export { notePath }
