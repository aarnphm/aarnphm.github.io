import { normalizePdfSlug, parsePdfManifest, PDF_MANIFEST_PATH } from '../../../util/pdf-marks'
import { isRecord } from '../../../util/type-guards'

const highlightName = 'pdf-search-match'
const stopWords = new Set([
  'the',
  'and',
  'from',
  'with',
  'that',
  'this',
  'why',
  'how',
  'for',
  'into',
])

function queryPattern(query: string): RegExp | null {
  const words = [...new Set(query.match(/[\p{L}\p{N}]+/gu) ?? [])]
    .filter(word => word.length > 1 && !stopWords.has(word.toLowerCase()))
    .sort((a, b) => b.length - a.length)
  return words.length ? new RegExp(words.join('|'), 'giu') : null
}

async function readRegions(url: URL, page: number, signal: AbortSignal) {
  const slug = normalizePdfSlug(url.pathname)
  if (!slug) return []
  const manifestResponse = await fetch(PDF_MANIFEST_PATH, { signal })
  if (!manifestResponse.ok) return []
  const manifest = parsePdfManifest(await manifestResponse.json())
  const doc = manifest?.documents[slug]?.doc
  if (!doc) return []
  const response = await fetch(`/embeddings/pdf-regions/${doc}.json`, { signal })
  if (!response.ok) return []
  const data: unknown = await response.json()
  if (!isRecord(data) || data.sourceSha256 !== doc || !isRecord(data.pages)) return []
  const regions: unknown = data.pages[String(page)]
  if (!Array.isArray(regions)) return []
  return regions.flatMap((region: unknown) => {
    if (!isRecord(region) || typeof region.text !== 'string' || !Array.isArray(region.bbox))
      return []
    const box: unknown[] = region.bbox
    if (
      box.length !== 4 ||
      !box.every(
        (n): n is number => typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1000,
      )
    )
      return []
    const [left, top, right, bottom] = box
    if (right <= left || bottom <= top) return []
    return [{ text: region.text, left, top, width: right - left, height: bottom - top }]
  })
}

export function highlightPdfPreview(
  root: HTMLElement,
  url: URL,
  page: number | undefined,
  query: string,
  signal: AbortSignal,
) {
  const pattern = queryPattern(query)
  if (!pattern || signal.aborted) return
  const selectedPage = page ?? 1
  let frame = 0
  const textReady = new Set(
    Array.from(
      root.querySelectorAll<HTMLElement>('.pdf-embed-text-layer[data-pdf-text-ready="true"]'),
    ).map(layer =>
      Number(layer.closest<HTMLElement>('[data-pdf-page-number]')?.dataset.pdfPageNumber),
    ),
  )
  let regions: Awaited<ReturnType<typeof readRegions>> = []
  let requestedRegions = false
  let nativeTextSeen = false
  const update = () => {
    frame = 0
    if (signal.aborted) return
    const ranges: Range[] = []
    for (const layer of root.querySelectorAll('.pdf-embed-text-layer')) {
      const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT)
      let node = walker.nextNode()
      while (node) {
        pattern.lastIndex = 0
        for (const match of (node.textContent ?? '').matchAll(pattern)) {
          const range = document.createRange()
          range.setStart(node, match.index)
          range.setEnd(node, match.index + match[0].length)
          ranges.push(range)
        }
        node = walker.nextNode()
      }
    }
    CSS.highlights.set(highlightName, new Highlight(...ranges))
    root.dataset.pdfSearchMatches = String(ranges.length)
    const slot = root.querySelector<HTMLElement>(
      `.pdf-embed-page[data-pdf-page-number="${selectedPage}"]`,
    )
    if (!slot) return
    const text = slot.querySelector('.pdf-embed-text-layer')?.textContent?.trim()
    if (text) nativeTextSeen = true
    if (nativeTextSeen || !textReady.has(selectedPage)) return
    if (!requestedRegions) {
      requestedRegions = true
      void readRegions(url, selectedPage, signal)
        .then(value => {
          regions = value
          schedule()
        })
        .catch(error => {
          if (!signal.aborted) console.warn('[Search] OCR highlight unavailable:', error)
        })
    }
    if (!regions.length || slot.querySelector('.pdf-search-regions')) return
    const overlay = document.createElement('div')
    overlay.className = 'pdf-search-regions'
    overlay.setAttribute('aria-label', 'Matching OCR text regions')
    for (const region of regions) {
      pattern.lastIndex = 0
      if (!pattern.test(region.text)) continue
      const mark = document.createElement('span')
      mark.title = 'matching OCR text region'
      mark.style.left = `${region.left / 10}%`
      mark.style.top = `${region.top / 10}%`
      mark.style.width = `${region.width / 10}%`
      mark.style.height = `${region.height / 10}%`
      overlay.append(mark)
    }
    slot.append(overlay)
    root.dataset.pdfSearchRegions = String(overlay.childElementCount)
  }
  const schedule = () => {
    if (!frame && !signal.aborted) frame = requestAnimationFrame(update)
  }
  const observer = new MutationObserver(records => {
    if (
      records.some(
        record =>
          !(record.target instanceof Element && record.target.closest('.pdf-search-regions')),
      )
    )
      schedule()
  })
  root.addEventListener(
    'pdf-text-rendered',
    event => {
      if (!(event instanceof CustomEvent)) return
      const number: unknown = event.detail
      if (typeof number !== 'number') return
      textReady.add(number)
      schedule()
    },
    { signal },
  )
  observer.observe(root, { childList: true, subtree: true })
  schedule()
  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      cancelAnimationFrame(frame)
      CSS.highlights.delete(highlightName)
      root.querySelectorAll('.pdf-search-regions').forEach(layer => layer.remove())
      delete root.dataset.pdfSearchMatches
      delete root.dataset.pdfSearchRegions
    },
    { once: true },
  )
}
