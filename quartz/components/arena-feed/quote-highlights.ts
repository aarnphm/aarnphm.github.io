import type { ArenaNoteQuote } from '../../util/arena-reader'

interface TextSegment {
  node: Text
  start: number
  end: number
}

export function compactQuoteText(value: string): string {
  return value.replace(/\s/gu, '')
}

export function findQuoteMatch(
  text: string,
  quote: ArenaNoteQuote,
): { start: number; end: number } | null {
  const exact = compactQuoteText(quote.exact)
  if (!exact) return null
  const matches: number[] = []
  let from = 0
  while (from < text.length) {
    const at = text.indexOf(exact, from)
    if (at < 0) break
    matches.push(at)
    from = at + 1
  }
  if (matches.length === 0) return null
  if (matches.length === 1) return { start: matches[0], end: matches[0] + exact.length }

  const prefix = compactQuoteText(quote.prefix)
  const suffix = compactQuoteText(quote.suffix)
  const contextual = matches.filter(at => {
    const before = text.slice(Math.max(0, at - prefix.length), at)
    const after = text.slice(at + exact.length, at + exact.length + suffix.length)
    return (!prefix || before === prefix) && (!suffix || after === suffix)
  })
  return contextual.length === 1
    ? { start: contextual[0], end: contextual[0] + exact.length }
    : null
}

function textIndex(root: HTMLElement): { text: string; segments: TextSegment[] } {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  const segments: TextSegment[] = []
  let text = ''
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (!(node instanceof Text)) continue
    if (node.parentElement?.closest('button, .pdf-embed-toolbar, .pdf-embed-page-label')) continue
    const compact = compactQuoteText(node.data)
    if (!compact) continue
    segments.push({ node, start: text.length, end: text.length + compact.length })
    text += compact
  }
  return { text, segments }
}

function textPoint(
  segments: TextSegment[],
  character: number,
  after: boolean,
): { node: Text; offset: number } | null {
  const segment = segments.find(item => item.start <= character && character < item.end)
  if (!segment) return null
  let position = segment.start
  for (let offset = 0; offset < segment.node.length; offset++) {
    if (/\s/u.test(segment.node.data[offset])) continue
    if (position === character) return { node: segment.node, offset: offset + Number(after) }
    position += 1
  }
  return null
}

export function rangesForQuotes(root: HTMLElement, quotes: ArenaNoteQuote[]): Range[] {
  const index = textIndex(root)
  const ranges: Range[] = []
  for (const quote of quotes) {
    const match = findQuoteMatch(index.text, quote)
    if (!match) continue
    const start = textPoint(index.segments, match.start, false)
    const end = textPoint(index.segments, match.end - 1, true)
    if (!start || !end) continue
    const range = document.createRange()
    range.setStart(start.node, start.offset)
    range.setEnd(end.node, end.offset)
    ranges.push(range)
  }
  return ranges
}
