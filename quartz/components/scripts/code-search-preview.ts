import type { FullSlug } from '../../util/path'
import type { CodeSearchRow } from './code-search-controls'
import { codeMatchLocation, codeMatchUrl } from './code-search-controls'

export async function renderCodeMatch(
  current: FullSlug,
  match: CodeSearchRow,
  target: HTMLElement,
  signal: AbortSignal,
  compact = false,
  onMatchChange?: () => void,
) {
  const url = codeMatchUrl(current, match)
  const { readCodeManifest, readCodeShard } = await import('../../util/code-search-format')
  const manifestUrl = new URL('/static/code-search/manifest.json', location.href)
  const response = await fetch(manifestUrl, { signal, cache: 'no-cache' })
  if (!response.ok) throw new Error(`Source manifest unavailable: ${response.status}`)
  const manifest = readCodeManifest(await response.json())
  const reference = manifest.shards.find(
    shard =>
      shard.language === match.unit.language && shard.sourceIds.includes(match.unit.sourceId),
  )
  if (!reference) throw new Error('Source changed. Run the query again')
  const payload = await fetch(new URL(reference.path, manifestUrl), { signal })
  if (!payload.ok) throw new Error(`Source unavailable: ${payload.status}`)
  const shard = await readCodeShard(
    new Uint8Array(await payload.arrayBuffer()),
    reference,
    manifest.generation,
  )
  const source = shard.sources.find(source => source.id === match.unit.sourceId)?.text
  if (source === undefined) throw new Error('Source is unavailable')
  const firstLine = 1
  const { highlightCode, codeAttributes } = await import('../../util/highlight-source')
  if (signal.aborted) return
  // The previous match's figure stays mounted; only its caption and source swap.
  const mounted = target.querySelector<HTMLElement>(':scope > .code-match-preview')
  const mountedCaption = mounted?.querySelector<HTMLElement>(':scope > figure > figcaption')
  const mountedPre = mounted?.querySelector<HTMLPreElement>(':scope > figure > pre')
  const caption = mountedCaption ?? document.createElement('figcaption')
  const label = document.createElement('span')
  label.className = 'code-match-label'
  caption.replaceChildren(label)
  const previous = document.createElement('button')
  previous.type = 'button'
  previous.textContent = 'prev'
  previous.setAttribute('aria-label', 'previous match')
  const counter = document.createElement('span')
  counter.className = 'code-match-counter'
  counter.setAttribute('role', 'status')
  const next = document.createElement('button')
  next.type = 'button'
  next.textContent = 'next'
  next.setAttribute('aria-label', 'next match')
  // One range has nothing to step through, so the stepper only appears for several.
  if (!compact && match.ranges.length > 1) caption.append(previous, counter, next)
  const open = document.createElement('a')
  open.href = url.toString()
  open.textContent = match.unit.kind === 'attachment' ? 'open file' : 'open block'
  const pre = mountedPre ?? document.createElement('pre')
  pre.tabIndex = 0
  pre.setAttribute('aria-label', `Source code: ${match.unit.path}`)
  const highlighted = highlightCode(source, match.unit.language)
  const paint = () => {
    label.textContent = `${match.unit.language} · ${codeMatchLocation(match)}`
    counter.textContent = `${match.selectedRange + 1}/${match.ranges.length}`
    const code = document.createElement('code')
    for (const [name, value] of Object.entries(codeAttributes(highlighted))) {
      code.setAttribute(name, value)
    }
    code.innerHTML = highlighted.html
    code.style.counterReset = `line ${firstLine - 1}`
    const lines = code.querySelectorAll<HTMLElement>('[data-line]')
    lines.forEach((line, index) => {
      const lineNumber = index + 1
      line.dataset.sourceLine = String(firstLine + index)
      const intervals = match.ranges
        .slice(match.selectedRange, match.selectedRange + 1)
        .flatMap(range => {
          if (lineNumber < range.start.line || lineNumber > range.end.line) return []
          const start = lineNumber === range.start.line ? range.start.column - 1 : 0
          const end =
            lineNumber === range.end.line ? range.end.column - 1 : (line.textContent?.length ?? 0)
          return end > start ? [{ start, end }] : []
        })
      if (!intervals.length) return
      line.dataset.codeMatch = 'true'
      const walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT)
      const nodes: Text[] = []
      while (walker.nextNode())
        if (walker.currentNode instanceof Text) nodes.push(walker.currentNode)
      let offset = 0
      for (const node of nodes) {
        const text = node.data
        const fragment = document.createDocumentFragment()
        let cursor = 0
        for (const interval of intervals) {
          const start = Math.max(0, interval.start - offset)
          const end = Math.min(text.length, interval.end - offset)
          if (end <= start) continue
          fragment.append(text.slice(cursor, start))
          const mark = document.createElement('mark')
          mark.textContent = text.slice(start, end)
          fragment.append(mark)
          cursor = end
        }
        fragment.append(text.slice(cursor))
        node.replaceWith(fragment)
        offset += text.length
      }
    })
    pre.replaceChildren(code)
    const marked = code.querySelector<HTMLElement>('[data-code-match]')
    if (marked) pre.scrollTop = Math.max(0, marked.offsetTop - pre.offsetTop - 32)
  }
  const move = (direction: number) => {
    match.selectedRange =
      (match.selectedRange + direction + match.ranges.length) % match.ranges.length
    const range = match.ranges[match.selectedRange]
    if (range) {
      match.excerpt = source
        .split('\n')
        .slice(range.start.line - 1, range.end.line)
        .join(' ')
        .trim()
    }
    paint()
    onMatchChange?.()
  }
  previous.addEventListener('click', () => move(-1), { signal })
  next.addEventListener('click', () => move(1), { signal })
  caption.append(open)
  if (!mountedCaption || !mountedPre) {
    const inner = document.createElement('div')
    inner.className = 'preview-inner search-code-preview code-match-preview'
    const figure = document.createElement('figure')
    figure.append(caption, pre)
    inner.append(figure)
    target.replaceChildren(inner)
  }
  paint()
}
