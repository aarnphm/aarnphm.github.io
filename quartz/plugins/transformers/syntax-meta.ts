export interface CharacterHighlight {
  text: string
  occurrences: Set<number>
  id?: string
  seen: number
}

function range(value: string): number[] {
  return value.split(',').flatMap(part => {
    const match = part.trim().match(/^(\d+)(?:-(\d+))?$/)
    if (!match) return []
    const start = Number(match[1])
    const end = Number(match[2] ?? match[1])
    if (end < start || end - start > 100_000) return []
    return Array.from({ length: end - start + 1 }, (_, index) => start + index)
  })
}

export function parseCodeMeta(meta: string) {
  const label = (name: string) => {
    const expression = new RegExp(`${name}="([^"]*)"`)
    const match = meta.match(expression)
    if (match) meta = meta.replace(match[0], '')
    return match?.[1]
  }
  const title = label('title')
  const caption = label('caption')
  const chars: CharacterHighlight[] = []
  meta = meta.replace(
    /(["/])(.*?)\1([^\s]*)/g,
    (_match, _delimiter: string, text: string, suffix: string) => {
      const [numbers, id] = suffix.split('#')
      if (text) chars.push({ text, occurrences: new Set(range(numbers)), id, seen: 0 })
      return ''
    },
  )
  const numbering = meta.match(/(?:^|\s)showLineNumbers(?:\{(\d+)\})?(?=\s|$)/)
  meta = meta.replace(/(?:^|\s)showLineNumbers(?:\{\d+\})?(?=\s|$)/g, '')
  const lines = new Map<number, string | undefined>()
  for (const match of meta.matchAll(/\{([\d, -]+)\}(?:#([a-zA-Z0-9]+))?/g)) {
    for (const line of range(match[1])) lines.set(line, match[2])
  }
  return {
    title,
    caption,
    chars,
    lines,
    lineStart: numbering ? Number(numbering[1] ?? 1) : undefined,
  }
}

export function characterRanges(source: string, highlights: CharacterHighlight[]) {
  const occupied = new Uint8Array(source.length)
  const matches: Array<{ start: number; end: number; id?: string }> = []
  for (const highlight of highlights) {
    let start = source.indexOf(highlight.text)
    while (start !== -1) {
      const end = start + highlight.text.length
      if (occupied.subarray(start, end).some(Boolean)) {
        start = source.indexOf(highlight.text, start + 1)
        continue
      }
      occupied.fill(1, start, end)
      highlight.seen++
      if (highlight.occurrences.size === 0 || highlight.occurrences.has(highlight.seen)) {
        matches.push({ start, end, id: highlight.id })
      }
      start = source.indexOf(highlight.text, end)
    }
  }
  return matches.sort((a, b) => a.start - b.start)
}
