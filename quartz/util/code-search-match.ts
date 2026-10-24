import type { CodeMatch, CodePoint, CodeQuery, CodeRange, CodeUnit } from './code-search-types'

function sourcePoints(text: string): (offset: number) => CodePoint {
  const starts = [0]
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n' || (text[i] === '\r' && text[i + 1] !== '\n')) starts.push(i + 1)
  }
  return offset => {
    let low = 0
    let high = starts.length
    while (low + 1 < high) {
      const middle = (low + high) >>> 1
      if (starts[middle] <= offset) low = middle
      else high = middle
    }
    return { offset, line: low + 1, column: offset - starts[low] + 1 }
  }
}

export function literalRanges(text: string, pattern: string, caseSensitive: boolean): CodeRange[] {
  if (!pattern) return []
  const ranges: CodeRange[] = []
  let point: ReturnType<typeof sourcePoints> | undefined
  // Regex case folding preserves original offsets even when lowercasing expands a character.
  const insensitive = caseSensitive
    ? undefined
    : new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'giu')
  let from = 0
  while (from <= text.length) {
    if (insensitive) insensitive.lastIndex = from
    const found = insensitive?.exec(text)
    const start = insensitive ? (found?.index ?? -1) : text.indexOf(pattern, from)
    if (start < 0) break
    point ??= sourcePoints(text)
    const end = start + (found?.[0].length ?? pattern.length)
    ranges.push({ start: point(start), end: point(end) })
    const codepoint = text.codePointAt(start)
    from = start + (codepoint !== undefined && codepoint > 0xffff ? 2 : 1)
  }
  return ranges
}

export function codeUnitSelected(unit: CodeUnit, query: CodeQuery): boolean {
  return (
    (query.language ? unit.language === query.language : unit.language !== 'json') &&
    (!query.path || unit.path.toLowerCase().includes(query.path.toLowerCase()))
  )
}

export function codeMatch(
  unit: CodeUnit,
  text: string,
  ranges: CodeRange[],
  query: CodeQuery,
): CodeMatch {
  const line = ranges[0]?.start.line ?? 1
  const lines = text.split(/\r\n|\r|\n/)
  const first = Math.max(0, line - 2)
  const last = Math.max(line + 1, ranges[0]?.end.line ?? line)
  const term = query.pattern.toLowerCase()
  const score =
    (term && unit.title.toLowerCase().includes(term) ? 20 : 0) +
    (term && unit.path.toLowerCase().includes(term) ? 10 : 0) +
    Math.min(ranges.length, 9)
  return {
    unit,
    ranges,
    excerpt: lines.slice(first, last).join('\n'),
    excerptStartLine: unit.startLine + first,
    score,
  }
}

export function rankCodeMatches(matches: CodeMatch[]): CodeMatch[] {
  const owners = new Map<string, CodeMatch[]>()
  for (const match of matches) {
    const group = owners.get(match.unit.ownerSlug) ?? []
    group.push(match)
    owners.set(match.unit.ownerSlug, group)
  }
  const compare = (a: CodeMatch, b: CodeMatch) =>
    b.score - a.score ||
    a.unit.path.localeCompare(b.unit.path) ||
    a.unit.startLine - b.unit.startLine ||
    a.unit.id.localeCompare(b.unit.id)
  const groups = [...owners.values()]
    .map(group => group.sort(compare))
    .sort((a, b) => compare(a[0], b[0]))
  // First page exposes each owner before repeating blocks from the same note.
  const result: CodeMatch[] = []
  for (let i = 0; groups.some(group => i < group.length); i++) {
    for (const group of groups) if (group[i]) result.push(group[i])
  }
  return result
}
