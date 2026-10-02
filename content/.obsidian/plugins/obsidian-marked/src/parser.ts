export const INTENSITIES = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7'] as const
export type Intensity = (typeof INTENSITIES)[number]

export interface MarkerMatch {
  from: number
  to: number
  contentFrom: number
  contentTo: number
  content: string
  intensity: Intensity
  raw: string
  escaped: boolean
}

function escapedAt(text: string, index: number): boolean {
  let backslashes = 0
  while (text[--index] === '\\') backslashes++
  return backslashes % 2 === 1
}

// Keep source offsets intact so the same matches drive editor decorations and DOM ranges.
export function maskMarkdown(text: string): string {
  const mask = (value: string) => value.replace(/[^\n]/g, ' ')
  return text
    .replace(/^---\r?\n[\s\S]*?\r?\n(?:---|\.\.\.)(?=\r?\n|$)/, mask)
    .replace(
      /^ {0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:^ {0,3}\1[^\n]*(?=\n|$)|$(?![\s\S]))|^ {4}[^\n]+|\t[^\n]+|(`+)[^`]*?\2|%%[\s\S]*?%%|<!--[\s\S]*?-->|\$\$[\s\S]*?\$\$|(?<!\\)\$(?!\s)[^\n$]*?(?<!\\)\$/gm,
      mask,
    )
    .replace(/\]\([^\n)]*\)/g, mask)
}

export function findMarkers(
  text: string,
  masked = maskMarkdown(text),
  includeEscaped = false,
): MarkerMatch[] {
  const matches: MarkerMatch[] = []
  const delimiters = /(?<!:)::(?!:)([^\n]+?)::(?!:)/g
  for (const match of masked.matchAll(delimiters)) {
    const from = match.index
    if (from === undefined) continue
    const to = from + match[0].length
    const escaped = escapedAt(text, from) || escapedAt(text, to - 2)
    if (escaped && !includeEscaped) continue
    const suffix = /\{(h[1-7])\}$/.exec(text.slice(from + 2, to - 2))
    const intensity = INTENSITIES.find(level => level === suffix?.[1]) ?? 'h2'
    const contentFrom = from + 2
    const contentTo = to - 2 - (suffix?.[0].length ?? 0)
    const content = text.slice(contentFrom, contentTo)
    if (!content.trim()) continue
    matches.push({
      from,
      to,
      contentFrom,
      contentTo,
      content,
      intensity,
      escaped,
      raw: text.slice(from, to),
    })
  }
  return matches
}
