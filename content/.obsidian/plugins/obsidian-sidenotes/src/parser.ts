import type { ParsedSidenote, SidenoteMatch, SidenoteProperties } from './types'

const OPEN = '{{sidenotes'

// Preserve offsets while excluding Markdown examples and TeX braces from delimiter searches.
function maskMarkdown(text: string): string {
  const mask = (value: string) => value.replace(/[^\n]/g, ' ')
  return text
    .replace(/^---\r?\n[\s\S]*?\r?\n(?:---|\.\.\.)(?=\r?\n|$)/, mask)
    .replace(
      /^ {0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:^ {0,3}\1[^\n]*(?=\n|$)|$(?![\s\S]))|(`+)[^`]*?\2|\$\$[\s\S]*?\$\$|(?<!\\)\$(?!\s)[^\n$]*?(?<!\\)\$/gm,
      mask,
    )
}

export function findSidenotes(text: string): SidenoteMatch[] {
  const masked = maskMarkdown(text)
  const matches: SidenoteMatch[] = []
  let index = 0
  while (index < text.length) {
    const start = masked.indexOf(OPEN, index)
    if (start === -1) break
    const parsed = parseAt(text, masked, start)
    if (parsed) {
      matches.push(parsed)
      index = parsed.to
    } else {
      index = start + OPEN.length
    }
  }
  return matches
}

function parseAt(text: string, masked: string, start: number): SidenoteMatch | null {
  let index = start + OPEN.length
  let propertiesRaw: string | undefined
  if (text[index] === '<') {
    const end = text.indexOf('>', index + 1)
    if (end === -1) return null
    propertiesRaw = text.slice(index + 1, end)
    index = end + 1
  }

  let label: string | undefined
  if (text[index] === '[') {
    const labelStart = ++index
    let depth = 1
    while (index < text.length && depth > 0) {
      if (text[index] === '[') depth++
      if (text[index] === ']') depth--
      index++
    }
    if (depth !== 0) return null
    label = text.slice(labelStart, index - 1)
  }

  if (text[index] !== ':') return null
  index++
  const contentFrom = index
  while (index < text.length && /\s/.test(text[index])) index++
  const end = masked.indexOf('}}', index)
  if (end === -1 || masked.slice(index, end).includes(OPEN)) return null
  const data: ParsedSidenote = {
    raw: text.slice(start, end + 2),
    label,
    content: text.slice(index, end),
  }
  if (propertiesRaw) data.properties = parseProperties(propertiesRaw)
  return { from: start, to: end + 2, contentFrom, data }
}

export function parseProperties(raw: string): SidenoteProperties {
  const props: SidenoteProperties = {}
  const regex = /([\w-]+)\s*:\s*((?:\[\[[^\]]+\]\]\s*,?\s*)+|[^,]+?)(?=\s*,\s*[\w-]+\s*:|$)/gs
  let match: RegExpExecArray | null
  while ((match = regex.exec(raw)) !== null) {
    const key = match[1]?.trim()
    if (!key) continue
    const value = (match[2] ?? '').trim()
    const wikilinks = value.match(/\[\[[^\]]+\]\]/g)
    props[key] = wikilinks?.length ? wikilinks : value
  }
  return props
}
