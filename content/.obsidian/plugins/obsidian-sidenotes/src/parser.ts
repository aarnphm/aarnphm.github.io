import type { ParsedSidenote, Segment, SidenoteProperties } from "./types"

const KEYWORD = "sidenotes"
const OPEN = "{{"
const CLOSE = "}}"

interface ParseResult {
  node: ParsedSidenote
  start: number
  end: number
}

export function extractSegments(text: string): Segment[] {
  const segments: Segment[] = []
  let index = 0

  while (index < text.length) {
    const start = text.indexOf(OPEN, index)

    if (start === -1) {
      segments.push({ type: "text", value: text.slice(index) })
      break
    }

    const parsed = parseAt(text, start)

    if (!parsed) {
      segments.push({ type: "text", value: text.slice(index, start + OPEN.length) })
      index = start + OPEN.length
      continue
    }

    if (parsed.start > index) {
      segments.push({ type: "text", value: text.slice(index, parsed.start) })
    }

    segments.push({ type: "sidenote", data: parsed.node })
    index = parsed.end
  }

  return segments
}

function parseAt(text: string, start: number): ParseResult | null {
  let index = start

  if (!text.startsWith(OPEN, index)) return null
  index += OPEN.length

  if (!text.startsWith(KEYWORD, index)) return null
  index += KEYWORD.length

  let propertiesRaw: string | undefined

  if (text[index] === "<") {
    const propsEnd = text.indexOf(">", index + 1)
    if (propsEnd === -1) return null
    propertiesRaw = text.slice(index + 1, propsEnd)
    index = propsEnd + 1
  }

  let label: string | undefined
  if (text[index] === "[") {
    const labelEnd = text.indexOf("]", index + 1)
    if (labelEnd === -1) return null
    label = text.slice(index + 1, labelEnd)
    index = labelEnd + 1
  }

  if (text[index] !== ":") return null
  index += 1
  while (text[index] === " " || text[index] === "\n" || text[index] === "\t") {
    index += 1
  }

  const contentEnd = text.indexOf(CLOSE, index)
  if (contentEnd === -1) return null

  const raw = text.slice(start, contentEnd + CLOSE.length)
  const node: ParsedSidenote = {
    raw,
    label,
    content: text.slice(index, contentEnd),
  }

  if (propertiesRaw && propertiesRaw.trim().length > 0) {
    node.properties = parseProperties(propertiesRaw)
  }

  return { node, start, end: contentEnd + CLOSE.length }
}

export function parseProperties(raw: string): SidenoteProperties {
  const props: Record<string, string | string[]> = {}

  const regex = /([\w-]+)\s*:\s*((?:\[\[[^\]]+\]\]\s*,?\s*)+|[^,]+?)(?=\s*,\s*[\w-]+\s*:|$)/gs
  let match: RegExpExecArray | null

  while ((match = regex.exec(raw)) !== null) {
    const key = match[1]?.trim()
    if (!key) continue

    const value = (match[2] ?? "").trim()

    if (value.includes("[[")) {
      const wikilinks = value.match(/\[\[[^\]]+\]\]/g) || []
      props[key] = wikilinks.length > 0 ? wikilinks : value
    } else {
      props[key] = value
    }
  }

  return props
}
