import type { CodeQuery } from './code-search-types'
import { CODE_FILE_EXTENSIONS, canonicalCodeLanguage } from './code-languages'

const languages = new Set(
  [...CODE_FILE_EXTENSIONS].map(extension => canonicalCodeLanguage(extension)),
)
languages.add('plain')

function unquote(value: string): string {
  const quote = value[0]
  if (quote !== '"' && quote !== "'") return value
  for (let i = 1; i < value.length; i++) {
    if (value[i] === '\\') i++
    else if (value[i] === quote) return i === value.length - 1 ? value.slice(1, -1) : value
  }
  return value
}

export function* codeQueryTokens(raw: string) {
  let position = 0
  while (position < raw.length) {
    while (/\s/.test(raw[position] ?? '') && position < raw.length) position++
    const index = position
    let quote: string | undefined
    while (position < raw.length) {
      const character = raw[position]
      if (quote) {
        if (character === '\\') position++
        else if (character === quote) quote = undefined
      } else if (character === '"' || character === "'" || character === '`') quote = character
      else if (/\s/.test(character)) break
      position++
    }
    if (position > index) yield { value: raw.slice(index, position), index }
  }
}

/** Operators are tokens outside code strings; ast: owns the untouched remainder. */
export function parseCodeQuery(raw: string): CodeQuery | null {
  const tokens = [...codeQueryTokens(raw)]
  const removed: [number, number][] = []
  let selected = false
  let language: string | undefined
  let path: string | undefined
  let structural: number | undefined
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]
    const value = token.value
    if (value.startsWith('ast:')) {
      structural = token.index + 4
      break
    }
    const filter = /^filter:(.*)$/i.exec(value)
    if (filter) {
      const argument = filter[1] || tokens[i + 1]?.value || ''
      const name = argument.toLowerCase().replace(/^\./, '')
      const candidate = canonicalCodeLanguage(name)
      const canonical = languages.has(candidate) ? candidate : undefined
      if (name === 'code' || canonical) {
        selected = true
        language = name === 'code' ? undefined : canonical
        if (!filter[1]) i++
        const end = tokens[i]
        removed.push([token.index, end.index + end.value.length])
      }
    } else if (value.startsWith('path:')) {
      path = unquote(value.slice(5))
      removed.push([token.index, token.index + value.length])
    }
  }
  if (!selected) return null
  let pattern: string
  if (structural !== undefined) pattern = raw.slice(structural)
  else {
    let end = 0
    pattern = ''
    for (const [start, stop] of removed) {
      pattern += raw.slice(end, start)
      end = stop
      // Horizontal separators belong to operators. Newlines belong to source text.
      while (raw[end] === ' ' || raw[end] === '\t') end++
    }
    pattern += raw.slice(end)
    pattern = unquote(/[\r\n]/.test(pattern) ? pattern : pattern.replace(/^[ \t]+|[ \t]+$/g, ''))
  }
  return {
    mode: structural === undefined ? 'literal' : 'structural',
    pattern,
    ...(language ? { language } : {}),
    ...(path ? { path } : {}),
    caseSensitive: true,
  }
}
