import { CODE_FILE_EXTENSIONS, canonicalCodeLanguage } from './code-languages'
import { codeQueryTokens, parseCodeQuery } from './code-query'

export const SEARCH_QUERY_PLACEHOLDER = 'search · filter: · path: · ast:'

export interface SearchCompletion {
  operator: string
  value: string
  description: string
  insert: string
  from: number
  to: number
}

const scopeDescriptions: Readonly<Record<string, string>> = {
  notes: 'notes and notebooks',
  images: 'images',
  pdf: 'PDF documents and passages',
  code: 'source files, code blocks and notebook cells',
  all: 'all searchable files',
}
const languageAliases = new Map<string, string[]>()
for (const extension of CODE_FILE_EXTENSIONS) {
  const language = canonicalCodeLanguage(extension)
  const aliases = languageAliases.get(language) ?? []
  aliases.push(extension.slice(1))
  languageAliases.set(language, aliases)
}
languageAliases.set('plain', ['text', 'plaintext'])
const languages = [
  ...new Set(['python', 'javascript', 'typescript', 'tsx', ...languageAliases.keys()]),
]

/** Complete operators before source text; ast: owns everything after its prefix. */
export function searchCompletions(raw: string, cursor: number): SearchCompletion[] {
  if (raw.trimStart().startsWith('>')) return []
  const prefix = raw.slice(0, cursor)
  const tokens = [...codeQueryTokens(prefix)]
  if (tokens.some(token => token.value.startsWith('ast:'))) return []
  const current = tokens.at(-1)
  if (!current) return []
  let from = current.index
  let value = current.value
  const previous = tokens.at(-2)
  if (previous?.value.toLowerCase() === 'filter:') {
    from = previous.index
    value = `filter:${value}`
  } else if (/\s$/.test(prefix) && value.toLowerCase() !== 'filter:') return []
  const whole = [...codeQueryTokens(raw)].find(token => token.index === current.index)
  const tokenEnd = whole ? whole.index + whole.value.length : cursor
  let to = tokenEnd
  while (raw[to] === ' ' || raw[to] === '\t') to++
  const choice = (operator: string, item: string, description: string): SearchCompletion => ({
    operator,
    value: item,
    description,
    insert: `${operator}${item}${item ? ' ' : ''}`,
    from,
    to: item ? to : tokenEnd,
  })
  const filter = /^filter:(.*)$/i.exec(value)
  if (filter) {
    const typed = filter[1].toLowerCase().replace(/^\./, '')
    if (typed && (typed in scopeDescriptions || languages.includes(typed))) return []
    return [
      ...Object.entries(scopeDescriptions)
        .filter(([scope]) => scope.startsWith(typed))
        .map(([scope, description]) => choice('filter:', scope, description)),
      ...languages
        .filter(
          language =>
            language.startsWith(typed) ||
            (typed.length > 0 && canonicalCodeLanguage(typed) === language) ||
            languageAliases.get(language)?.some(alias => alias.startsWith(typed)),
        )
        .map(language => choice('filter:', language, `${language} code`)),
    ]
  }
  if ('filter:'.startsWith(value.toLowerCase()))
    return [choice('filter:', '', 'choose notes, images, PDF, code or a language')]
  if (!parseCodeQuery(raw.slice(0, from))) return []
  if (value && 'path:'.startsWith(value) && value !== 'path:')
    return [choice('path:', '', 'limit code to a file or part of its path')]
  if (value && 'ast:'.startsWith(value))
    return [choice('ast:', '', 'match code structure with $X and $$$ARGS')]
  return []
}
