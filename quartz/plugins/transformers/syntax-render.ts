import type { Element, ElementContent } from 'hast'
import { codeToTokens, type Theme, type ThemeFamily, type ThemedToken } from '@pierre/highlights'
import { themes } from '@pierre/highlights/themes/loader'
import { h } from 'hastscript'
import { SYNTAX_TOKEN_OPTIONS, syntaxLanguage, syntaxRole } from '../../util/syntax-tokens'
import { characterRanges, type CharacterHighlight } from './syntax-meta'
import { prologTokens } from './syntax-prolog'

export type CodeTheme = keyof typeof themes | Theme | ThemeFamily
export type CodeThemes = Record<string, CodeTheme>

export async function resolveThemes(input: CodeThemes) {
  const entries = await Promise.all(
    Object.entries(input).map(
      async ([key, theme]): Promise<[string, Theme | ThemeFamily]> => [
        key,
        typeof theme === 'string' ? (await themes[theme]()).default : theme,
      ],
    ),
  )
  return Object.fromEntries(entries)
}

function tokenNode(token: ThemedToken, content: string): Element {
  const styles = token.htmlStyle ?? {}
  const role = syntaxRole(styles)
  return h(
    'span',
    {
      style: Object.entries(styles)
        .map(([key, value]) => `${key}:${value};`)
        .join(''),
      ...(role ? { 'data-syntax-role': role } : {}),
    },
    content,
  )
}

export function renderLine(
  tokens: ThemedToken[],
  highlights: CharacterHighlight[],
  grid: boolean,
): ElementContent[] {
  const source = tokens.map(token => token.content).join('')
  if (!source) return grid ? [{ type: 'text', value: ' ' }] : []
  const ranges = characterRanges(source, highlights)
  const slice = (from: number, to: number): ElementContent[] => {
    let offset = 0
    return tokens.flatMap(token => {
      const start = Math.max(from - offset, 0)
      const end = Math.min(to - offset, token.content.length)
      offset += token.content.length
      return end > start ? [tokenNode(token, token.content.slice(start, end))] : []
    })
  }
  const result: ElementContent[] = []
  let offset = 0
  for (const match of ranges) {
    result.push(...slice(offset, match.start))
    result.push(
      h(
        'mark',
        {
          'data-highlighted-chars': '',
          'data-chars-id': match.id,
          'data-highlighted-chars-mark': '',
        },
        slice(match.start, match.end),
      ),
    )
    offset = match.end
  }
  result.push(...slice(offset, source.length))
  return result
}

export function tokenize(
  source: string,
  language: string,
  theme: Record<string, Theme | ThemeFamily>,
) {
  const lang = syntaxLanguage(language)
  const result = codeToTokens(source, { lang, themes: theme, ...SYNTAX_TOKEN_OPTIONS })
  if (language.toLowerCase() === 'prolog') {
    return {
      ...result,
      tokens: prologTokens(source, theme),
      backend: 'highlight.js',
      language: 'prolog',
    }
  }
  return { ...result, backend: 'highlights', language: lang }
}
