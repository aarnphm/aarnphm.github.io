import { codeToTokens, type Lang } from '@pierre/highlights'
import githubDark from '@pierre/highlights/themes/github-dark'
import githubLight from '@pierre/highlights/themes/github-light'
import { languageFromExt } from './code-languages'
import { escapeHTML } from './escape'
import { SYNTAX_TOKEN_OPTIONS, syntaxLanguage, syntaxRole } from './syntax-tokens'

// Lazy callers resolve languages through this module so the Wasm loads once, on demand.
export { syntaxLanguage }

/** The data-theme value syntax.scss needs to switch token colours with the site theme. */
export const CODE_THEME = 'github-light github-dark'

const themes = { light: githubLight, dark: githubDark }
// Past this length one span per token stalls the UI thread, so the code stays plain lines.
const MAX_HIGHLIGHTED_LENGTH = 200_000

export type HighlightedCode = { html: string; language: Lang; lines: number }

/**
 * The inner HTML of a code element, in the line markup that rehype-highlights emits.
 * Each line of `source` becomes one row, so a trailing newline adds an empty last row.
 */
export function highlightCode(source: string, label: string): HighlightedCode {
  const language = source.length <= MAX_HIGHLIGHTED_LENGTH ? syntaxLanguage(label) : 'plain'
  const { tokens } = codeToTokens(source, {
    lang: language,
    themes,
    tokenizeMaxLineLength: 2_000,
    ...SYNTAX_TOKEN_OPTIONS,
  })
  // Equal runs share one htmlStyle object.
  const attributes = new Map<Record<string, string>, string>()
  const html = tokens
    .map(line => {
      const spans = line
        .map(token => {
          const styles = token.htmlStyle ?? {}
          let attrs = attributes.get(styles)
          if (attrs === undefined) {
            const style = Object.entries(styles)
              .map(([key, value]) => `${key}:${value}`)
              .join(';')
            const role = syntaxRole(styles)
            attrs = ` style="${escapeHTML(style)}"${role ? ` data-syntax-role="${role}"` : ''}`
            attributes.set(styles, attrs)
          }
          return `<span${attrs}>${escapeHTML(token.content)}</span>`
        })
        .join('')
      // An empty grid row collapses, so blank lines keep one space.
      return `<span data-line>${spans || ' '}</span>`
    })
    .join('\n')
  return { html, language, lines: tokens.length }
}

/** Highlights a file by its name, without the empty row after its final newline. */
export function highlightSourceCode(source: string, fileName: string): HighlightedCode {
  const name = fileName.split('/').at(-1)?.toLowerCase() ?? ''
  const extension = name.lastIndexOf('.') > 0 ? name.slice(name.lastIndexOf('.') + 1) : ''
  const label = extension ? (languageFromExt(extension) ?? extension) : name
  return highlightCode(source.replace(/\n$/, ''), label)
}

/** Attributes the code element needs for theme colours and line-number width. */
export function codeAttributes({ language, lines }: HighlightedCode) {
  return {
    'data-language': language,
    'data-theme': CODE_THEME,
    style: `--line-digits:${String(lines).length}`,
  }
}
