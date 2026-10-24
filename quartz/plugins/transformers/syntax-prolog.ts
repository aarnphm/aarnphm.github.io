import type { Theme, ThemeFamily, ThemedToken } from '@pierre/highlights'
import type { RootContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import hljs from 'highlight.js/lib/core'
import prolog from 'highlight.js/lib/languages/prolog'

const highlighter = hljs.newInstance()
highlighter.registerLanguage('prolog', prolog)

const scopes: Record<string, string> = {
  comment: 'comment',
  string: 'string',
  number: 'number',
  keyword: 'keyword',
  symbol: 'string.special.symbol',
  built_in: 'function.builtin',
  title: 'function',
  variable: 'variable',
  meta: 'preproc',
}

function scopeStyle(
  family: Theme | ThemeFamily,
  scope: string,
  key: string,
): Record<string, string> {
  const theme = 'themes' in family ? family.themes[0] : family
  if (!theme) return {}
  const syntax = theme.style.syntax ?? {}
  let name = scope
  while (name && !syntax[name])
    name = name.includes('.') ? name.slice(0, name.lastIndexOf('.')) : ''
  const entry = syntax[name]
  const setting = typeof entry === 'string' ? { color: entry } : entry
  const color = setting?.color ?? theme.style['editor.foreground'] ?? theme.style.text ?? '#000000'
  const result: Record<string, string> = { [`--syntax-${key}`]: color }
  if (setting?.font_style) result[`--syntax-${key}-font-style`] = setting.font_style
  if (setting?.font_weight) result[`--syntax-${key}-font-weight`] = String(setting.font_weight)
  return result
}

/** The installed Pierre lexers do not yet include Prolog. */
export function prologTokens(
  source: string,
  themes: Record<string, Theme | ThemeFamily>,
): ThemedToken[][] {
  const tree = fromHtml(
    highlighter.highlight(source, { language: 'prolog', ignoreIllegals: true }).value,
    { fragment: true },
  )
  const lines: ThemedToken[][] = [[]]
  let offset = 0
  const append = (node: RootContent, inherited = '') => {
    if (node.type === 'element') {
      const classes = node.properties.className
      const label = Array.isArray(classes)
        ? classes.find(value => typeof value === 'string' && value.startsWith('hljs-'))
        : undefined
      const scope = typeof label === 'string' ? (scopes[label.slice(5)] ?? inherited) : inherited
      for (const child of node.children) append(child, scope)
    } else if (node.type === 'text') {
      const style: Record<string, string> = {}
      for (const [key, theme] of Object.entries(themes)) {
        Object.assign(style, scopeStyle(theme, inherited, key))
      }
      const parts = node.value.split('\n')
      parts.forEach((content, index) => {
        if (index) {
          lines.push([])
          offset++
        }
        if (content) lines[lines.length - 1].push({ content, offset, htmlStyle: style })
        offset += content.length
      })
    }
  }
  tree.children.forEach(node => append(node))
  return lines
}
