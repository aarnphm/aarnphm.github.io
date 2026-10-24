import type { Element, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { visit, SKIP } from 'unist-util-visit'
import { parseCodeMeta } from './syntax-meta'
import { renderLine, resolveThemes, tokenize, type CodeThemes } from './syntax-render'

export interface HighlightsOptions {
  theme?: CodeThemes
  keepBackground?: boolean
  grid?: boolean
  bypassInlineCode?: boolean
  defaultLang?: string | { block?: string; inline?: string }
  filterMetaString?: (meta: string) => string
  onVisitLine?: (element: Element) => void
  onVisitHighlightedLine?: (element: Element, id?: string) => void
  onVisitHighlightedChars?: (element: Element, id?: string) => void
  onVisitTitle?: (element: Element) => void
  onVisitCaption?: (element: Element) => void
}

/** Retains the markup consumed by notebooks, embeds, and the code-block stylesheet. */
export function rehypeHighlights(options: HighlightsOptions = {}) {
  const {
    theme = { light: 'github-light', dark: 'github-dark' },
    keepBackground = false,
    grid = true,
  } = options
  const resolved = resolveThemes(theme)
  const themeNames = Object.values(theme)
    .map(value => (typeof value === 'string' ? value : (value.name ?? '')))
    .join(' ')
  const defaults =
    typeof options.defaultLang === 'string'
      ? { block: options.defaultLang, inline: options.defaultLang }
      : options.defaultLang
  return async (tree: Root) => {
    const themes = await resolved
    visit(tree, 'element', (element, _index, parent) => {
      const block =
        element.tagName === 'pre' &&
        element.children.length === 1 &&
        element.children[0].type === 'element' &&
        element.children[0].tagName === 'code'
      const inline =
        !options.bypassInlineCode &&
        element.tagName === 'code' &&
        parent?.type === 'element' &&
        parent.tagName !== 'pre'
      if (!block && !inline) return
      const code = block ? element.children[0] : element
      if (code.type !== 'element' || code.children.length !== 1 || code.children[0].type !== 'text')
        return
      let source = toString(code)
      let language = defaults?.block ?? ''
      let meta = ''
      if (inline) {
        const suffix = source.match(/(\\)?\{:([a-zA-Z.-]+)\}$/)
        if (suffix?.[1]) {
          code.children[0].value = source.replace(/\\(\{:[a-zA-Z.-]+\})$/, '$1')
          return SKIP
        }
        language = suffix?.[2] ?? defaults?.inline ?? ''
        if (suffix) source = source.slice(0, suffix.index)
      } else {
        const classes = code.properties.className
        if (Array.isArray(classes)) {
          const label = classes.find(
            value => typeof value === 'string' && value.startsWith('language-'),
          )
          if (typeof label === 'string') language = label.slice('language-'.length)
        }
        const rawMeta = code.data?.meta ?? code.properties.metastring
        meta = typeof rawMeta === 'string' ? rawMeta : ''
        meta = options.filterMetaString?.(meta) ?? meta
        source = source.replace(/\n$/, '')
      }
      if (!language || language === 'math') return SKIP
      const parsed = parseCodeMeta(meta)
      const highlighted = tokenize(source, language, themes)
      const properties = { 'data-language': language, 'data-theme': themeNames }
      const lines = highlighted.tokens.map((tokens, index) => {
        const line = h(
          'span',
          { 'data-line': '' },
          renderLine(tokens, parsed.chars, !inline && grid),
        )
        options.onVisitLine?.(line)
        if (parsed.lines.has(index + 1)) {
          const id = parsed.lines.get(index + 1)
          line.properties['data-highlighted-line'] = ''
          if (id) line.properties['data-highlighted-line-id'] = id
          options.onVisitHighlightedLine?.(line, id)
        }
        visit(line, 'element', node => {
          if (node.tagName !== 'mark') return
          const id = node.properties['data-chars-id']
          options.onVisitHighlightedChars?.(node, typeof id === 'string' ? id : undefined)
        })
        return line
      })
      const rendered = h(
        'code',
        properties,
        lines.flatMap((line, index) => (index ? [{ type: 'text', value: '\n' }, line] : [line])),
      )
      rendered.data = code.data
      element.tagName = inline ? 'span' : 'figure'
      element.properties['data-rehype-pretty-code-figure'] = ''
      element.properties['data-highlighter'] = highlighted.backend
      element.properties['data-highlight-language'] = highlighted.language
      if (inline) {
        if (keepBackground) rendered.properties.style = highlighted.rootStyle
        element.children = [rendered]
        return SKIP
      }
      let style = grid ? 'display: grid;' : ''
      if (parsed.lineStart !== undefined) {
        rendered.properties['data-line-numbers'] = ''
        rendered.properties['data-line-numbers-max-digits'] = String(
          parsed.lineStart - 1 + lines.length,
        ).length
        style = `counter-set: line ${parsed.lineStart - 1};${style}`
      }
      if (style) rendered.properties.style = style
      const pre = h(
        'pre',
        { ...properties, tabindex: 0, ...(keepBackground ? { style: highlighted.rootStyle } : {}) },
        [rendered],
      )
      // The original code node carries this flag through remark-rehype.
      if (
        code.properties['data-disable-line-number'] !== undefined ||
        code.properties.dataDisableLineNumber !== undefined
      ) {
        rendered.properties['data-disable-line-number'] = 'true'
      }
      element.children = []
      if (parsed.title) {
        const title = h(
          parsed.caption ? 'div' : 'figcaption',
          { ...properties, 'data-rehype-pretty-code-title': '' },
          parsed.title,
        )
        options.onVisitTitle?.(title)
        element.children.push(title)
      }
      element.children.push(pre)
      if (parsed.caption) {
        const caption = h(
          'figcaption',
          { ...properties, 'data-rehype-pretty-code-caption': '' },
          parsed.caption,
        )
        options.onVisitCaption?.(caption)
        element.children.push(caption)
      }
      return SKIP
    })
  }
}
