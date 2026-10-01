import { Marked, type TokenizerAndRendererExtension } from 'marked'
import { sanitizeReaderHtml } from '../arena-feed/content'

function escapeAttribute(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;')
}

// `sanitizeReaderHtml` renders `math[data-latex]` through KaTeX, so notes only need to emit that.
const blockMath: TokenizerAndRendererExtension = {
  name: 'blockMath',
  level: 'block',
  start: source => source.indexOf('$$'),
  tokenizer(source) {
    const match = /^\$\$([\s\S]+?)\$\$(?:\n|$)/.exec(source)
    if (match) return { type: 'blockMath', raw: match[0], text: match[1].trim() }
  },
  renderer: token => `<math display="block" data-latex="${escapeAttribute(token.text)}"></math>\n`,
}

const inlineMath: TokenizerAndRendererExtension = {
  name: 'inlineMath',
  level: 'inline',
  start: source => source.indexOf('$'),
  tokenizer(source) {
    const match = /^\$(?!\s)((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)/.exec(source)
    if (match) return { type: 'inlineMath', raw: match[0], text: match[1] }
  },
  renderer: token => `<math data-latex="${escapeAttribute(token.text)}"></math>`,
}

const marked = new Marked({ gfm: true, breaks: true })
marked.use({ extensions: [blockMath, inlineMath] })

const cache = new Map<string, string>()

export function renderBody(source: string): string {
  const cached = cache.get(source)
  if (cached !== undefined) return cached
  const html = sanitizeReaderHtml(marked.parse(source, { async: false }) as string)
  if (cache.size > 200) cache.clear()
  cache.set(source, html)
  return html
}
