import type { Element, ElementContent, Properties } from 'hast'
// @ts-ignore
import Lexer from 'pseudocode/src/Lexer.js'
// @ts-ignore
import Parser from 'pseudocode/src/Parser.js'
// @ts-ignore
import Renderer from 'pseudocode/src/Renderer.js'

/**
 * A `pseudo` fence as the note page draws it: the same pseudocode.js renderer and options the
 * Pseudocode transformer uses (quartz/plugins/transformers/pseudocode.ts), so the markup carries
 * the `.ps-root` classes components/styles/pseudocode.scss already styles. The library's own
 * math backend is KaTeX, which would join the lazy bundle; math leaves as `span.math` holding its
 * TeX instead, the same spans a passage's `$…$` becomes, and the page's Latex plugin (build) or
 * the reader's CDN KaTeX (client) typesets them. The tree is built here from the library's
 * markup string, a small subset of html (div, p, span, text), so no html parser ships.
 */

const OPTIONS = {
  indentSize: '0.6em',
  commentDelimiter: '  ▷',
  lineNumberPunc: ':',
  noEnd: false,
  scopeLines: false,
  titlePrefix: 'Algorithm',
}

const ENTITY: Record<string, string> = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'",
  '&#x27;': "'",
  '&#x2F;': '/',
}
const unescape = (text: string) => text.replace(/&(?:amp|lt|gt|quot|#39|#x27|#x2F);/gi, m => ENTITY[m] ?? m)
const escape = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/** `lineNumber=false` in the info string turns the numbers off; the site default is on. */
function lineNumbers(info: string): boolean {
  const flag = /\blineNumber=(false|true|0|1)\b/i.exec(info)?.[1]?.toLowerCase()
  return flag === undefined ? true : flag === 'true' || flag === '1'
}

const TAG = /<(\/?)([a-z][a-z0-9]*)((?:\s+[a-z-]+="[^"]*")*)\s*>|([^<]+)/gi
const ATTR = /([a-z-]+)="([^"]*)"/gi

/** The renderer's markup as hast; its tags are balanced, so a stack is enough. */
function parseMarkup(markup: string): Element | null {
  const root: Element = { type: 'element', tagName: 'root', properties: {}, children: [] }
  const stack: Element[] = [root]
  for (const m of markup.matchAll(TAG)) {
    const top = stack[stack.length - 1]
    if (m[4] !== undefined) {
      // the builder breaks a line after each block tag; spaces inside a line are content
      if (m[4] !== '\n') top.children.push({ type: 'text', value: unescape(m[4]) })
    } else if (m[1]) {
      if (stack.length > 1) stack.pop()
    } else {
      const properties: Properties = {}
      for (const a of (m[3] ?? '').matchAll(ATTR)) {
        const value = unescape(a[2])
        if (a[1] === 'class') properties.className = value.split(/\s+/).filter(Boolean)
        else properties[a[1]] = value
      }
      const el: Element = { type: 'element', tagName: m[2].toLowerCase(), properties, children: [] }
      top.children.push(el as ElementContent)
      stack.push(el)
    }
  }
  const first = root.children[0]
  return first && first.type === 'element' ? first : null
}

/** From `\begin{algorithm}` on: what sits before it is a preamble of macros a passage cannot use. */
function algorithmSource(body: string): string {
  const start = body.indexOf('\\begin{algorithm}')
  return (start > 0 ? body.slice(start) : body).replace(/\n+$/, '')
}

/** The algorithm as a `.ps-root` element, or null when the library cannot parse it. */
export function pseudocodeBlock(body: string, info = ''): Element | null {
  try {
    const renderer = new Renderer(
      new Parser(new Lexer(algorithmSource(body))),
      { ...OPTIONS, lineNumber: lineNumbers(info), captionCount: 0 },
    )
    renderer.backend = {
      name: 'katex',
      driver: { renderToString: (tex: string) => `<span class="math math-inline">${escape(tex)}</span>` },
    }
    const root = parseMarkup(renderer.toMarkup())
    if (!root) return null
    root.properties = { ...root.properties, dataLatPseudo: '' }
    return root
  } catch {
    return null
  }
}
