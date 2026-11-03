import type { Element, ElementContent } from 'hast'
import { s } from 'hastscript'
import { decodeEntities } from './html'

/**
 * Inline svg that a note writes into its markdown (the nltk parse trees in the course notes,
 * drawn as nested `<svg>`s of `<text>` and `<line>`), rebuilt as hast for the reader. Only
 * drawing elements and their geometry, text and paint survive: no scripts, handlers, links,
 * foreign objects, ids or `url(…)` references. An element outside the allowlist drops with
 * everything inside it. A drawing cut off at the passage end closes where it stops.
 */

const ELEMENTS = new Set([
  'svg',
  'g',
  'defs',
  'line',
  'text',
  'tspan',
  'path',
  'rect',
  'circle',
  'ellipse',
  'polyline',
  'polygon',
  'title',
  'desc',
])
// elements whose text is drawn (or read out); text anywhere else is whitespace between tags
const TEXT = new Set(['text', 'tspan', 'title', 'desc'])
const ATTRIBUTES = new Set([
  'x',
  'y',
  'x1',
  'x2',
  'y1',
  'y2',
  'dx',
  'dy',
  'cx',
  'cy',
  'r',
  'rx',
  'ry',
  'width',
  'height',
  'viewbox',
  'preserveaspectratio',
  'baseprofile',
  'version',
  'd',
  'points',
  'transform',
  'style',
  'class',
  'fill',
  'fill-opacity',
  'fill-rule',
  'stroke',
  'stroke-width',
  'stroke-dasharray',
  'stroke-linecap',
  'stroke-linejoin',
  'stroke-opacity',
  'opacity',
  'text-anchor',
  'dominant-baseline',
  'alignment-baseline',
  'font-family',
  'font-size',
  'font-weight',
  'font-style',
  'textlength',
  'lengthadjust',
])
const UNSAFE = /url\s*\(|javascript:|expression\s*\(|@import|[<>]/i

const TOKEN =
  /<!--[\s\S]*?(?:-->|$)|<(\/?)([A-Za-z][\w:-]*)((?:\s+[^\s=/>]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*)\s*(\/?)>/g
const ATTRIBUTE = /([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g

function attributes(source: string): Record<string, string> {
  const out: Record<string, string> = {}
  for (const m of source.matchAll(ATTRIBUTE)) {
    const name = m[1].toLowerCase()
    const value = decodeEntities(m[2] ?? m[3] ?? m[4] ?? '')
    if (ATTRIBUTES.has(name) && !UNSAFE.test(value)) out[name] = value
  }
  return out
}

/** The `<svg>` drawings in an html source, sanitised; everything outside them is ignored. */
export function svgDrawings(source: string): Element[] {
  const out: Element[] = []
  const stack: Element[] = []
  // names of disallowed elements still open inside a drawing: their content drops
  const skipping: string[] = []
  let last = 0
  const text = (value: string) => {
    const top = stack[stack.length - 1]
    if (!top || skipping.length || !TEXT.has(top.tagName) || !value) return
    top.children.push({ type: 'text', value: decodeEntities(value) })
  }
  for (const m of source.matchAll(TOKEN)) {
    text(source.slice(last, m.index))
    last = m.index + m[0].length
    if (m[2] === undefined) continue // comment
    const close = m[1] === '/'
    const name = m[2].toLowerCase()
    const selfClosing = m[4] === '/'
    if (skipping.length) {
      if (close && name === skipping[skipping.length - 1]) skipping.pop()
      else if (!close && !selfClosing && name === skipping[skipping.length - 1]) skipping.push(name)
      continue
    }
    if (close) {
      for (let at = stack.length - 1; at >= 0; at--)
        if (stack[at].tagName.toLowerCase() === name) {
          stack.length = at
          break
        }
      continue
    }
    // nothing outside the first `<svg>` (a figure wrapper, a caption) belongs to the drawing
    if (!stack.length && name !== 'svg') continue
    if (!ELEMENTS.has(name)) {
      if (!selfClosing) skipping.push(name)
      continue
    }
    const el = s(name, attributes(m[3] ?? ''))
    if (stack.length) stack[stack.length - 1].children.push(el as ElementContent)
    else out.push(el)
    if (!selfClosing) stack.push(el)
  }
  text(source.slice(last))
  return out
}
