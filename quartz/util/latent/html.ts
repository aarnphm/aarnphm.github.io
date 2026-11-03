import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'

/**
 * Passage html, sanitised. OCR'd pdf passages carry html tables (`<table border="1">`,
 * colspan/rowspan, often cut across two passages) and a few inline tags; notes carry raw html
 * from their markdown, pipe tables and code. Only an allowlist becomes markup:
 *
 * - table structure (table, caption, thead, tbody, tfoot, tr, td, th) with a clamped
 *   colspan/rowspan, the only attributes that survive;
 * - inline tags (sup, sub, u, i, em, b, strong, code, s, del, small, mark, kbd) and br;
 * - wrappers (div, span, font, a, p, center, figure, …) drop to their content, images drop
 *   (their src is relative to a page that is not this one);
 * - anything else stays literal text, so `<class 'int'>` or `a <b` reads as written.
 *
 * Text between tags goes through `leaf`, which the caller uses for entities and math. Fenced
 * code and (for notes) backtick spans stay literal, so html inside code is never parsed.
 */

export type Leaf = (text: string) => ElementContent[]

const NAMED: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: '\u00a0',
}
const ENTITY = /&(?:#(\d{1,7})|#x([0-9a-f]{1,6})|(amp|lt|gt|quot|apos|nbsp));/gi

function decodeOnce(text: string): string {
  return text.replace(ENTITY, (whole, dec?: string, hex?: string, name?: string) => {
    if (name) return NAMED[name.toLowerCase()]
    const code = dec ? Number(dec) : parseInt(hex!, 16)
    return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole
  })
}

/**
 * The embedding text keeps html-escaped entities from the source, some escaped twice
 * (`&amp;#039;`); hast escapes again on output, so decode named and numeric entities, twice
 * when the first pass leaves one behind.
 */
export function decodeEntities(text: string): string {
  if (!text.includes('&')) return text
  const once = decodeOnce(text)
  return once !== text && /&(?:#\d|#x[0-9a-f]|amp;|lt;|gt;|quot;|apos;)/i.test(once)
    ? decodeOnce(once)
    : once
}

const TABLE = new Set(['table', 'caption', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th'])
const INLINE = new Set([
  'sup',
  'sub',
  'u',
  'i',
  'em',
  'b',
  'strong',
  'code',
  's',
  'del',
  'small',
  'mark',
  'kbd',
])
const UNWRAP = new Set([
  'div',
  'span',
  'font',
  'a',
  'p',
  'center',
  'figure',
  'figcaption',
  'section',
  'blockquote',
])
const DROP = new Set(['img', 'wbr', 'hr', 'col', 'colgroup'])
const CELL = new Set(['td', 'th'])
const ROWS = new Set(['thead', 'tbody', 'tfoot'])

// lowercase names only: `a <B` and `<Figure 3>` in prose are text, not tags
const ATTRS = String.raw`(?:\s+[a-zA-Z_:][-\w:.]*(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>\x60]+))?)*`
const TAG = String.raw`<(\/?)([a-z][a-z0-9]*)(${ATTRS})\s*\/?>`
const CODE_SPAN = '`([^`\\n]+)`'
const TOKENS = new RegExp(TAG, 'g')
const TOKENS_CODE = new RegExp(`${CODE_SPAN}|${TAG}`, 'g')
const CODE_SPANS = new RegExp(CODE_SPAN, 'g')

const known = (name: string) =>
  TABLE.has(name) || INLINE.has(name) || UNWRAP.has(name) || DROP.has(name) || name === 'br'

function spanOf(attrs: string, key: 'colspan' | 'rowspan'): number | undefined {
  const match = new RegExp(String.raw`\b${key}\s*=\s*["']?(\d{1,4})`, 'i').exec(attrs)
  const n = match ? Number(match[1]) : 1
  return n > 1 ? Math.min(n, 64) : undefined
}

export interface HtmlOptions {
  /** backtick spans become literal `code` (notes; pdfs use `` ``quotes'' `` from LaTeX) */
  code?: boolean
}

/**
 * Build `source` into `root` with the allowlist. With a `table` root the stack also repairs what
 * the OCR and the chunker leave out: a cell or row opened implicitly for stray text, unclosed
 * cells and rows closed by the next one, a table cut off at the passage end closed at the end.
 */
function build(root: Element, source: string, leaf: Leaf, options: HtmlOptions): Element {
  const inTable = root.tagName === 'table'
  const stack: Element[] = [root]
  const top = () => stack[stack.length - 1]
  const open = (el: Element) => {
    top().children.push(el)
    stack.push(el)
  }
  const inCell = () => stack.some(el => CELL.has(el.tagName) || el.tagName === 'caption')
  // pop to the nearest element `match` accepts, never past one `stop` accepts
  const closeTo = (match: (tag: string) => boolean, stop: (tag: string) => boolean) => {
    for (let i = stack.length - 1; i > 0; i--) {
      if (match(stack[i].tagName)) {
        stack.length = i
        return
      }
      if (stop(stack[i].tagName)) return
    }
  }
  const popWhile = (pred: (tag: string) => boolean) => {
    while (stack.length > 1 && pred(top().tagName)) stack.pop()
  }
  // text or inline markup inside a table but outside any cell: open the row and cell it implies
  const ensureCell = () => {
    if (!inTable || inCell()) return
    if (top().tagName === 'table' || ROWS.has(top().tagName)) open(h('tr'))
    if (top().tagName === 'tr') open(h('td'))
  }
  const text = (value: string) => {
    if (!value) return
    if (inTable && !inCell() && !value.trim()) return
    ensureCell()
    top().children.push(...leaf(value))
  }

  const tokens = options.code ? TOKENS_CODE : TOKENS
  const offset = options.code ? 1 : 0
  let last = 0
  for (const match of source.matchAll(tokens)) {
    text(source.slice(last, match.index))
    last = match.index + match[0].length
    if (options.code && match[1] !== undefined) {
      ensureCell()
      top().children.push(h('code', decodeEntities(match[1])))
      continue
    }
    const close = match[1 + offset] === '/'
    const name = match[2 + offset]
    const attrs = match[3 + offset] ?? ''
    if (!known(name) || (TABLE.has(name) && !inTable)) {
      text(match[0])
    } else if (DROP.has(name) || UNWRAP.has(name)) {
      continue
    } else if (name === 'br') {
      if (close) continue
      ensureCell()
      top().children.push(h('br'))
    } else if (INLINE.has(name)) {
      if (close)
        closeTo(
          tag => tag === name,
          tag => CELL.has(tag) || TABLE.has(tag),
        )
      else {
        ensureCell()
        open(h(name))
      }
    } else if (close) {
      // table structure: a close with nothing to close is dropped
      if (name === 'table')
        closeTo(
          tag => tag === 'table',
          () => false,
        )
      else
        closeTo(
          tag => tag === name,
          tag => tag === 'table',
        )
    } else if (name === 'table') {
      ensureCell()
      open(h('table'))
    } else if (name === 'caption') {
      popWhile(tag => tag !== 'table')
      open(h('caption'))
    } else if (ROWS.has(name)) {
      popWhile(tag => tag !== 'table')
      open(h(name))
    } else if (name === 'tr') {
      popWhile(tag => tag !== 'table' && !ROWS.has(tag))
      open(h('tr'))
    } else {
      popWhile(tag => tag !== 'tr' && tag !== 'table' && !ROWS.has(tag))
      if (top().tagName !== 'tr') open(h('tr'))
      open(h(name, { colSpan: spanOf(attrs, 'colspan'), rowSpan: spanOf(attrs, 'rowspan') }))
    }
  }
  text(source.slice(last))
  return root
}

/** Inline runs of a paragraph or a cell, html allowlisted. */
export function htmlRuns(source: string, leaf: Leaf, options: HtmlOptions = {}): ElementContent[] {
  return build(h('span'), source, leaf, options).children
}

function rowsOf(table: Element): Element[] {
  const rows: Element[] = []
  for (const child of table.children) {
    if (child.type !== 'element') continue
    if (child.tagName === 'tr') rows.push(child)
    else if (ROWS.has(child.tagName))
      for (const row of child.children)
        if (row.type === 'element' && row.tagName === 'tr') rows.push(row)
  }
  return rows
}

const container = (table: Element) => h('div.table-container.lat-table', [table])

/**
 * An html table block. OCR tables have no `th`; their first row is the header, so it is promoted
 * unless the passage starts mid-table (`continued`), where the first row is a body row.
 */
export function htmlTable(source: string, leaf: Leaf, continued: boolean): Element {
  const table = build(h('table'), source.replace(/^\s*<table\b[^>]*>/, ''), leaf, {})
  // a row the cut left empty
  for (const group of [table, ...table.children])
    if (group.type === 'element')
      group.children = group.children.filter(
        row => row.type !== 'element' || row.tagName !== 'tr' || row.children.length > 0,
      )
  const rows = rowsOf(table)
  const hasHead = rows.some(row =>
    row.children.some(cell => cell.type === 'element' && cell.tagName === 'th'),
  )
  if (!continued && !hasHead && rows.length > 1)
    for (const cell of rows[0].children) if (cell.type === 'element') cell.tagName = 'th'
  if (continued) table.properties = { ...table.properties, dataContinued: '' }
  return container(table)
}

const NAMES = [...TABLE, ...INLINE, ...UNWRAP, ...DROP, 'br']
const HEAD_CUT =
  /^\s*(\/?[a-z0-9]*(?=[\s>]))?((?:\s*[a-z-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))*)\s*>/
const TAIL_CUT = /<(\/?)([a-z][a-z0-9]*)?(?:\s[^<>]*)?$/

/**
 * The chunker cuts passages by length, sometimes inside a tag: one passage ends `<td` or
 * `<table`, the next starts `rowspan="2">` or `border="1">`. Drop a tag cut off at the end, and
 * rebuild the tag a passage starts inside of (a cell, a row, or the table itself).
 */
export function repairCuts(text: string): string {
  let out = text
  const tail = TAIL_CUT.exec(out)
  if (tail && (tail[2] ? NAMES.some(n => n.startsWith(tail[2])) : /<\/?t[rdh]\b/.test(out)))
    out = out.slice(0, tail.index)
  const head = HEAD_CUT.exec(out)
  if (head && /<\/?t[rdh]\b|<\/table>/.test(out)) {
    const name = head[1] ?? ''
    const attrs = head[2].trim()
    const rest = out.slice(head[0].length)
    if (name.startsWith('/')) out = rest
    else if (!name && /\bborder\s*=/.test(attrs)) out = `<table>${rest}`
    else if ('tr'.endsWith(name) && name) out = `<tr>${rest}`
    else if ('td'.endsWith(name) || 'th'.endsWith(name))
      out = `<${name.endsWith('h') ? 'th' : 'td'}${attrs ? ` ${attrs}` : ''}>${rest}`
  }
  return out
}

export interface Span {
  start: number
  end: number
  kind: 'table' | 'fence'
  /** a table whose opening tag is in the previous passage */
  continued?: boolean
  /** a fence's body */
  body?: string
  /** a fence's info string: `python`, `quotes`, `pseudo lineNumber=false` */
  info?: string
}

const FENCE = /^[ \t]*(`{3,}|~{3,})([^\n]*)\n?/gm
const TABLE_EDGE = /<(\/?)table\b[^>]*>/g

/**
 * Fenced code and html tables in a passage, in order. A fence missing its close runs to the end;
 * so does a table (the chunker cut it). A `</table>` before any `<table` means the passage
 * opens inside a table, which then starts at the passage start.
 */
export function blockSpans(source: string, options: HtmlOptions = {}): Span[] {
  const spans: Span[] = []
  const fences: Span[] = []
  FENCE.lastIndex = 0
  for (let m = FENCE.exec(source); m; m = FENCE.exec(source)) {
    const marker = m[1]
    const bodyStart = m.index + m[0].length
    const closer = new RegExp(String.raw`^[ \t]*${marker[0]}{${marker.length},}[ \t]*$`, 'm')
    const rest = source.slice(bodyStart)
    const end = closer.exec(rest)
    // OCR leaves fences open; in a pdf an open one ends where an html table starts
    const table = end || options.code ? -1 : rest.search(/^[ \t]*<table\b/m)
    const open = table >= 0 ? bodyStart + table : source.length
    const stop = end ? bodyStart + end.index : open
    const after = end ? stop + end[0].length : open
    fences.push({
      start: m.index,
      end: after,
      kind: 'fence',
      body: source.slice(bodyStart, stop),
      info: m[2].trim(),
    })
    FENCE.lastIndex = after
  }
  // tables in the stretches between fences; backtick spans masked so `<table>` in code is code
  let cursor = 0
  const scan = (from: number, to: number) => {
    let segment = source.slice(from, to)
    if (options.code) segment = segment.replace(CODE_SPANS, m => ' '.repeat(m.length))
    let depth = 0
    let start = -1
    const edges = [...segment.matchAll(TABLE_EDGE)]
    // a passage that opens inside a table: rows before the first `</table>`, or before nothing
    // at all when the table runs on past this passage too
    if (!options.code && from === 0 && (edges.length === 0 || edges[0][1])) {
      const stop = edges.length > 0 ? edges[0].index : segment.length
      if (/<\/?t[rdh]\b[^<>]*>/.test(segment.slice(0, stop))) {
        const end = edges.length > 0 ? stop + edges[0][0].length : to
        spans.push({ start: 0, end, kind: 'table', continued: true })
        edges.shift()
        if (edges.length === 0 && end === to) return
      }
    }
    for (const m of edges) {
      if (m[1]) {
        if (depth === 0) continue
        depth--
        if (depth === 0)
          spans.push({ start: from + start, end: from + m.index + m[0].length, kind: 'table' })
      } else {
        if (depth === 0) start = m.index
        depth++
      }
    }
    if (depth > 0) spans.push({ start: from + start, end: to, kind: 'table' })
  }
  for (const fence of fences) {
    scan(cursor, fence.start)
    spans.push(fence)
    cursor = fence.end
  }
  scan(cursor, source.length)
  return spans
}

export function fenceBlock(body: string): Element {
  return h('pre.lat-code', [h('code', decodeEntities(body.replace(/\n+$/, '')))])
}

// ---- pipe tables ------------------------------------------------------------------------------

const DELIM_CELL = /^\s*:?-+:?\s*$/

/** cells of a pipe row, split on `|` outside `$…$` and backticks; `\|` outside math is a pipe */
function pipeCells(line: string): string[] {
  let row = line.trim()
  if (row.startsWith('|')) row = row.slice(1)
  if (row.endsWith('|') && !row.endsWith('\\|')) row = row.slice(0, -1)
  const cells: string[] = []
  let cell = ''
  let math = false
  let code = false
  for (let i = 0; i < row.length; i++) {
    const ch = row[i]
    if (ch === '\\' && row[i + 1] === '|' && !math) {
      cell += '|'
      i++
    } else if (ch === '\\' && i + 1 < row.length) {
      cell += ch + row[i + 1]
      i++
    } else if (ch === '$' && !code) {
      math = !math
      cell += ch
    } else if (ch === '`' && !math) {
      code = !code
      cell += ch
    } else if (ch === '|' && !math && !code) {
      cells.push(cell.trim())
      cell = ''
    } else cell += ch
  }
  cells.push(cell.trim())
  return cells
}

function isDelimiter(line: string): boolean {
  if (!line.includes('|') && !line.includes('-')) return false
  const cells = pipeCells(line)
  return line.includes('|') && cells.length > 0 && cells.every(c => DELIM_CELL.test(c))
}

/**
 * Split a paragraph block around a GFM pipe table: a header row with a pipe, a delimiter row,
 * then every following line with a pipe. Returns null when the block holds none.
 */
export function pipeTable(
  block: string,
  cellRuns: (text: string) => ElementContent[],
): { before: string; table: Element; after: string } | null {
  const lines = block.split('\n')
  const at = lines.findIndex((line, i) => i > 0 && lines[i - 1].includes('|') && isDelimiter(line))
  if (at < 0) return null
  const header = pipeCells(lines[at - 1])
  const align = pipeCells(lines[at]).map(c => {
    const t = c.trim()
    return t.endsWith(':') ? (t.startsWith(':') ? 'center' : 'right') : undefined
  })
  let end = at + 1
  while (end < lines.length && lines[end].includes('|')) end++
  const cell = (tag: 'th' | 'td', text: string, i: number) =>
    h(tag, { style: align[i] ? `text-align:${align[i]}` : undefined }, cellRuns(text))
  const body = lines.slice(at + 1, end).map(line => {
    const cells = pipeCells(line)
    return h(
      'tr',
      header.map((_, i) => cell('td', cells[i] ?? '', i)),
    )
  })
  const table = h('table', [
    h('thead', [
      h(
        'tr',
        header.map((text, i) => cell('th', text, i)),
      ),
    ]),
    body.length > 0 ? h('tbody', body) : null,
  ])
  return {
    before: lines.slice(0, at - 1).join('\n'),
    table: container(table),
    after: lines.slice(end).join('\n'),
  }
}

// ---- extracts ---------------------------------------------------------------------------------

/**
 * A link extract as plain text: allowlisted tags dropped (cells and rows leave a space, so a
 * table reads as its words), a tag the 200-character cut left open removed, fence markers gone.
 * Unknown tags stay, as they do in the reader.
 */
export function flattenHtml(text: string): string {
  return text
    .replace(/<\/?[a-z][a-z0-9]*(?:\s[^<>]*)?$/, '')
    .replace(new RegExp(TAG, 'g'), (whole, _slash: string, name: string) =>
      !known(name) ? whole : /^(td|th|tr|br|table|caption|p|div)$/.test(name) ? ' ' : '',
    )
    .replace(/^[ \t]*(`{3,}|~{3,})[^\n]*$/gm, '')
    .replace(/[ \t]{2,}/g, ' ')
}
