import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'

/**
 * Markdown for note passages. The builder hands the reader each note chunk's markdown slice
 * (wikilinks, citations and sidenotes already resolved), so this covers what notes write between
 * the blocks ./html and the caller take out first (fenced code, html tables): headings,
 * paragraphs with soft wraps, lists, block quotes and callouts, footnotes, rules, display math,
 * and inline emphasis, links, highlights and footnote refs around the caller's math.
 */

export interface MdHooks {
  /** inline runs of a text: html allowlist, entities, then `mdInline` */
  inline: (text: string) => ElementContent[]
  /** a pipe table split out of a paragraph, or null */
  pipe: (block: string) => { before: string; table: Element; after: string } | null
  /** display math */
  display: (tex: string) => Element
  /**
   * fenced code inside a quote or a list item (top-level fences are taken out earlier); `info`
   * names garden components (`quotes`, `poetry`, `sms`, `tracking`) the caller renders as such
   */
  fence: (body: string, info: string) => Element
}

// Obsidian block ids (`text ^ava-writing`, or `^id` on its own line) are anchors, not text
const BLOCK_ID = /(?:^|[ \t]+)\^(?=[\w-]*[A-Za-z-])[A-Za-z0-9][\w-]*[ \t]*$/gm
const unblock = (text: string) => text.replace(BLOCK_ID, '')
// a task item's box: `- [ ] todo`, `- [x] done`
const TASK = /^\[([ xX])\][ \t]+/

const MATH = String.raw`\$\$([\s\S]+?)\$\$|\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)`
const HREF = String.raw`<?([^()\s<>]*(?:\([^()\s]*\)[^()\s<>]*)*)>?`
const INLINE = new RegExp(
  [
    MATH, // 1 display tex, 2 inline tex
    String.raw`!\[[^\]\n]*\]\([^)\n]*\)`, // image: its src is relative to the note, drop it
    String.raw`\[((?:[^\[\]\n]|\[[^\[\]\n]*\])+)\]\(\s*${HREF}(?:\s+"[^"\n]*")?\s*\)`, // 3 text, 4 href
    String.raw`<(https?:\/\/[^\s<>]+)>`, // 5 autolink
    String.raw`\[\^([^\]\s]+)\]`, // 6 footnote ref
    String.raw`(\*\*|__)(?![\s*_])([^\n]+?)(?<![\s*_])\7`, // 7 delimiter, 8 strong
    String.raw`(\*|_)(?![\s*_])([^\n]+?)(?<![\s*_])\9`, // 9 delimiter, 10 em
    String.raw`~~(?!\s)([^\n]+?)(?<!\s)~~`, // 11 del
    String.raw`==(?!\s)([^\n]+?)(?<!\s)==`, // 12 mark
  ].join('|'),
  'g',
)

const SAFE_HREF = /^(?:https?:\/\/|mailto:|\/|#)/i
const dollarsBalanced = (text: string) =>
  (text.replace(/\\\$/g, '').split('$').length - 1) % 2 === 0

/**
 * Inline markdown of a text run. Math is matched first at every position, so `_` and `*` inside
 * TeX stay TeX; emphasis must hold balanced dollars, and `_` only opens and closes at word edges
 * (snake_case stays text). A rejected match gives its first character back as text.
 */
export function mdInline(
  text: string,
  math: (tex: string, display: boolean) => ElementContent,
): ElementContent[] {
  const out: ElementContent[] = []
  const re = new RegExp(INLINE.source, 'g')
  let last = 0
  for (let m = re.exec(text); m; m = re.exec(text)) {
    const node = convert(m, text, re.lastIndex, math)
    if (node === null) {
      re.lastIndex = m.index + 1
      continue
    }
    if (m.index > last) out.push({ type: 'text', value: text.slice(last, m.index) })
    out.push(...node)
    last = re.lastIndex
  }
  if (last < text.length) out.push({ type: 'text', value: text.slice(last) })
  return out
}

function convert(
  m: RegExpExecArray,
  text: string,
  end: number,
  math: (tex: string, display: boolean) => ElementContent,
): ElementContent[] | null {
  const inner = (s: string) => mdInline(s, math)
  if (m[1] !== undefined) return m[1].trim() ? [math(m[1].trim(), true)] : null
  if (m[2] !== undefined) return [math(m[2], false)]
  if (m[0].startsWith('!')) return []
  if (m[3] !== undefined) {
    const href = m[4] ?? ''
    if (!SAFE_HREF.test(href)) return inner(m[3])
    const internal = href.startsWith('/') || href.startsWith('#')
    return [
      h(
        'a',
        internal
          ? { href, className: ['internal'] }
          : { href, className: ['external'], rel: 'noopener noreferrer', target: '_blank' },
        inner(m[3]),
      ),
    ]
  }
  if (m[5] !== undefined)
    return [h('a.external', { href: m[5], rel: 'noopener noreferrer', target: '_blank' }, m[5])]
  if (m[6] !== undefined) return [h('sup.lat-fn', m[6])]
  const delimiter = m[7] ?? m[9]
  const body = m[8] ?? m[10] ?? m[11] ?? m[12]
  if (body === undefined || !dollarsBalanced(body)) return null
  if (delimiter?.startsWith('_')) {
    const before = text[m.index - 1]
    const after = text[end]
    if ((before && /\w/.test(before)) || (after && /\w/.test(after))) return null
  }
  if (m[8] !== undefined) return [h('strong', inner(body))]
  if (m[10] !== undefined) return [h('em', inner(body))]
  if (m[11] !== undefined) return [h('del', inner(body))]
  return [h('mark', inner(body))]
}

// ---- blocks -----------------------------------------------------------------------------------

const QUOTE = /^ {0,3}>/
const HEADING = /^ {0,3}#{1,6}\s+/
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/
const ITEM = /^([ \t]*)([-*+]|\d{1,9}[.)])[ \t]+(.*)$/
const FOOTNOTE = /^\[\^([^\]\s]+)\]:[ \t]?(.*)$/
const FENCE = /^[ \t]*(`{3,}|~{3,})/
const DISPLAY_OPEN = /^[ \t]*\$\$/
const META = /^([ \t]*)[-*+][ \t]+\[meta\]:[ \t]*$/

const indentOf = (line: string) => /^[ \t]*/.exec(line)![0].replace(/\t/g, '    ').length
const startsBlock = (line: string) =>
  QUOTE.test(line) ||
  HEADING.test(line) ||
  RULE.test(line) ||
  ITEM.test(line) ||
  FOOTNOTE.test(line) ||
  FENCE.test(line) ||
  DISPLAY_OPEN.test(line)

function dedent(lines: string[], by: number): string[] {
  return lines.map(line => {
    let cut = 0
    let width = 0
    while (cut < line.length && width < by && (line[cut] === ' ' || line[cut] === '\t')) {
      width += line[cut] === '\t' ? 4 : 1
      cut++
    }
    return line.slice(cut)
  })
}

/** Block markup of a markdown source, line by line. */
export function mdBlocks(source: string, hooks: MdHooks): Element[] {
  const lines = source.split('\n')
  const out: Element[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      i++
      continue
    }
    if (QUOTE.test(line)) {
      const body: string[] = []
      while (i < lines.length && QUOTE.test(lines[i]))
        body.push(lines[i++].replace(/^ {0,3}> ?/, ''))
      const callout = /^\[!([\w-]+)\][+-]?[ \t]*(.*)$/.exec(body[0] ?? '')
      const rest = callout ? body.slice(1) : body
      out.push(
        h('blockquote.lat-quote', { dataCallout: callout ? callout[1].toLowerCase() : undefined }, [
          callout ? h('p.lat-callout-title', hooks.inline(callout[2] || callout[1])) : null,
          ...mdBlocks(rest.join('\n'), hooks),
        ]),
      )
    } else if (FENCE.test(line)) {
      const marker = FENCE.exec(line)![1]
      const info = line.replace(FENCE, '').trim()
      const body: string[] = []
      i++
      while (
        i < lines.length &&
        !new RegExp(`^[ \\t]*${marker[0]}{${marker.length},}[ \\t]*$`).test(lines[i])
      )
        body.push(lines[i++])
      i++
      if (body.join('').trim()) out.push(hooks.fence(body.join('\n'), info))
    } else if (DISPLAY_OPEN.test(line)) {
      // `$$ … $$` on one line, or opening and closing lines around the TeX
      const first = line.replace(DISPLAY_OPEN, '')
      const close = first.indexOf('$$')
      const tex: string[] = []
      if (close >= 0) {
        tex.push(first.slice(0, close))
        i++
      } else {
        tex.push(first)
        i++
        while (i < lines.length && !lines[i].includes('$$')) tex.push(lines[i++])
        if (i < lines.length) tex.push(lines[i++].split('$$')[0])
      }
      if (tex.join('').trim()) out.push(hooks.display(tex.join('\n').trim()))
    } else if (HEADING.test(line)) {
      out.push(
        h(
          'p.lat-h',
          hooks.inline(unblock(line.replace(HEADING, '').replace(/[ \t]+#+[ \t]*$/, ''))),
        ),
      )
      i++
    } else if (RULE.test(line)) {
      out.push(h('hr'))
      i++
    } else if (FOOTNOTE.test(line)) {
      const [, id, first] = FOOTNOTE.exec(line)!
      const body = [first]
      i++
      while (i < lines.length && lines[i].trim() && indentOf(lines[i]) >= 2)
        body.push(lines[i++].trim())
      out.push(h('p.lat-fn-def', [h('sup.lat-fn', id), ' ', ...hooks.inline(body.join(' '))]))
    } else if (META.test(line)) {
      i = meta(lines, i, out)
    } else if (ITEM.test(line)) {
      i = list(lines, i, out, hooks)
    } else {
      const body: string[] = []
      while (i < lines.length && lines[i].trim() && (body.length === 0 || !startsBlock(lines[i])))
        body.push(lines[i++])
      const pipe = hooks.pipe(body.join('\n'))
      if (pipe) {
        out.push(...paragraph(pipe.before, hooks), pipe.table, ...mdBlocks(pipe.after, hooks))
      } else out.push(...paragraph(body.join('\n'), hooks))
    }
  }
  return out
}

/** Soft wraps join with a space; a line ending in two spaces or a backslash breaks. */
function paragraph(text: string, hooks: MdHooks): Element[] {
  text = unblock(text)
  if (!text.trim()) return []
  const joined = text
    .split('\n')
    .map((line, i, all) =>
      i === all.length - 1
        ? line.trim()
        : / {2,}$|\\$/.test(line)
          ? `${line.replace(/(?: {2,}|\\)$/, '').trim()}\n`
          : `${line.trim()} `,
    )
    .join('')
  return [h('p', hooks.inline(joined))]
}

/** A list from line `start`; returns the line after it. Nested lists come from the item bodies. */
function list(lines: string[], start: number, out: Element[], hooks: MdHooks): number {
  const head = ITEM.exec(lines[start])!
  const indent = indentOf(head[1])
  const ordered = /\d/.test(head[2])
  const items: Element[] = []
  let i = start
  let loose = false
  const sameList = (line: string) => {
    const m = ITEM.exec(line)
    return m !== null && indentOf(m[1]) === indent && /\d/.test(m[2]) === ordered
  }
  while (i < lines.length && sameList(lines[i])) {
    const m = ITEM.exec(lines[i])!
    // continuation lines sit under the item's text
    const content = indent + m[2].length + 1
    const task = TASK.exec(m[3])
    const body = [
      task ? `${task[1] === ' ' ? '\u2610' : '\u2611'}\u00a0${m[3].slice(task[0].length)}` : m[3],
    ]
    i++
    while (i < lines.length) {
      const line = lines[i]
      if (!line.trim()) {
        const next = lines.slice(i + 1).find(l => l.trim())
        if (next === undefined) break
        if (sameList(next)) {
          loose = true
          i++
          break
        }
        if (indentOf(next) > indent) {
          body.push('')
          i++
          continue
        }
        break
      }
      if (sameList(line) || (ITEM.test(line) && indentOf(line) < indent)) break
      if (indentOf(line) > indent) body.push(...dedent([line], content))
      else if (!startsBlock(line)) body.push(line.trim())
      else break
      i++
    }
    items.push(h('li', mdBlocks(body.join('\n'), hooks)))
  }
  const ol = ordered ? Number(/\d+/.exec(head[2])![0]) : 1
  out.push(
    h(
      ordered ? 'ol' : 'ul',
      {
        className: ['lat-list-md'],
        dataLoose: loose ? '' : undefined,
        start: ordered && ol !== 1 ? ol : undefined,
      },
      items,
    ),
  )
  return i
}

/**
 * A stream entry's `- [meta]:` list (date, tags, other `key: value` pairs, see content/stream.md)
 * as one line of metadata: `2026-10-28 16:07 · life · death · training log 087`.
 */
function meta(lines: string[], start: number, out: Element[]): number {
  const indent = indentOf(META.exec(lines[start])![1])
  const parts: string[] = []
  let date = ''
  let i = start + 1
  while (i < lines.length && lines[i].trim() && indentOf(lines[i]) > indent) {
    const entry = /^[ \t]*[-*+][ \t]+(?:([\w-]+):[ \t]*)?(.*)$/.exec(lines[i])
    i++
    if (!entry) continue
    const [, key, value] = entry
    if (key === 'date') date = value.replace(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}).*$/, '$1 $2')
    else if (key && value.trim()) parts.push(value.trim())
    else if (!key && value.trim()) parts.push(value.trim())
  }
  out.push(
    h('p.lat-meta', [
      date ? h('time', { dateTime: date.replace(' ', 'T') }, date) : null,
      ...parts.flatMap((part, n) => (date || n > 0 ? [' · ', part] : [part])),
    ]),
  )
  return i
}
