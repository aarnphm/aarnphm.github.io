import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import { canonicalizeCallout, formatMathCalloutTitle, isMathCallout } from '../callouts'
import { linkProperties } from './pdf'

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
  /** an html figure or svg block (`<figure>…</figure>`, `<svg>…</svg>`), whole or cut off */
  figure: (html: string) => Element[]
}

// Obsidian block ids (`text ^ava-writing`, or `^id` on its own line) are anchors, not text
const BLOCK_ID = /(?:^|[ \t]+)\^(?=[\w-]*[A-Za-z-])[A-Za-z0-9][\w-]*[ \t]*$/gm
const unblock = (text: string) => text.replace(BLOCK_ID, '')
// a task item's box: `- [ ] todo`, `- [x] done`
const TASK = /^\[([ xX])\][ \t]+/

const MATH = String.raw`\$\$([\s\S]+?)\$\$|\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)`
const HREF = String.raw`<?([^()\s<>]*(?:\([^()\s]*\)[^()\s<>]*)*)>?`
// a tag's characters (ofm.ts tagRegex is letters, marks, digits, emoji, `-` and `_`): anything but
// space, ASCII punctuation other than `-` and `_`, and the slash between segments
const TAG_CHAR = String.raw`[^\s!-,./:-@[-^\`{-~]`
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
    String.raw`(\`+)(?!\`)([^\n]*?[^\`\n])\13(?!\`)`, // 13 ticks, 14 code span
    // the Obsidian and GFM extras note pages render (ofm.ts, remark-gfm)
    String.raw`(?<![\w/@.\-])(https?:\/\/[^\s<>]+)`, // 15 bare url
    String.raw`(?<=^|\s)#(${TAG_CHAR}+(?:\/${TAG_CHAR}+)*)`, // 16 tag
    String.raw`::([^:\n]+?)(?:\{(h[1-7])\})?::`, // 17 marker text, 18 intensity
    String.raw`\^\[((?:[^\[\]\n]|\[[^\[\]\n]*\])+)\]`, // 19 inline footnote
    String.raw`(-{1,2}>|={1,2}>|<-{1,2}|<={1,2})`, // 20 arrow
  ].join('|'),
  'g',
)

// ofm.ts `arrowMapping`
const ARROWS: Record<string, string> = {
  '->': '→',
  '-->': '⇒',
  '=>': '⇒',
  '==>': '⇒',
  '<-': '←',
  '<--': '⇐',
  '<=': '⇐',
  '<==': '⇐',
}

/** `[text](href)`: a link when the href is a web, mail, site or anchor address, else its text */
export function mdLink(href: string, children: ElementContent[]): ElementContent[] {
  if (!SAFE_HREF.test(href)) return children
  return [h('a', linkProperties(href), children)]
}

/** rehype-pretty-code's language hint at the end of a code span (`berkeley mono{:text}`) */
export const codeText = (body: string) =>
  // one space each side is padding, as in CommonMark
  (/^ [\s\S]* $/.test(body) && body.trim() ? body.slice(1, -1) : body).replace(/\{:[^{}\s]*\}$/, '')

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
  if (m[3] !== undefined) return mdLink(m[4] ?? '', inner(m[3]))
  if (m[15] !== undefined) {
    // GFM leaves trailing punctuation, and a close paren the url did not open, outside the link
    let url = m[15].replace(/[.,:;!?'"*_~]+$/, '')
    while (url.endsWith(')') && url.split('(').length < url.split(')').length)
      url = url.slice(0, -1).replace(/[.,:;!?'"*_~]+$/, '')
    const rest = m[15].slice(url.length)
    return [
      h('a', linkProperties(url), url),
      ...(rest ? [{ type: 'text' as const, value: rest }] : []),
    ]
  }
  if (m[16] !== undefined) {
    // all digits and slashes (`#1`, `#2/3`) is not a tag
    if (/^[\d/]+$/.test(m[16])) return null
    const slug = m[16]
      .split('/')
      .map(s => s.replace(/&/g, '-and-').replace(/%/g, '-percent').replace(/\?/g, ''))
      .join('/')
    // the label is lowercase (the embeddings view shows every tag that way); the href keeps the slug
    return [h('a.internal.tag-link', { href: `/tags/${slug}` }, slug.toLowerCase())]
  }
  if (m[17] !== undefined)
    return [h('span', { className: ['marker', `marker-${m[18] ?? 'h3'}`] }, inner(m[17]))]
  if (m[19] !== undefined) return [h('small.lat-fn-inline', ['(', ...inner(m[19]), ')'])]
  if (m[20] !== undefined) return [{ type: 'text', value: ARROWS[m[20]] }]
  if (m[5] !== undefined) return [h('a', linkProperties(m[5]), m[5])]
  if (m[6] !== undefined) return [h('sup.lat-fn', m[6])]
  if (m[14] !== undefined) return [h('code', codeText(m[14]))]
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
const FIGURE = /^ {0,3}<(figure|svg)\b/
// raw `<pre>` (a hex dump, a listing with coloured spans) is a code block on the page too
const PRE = /^ {0,3}<pre\b/i

/**
 * Where the html block opening `source` ends: after the close that balances its first tag
 * (nested svgs inside an svg count), or at the end when the chunk cut it off.
 */
function figureEnd(source: string, tag: string): number {
  const re = new RegExp(`<(\\/?)${tag}\\b[^>]*?(\\/?)>`, 'g')
  let depth = 0
  for (const m of source.matchAll(re)) {
    if (m[1]) depth--
    else if (!m[2]) depth++
    if (depth <= 0) return m.index + m[0].length
  }
  return source.length
}

const indentOf = (line: string) => /^[ \t]*/.exec(line)![0].replace(/\t/g, '    ').length
const startsBlock = (line: string) =>
  QUOTE.test(line) ||
  HEADING.test(line) ||
  RULE.test(line) ||
  ITEM.test(line) ||
  FOOTNOTE.test(line) ||
  FENCE.test(line) ||
  DISPLAY_OPEN.test(line) ||
  FIGURE.test(line) ||
  PRE.test(line)

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

/**
 * Block markup of a markdown source. A chunk cut inside a raw `<pre>` opens on its listing: the
 * text before a `</pre>` no `<pre>` opened is that listing.
 */
export function mdSource(source: string, hooks: MdHooks): Element[] {
  const close = /<\/pre\s*>/i.exec(source)
  if (!close || /<pre\b/i.test(source.slice(0, close.index))) return mdBlocks(source, hooks)
  const listing = source.slice(0, close.index).replace(/<[^>]*>/g, '')
  return [
    ...(listing.trim() ? [hooks.fence(listing.replace(/^\n+/, ''), '')] : []),
    ...mdBlocks(source.slice(close.index + close[0].length), hooks),
  ]
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
      const head = CALLOUT.exec(body[0] ?? '')
      out.push(
        head
          ? callout(head, body.slice(1).join('\n'), hooks)
          : h('blockquote.lat-quote', mdBlocks(body.join('\n'), hooks)),
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
    } else if (PRE.test(line)) {
      // the block runs to its `</pre>` (or the chunk's end); tags inside drop, text after the
      // close on the same line carries on as markdown
      const rest = lines.slice(i).join('\n')
      const close = /<\/pre\s*>/i.exec(rest)
      const end = close ? close.index + close[0].length : rest.length
      const body = rest
        .slice(0, close ? close.index : rest.length)
        .replace(/^[^>]*>\n?/, '')
        .replace(/<[^>]*>/g, '')
      if (body.trim()) out.push(hooks.fence(body, ''))
      out.push(...mdBlocks(rest.slice(end), hooks))
      break
    } else if (FIGURE.test(line)) {
      // a drawing is one block however many lines (and blank lines) it spans; text after its
      // close on the same line carries on as markdown
      const rest = lines
        .slice(i)
        .join('\n')
        .replace(/^ {0,3}/, '')
      const end = figureEnd(rest, FIGURE.exec(line)![1])
      out.push(...hooks.figure(rest.slice(0, end)), ...mdBlocks(rest.slice(end), hooks))
      break
    } else if (META.test(line)) {
      i = meta(lines, i, out, hooks)
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

// `> [!type|metadata]± title`, as ObsidianFlavoredMarkdown reads a callout's first line
const CALLOUT = /^\[!([\w-]+)(?:\|([^\]]*))?\]([+-]?)[ \t]*(.*)$/

/**
 * A callout as the note page sets it (ofm.ts): `blockquote.callout.<type>` with its title and a
 * `.callout-content` body, foldable when the head says `+` or `-` (folded). callouts.scss styles
 * it and the reader's click handler folds it. A passage cannot count a page's theorems, so a
 * math callout keeps only a number its title gives.
 */
function callout(head: RegExpExecArray, body: string, hooks: MdHooks): Element {
  const [, raw, metadata, fold, title] = head
  const type = canonicalizeCallout(raw.toLowerCase())
  const math = isMathCallout(type)
  let shown = title.trim()
  if (math && type !== 'math')
    shown = (formatMathCalloutTitle(type, shown, shown, 0) ?? shown).replace(/^(\w+) 0\./, '$1.')
  else if (!shown) shown = (raw[0].toUpperCase() + raw.slice(1)).replace(/-/g, ' ')
  const content = mdBlocks(body, hooks)
  return h(
    'blockquote',
    {
      className: [
        'callout',
        type,
        math ? 'callout-math' : '',
        fold ? 'is-collapsible' : '',
        fold === '-' ? 'is-collapsed' : '',
      ].filter(Boolean),
      dataCallout: type,
      dataCalloutFold: fold ? '' : undefined,
      dataCalloutMetadata: metadata || undefined,
    },
    [
      h('div.callout-title', [h('div.callout-title-inner', [h('p', hooks.inline(shown))])]),
      content.length > 0 ? h('div.callout-content', content) : null,
    ],
  )
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
function meta(lines: string[], start: number, out: Element[], hooks: MdHooks): number {
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
    else if (!key && value.trim()) {
      // a bare item under `tags:` is a tag, shown lowercase; links and urls keep their case
      const item = value.trim()
      parts.push(/\]\(|\[\[|:\/\//.test(item) ? item : item.toLowerCase())
    }
  }
  out.push(
    h('p.lat-meta', [
      date ? h('time', { dateTime: date.replace(' ', 'T') }, date) : null,
      // a value can be a link (`[dualism](/thoughts/dualism)`) or a bare url
      ...parts.flatMap((part, n) => [...(date || n > 0 ? [' · '] : []), ...hooks.inline(part)]),
    ]),
  )
  return i
}
