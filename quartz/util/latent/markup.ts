import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import type {
  LatentAxis,
  LatentChunk,
  LatentDoc,
  LatentIndex,
  LatentKind,
  LatentPreview,
  LatentText,
  LatentTrackDoc,
} from './types'
import {
  type HtmlOptions,
  blockSpans,
  decodeEntities,
  fenceBlock,
  flattenHtml,
  htmlRuns,
  htmlTable,
  pipeTable,
  repairCuts,
} from './html'
import { type MdHooks, mdBlocks, mdInline } from './md'
import {
  LABEL_ROW,
  LABEL_X,
  LANE_X,
  LEADER_X,
  OPEN_WIDTH,
  RAIL_HEIGHT,
  RAIL_WIDTH,
  TRACK_X,
  UNDATED_FIRST,
  UNDATED_ROW,
  layoutRail,
  placeLabels,
  thinYears,
  truncate,
} from './rail'
import { KIND_ORDER, inScope, rowHref } from './types'

/**
 * hast builders for every pane. The transformer renders them into the page at build time; the
 * client renders the same trees to HTML for a focus change, so both sides agree to the glyph.
 */

export const KIND_LABEL: Record<LatentKind, string> = {
  note: 'note',
  pdf: 'pdf',
  image: 'image',
  base: 'base',
}

export function mark(kind: LatentKind): Element {
  return h('span.lat-mark', { dataKind: kind, title: KIND_LABEL[kind] }, KIND_LABEL[kind])
}

/** A pdf page's printed label; some stores carry it with its own "p." already. */
function pageLabel(label: string | undefined, page: number): string {
  return (label || String(page)).replace(/^p(?:p|age)?\.?\s*/i, '')
}

export function locus(preview: Pick<LatentPreview, 'c' | 'p' | 'l' | 's'>, total?: number): string {
  // `c` counts from 0 (per page for pdfs, per document for notes); both read from 1
  if (preview.s === 'pdf' && preview.p)
    return `p. ${pageLabel(preview.l, preview.p)}${total ? ` of ${total}` : ''} ¶${preview.c + 1}`
  if (preview.s === 'note') return total ? `§${preview.c + 1} of ${total}` : `§${preview.c + 1}`
  return ''
}

// Math follows the notes' syntax: `$$…$$` is display, `$…$` is inline under pandoc's rule (no
// space inside either dollar, no digit after the closing one, so "$5 and $10" stays text). The
// builder restores the dollars notes lost on the way to the embedding text.
const DISPLAY_MATH = /\$\$([\s\S]+?)\$\$/g
const INLINE = /\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)|\*\*(?!\s)([^*\n]+?)(?<!\s)\*\*/g

/**
 * Inline runs of a paragraph: text, `span.math-inline` and strong. The math spans carry their
 * TeX as text; the transformer renders them with KaTeX at build, the client when a part loads.
 */
function inlineRuns(text: string): ElementContent[] {
  const out: ElementContent[] = []
  let last = 0
  for (const match of text.matchAll(INLINE)) {
    if (match.index > last) out.push({ type: 'text', value: text.slice(last, match.index) })
    out.push(match[1] !== undefined ? h('span.math.math-inline', match[1]) : h('strong', match[2]))
    last = match.index + match[0].length
  }
  if (last < text.length) out.push({ type: 'text', value: text.slice(last) })
  return out
}

/**
 * A link's extract: entities decoded, heading hashes dropped, display math folded inline (the
 * extract is clamped to three lines), and math the 200-character preview cut off replaced by an
 * ellipsis rather than shown as raw TeX.
 */
function extractRuns(x: string): ElementContent[] {
  let text = decodeEntities(flattenHtml(x)).replace(/(^|\s)#{1,6}\s+/g, '$1')
  const display = text.split('$$').length - 1
  if (display % 2 === 1) text = `${text.slice(0, text.lastIndexOf('$$')).trimEnd()} …`
  text = text.replace(DISPLAY_MATH, (_, tex: string) => `$${tex.trim().replace(/\s+/g, ' ')}$`)
  const dollars = text.replace(/\\\$/g, '').split('$').length - 1
  if (dollars % 2 === 1) text = `${text.slice(0, text.lastIndexOf('$')).trimEnd()} …`
  return inlineRuns(text)
}

// text between tags: entities decoded (hast escapes again on output), then math and strong
const leaf = (text: string) => inlineRuns(decodeEntities(text))

// a table cell is one line, so display math folds inline there
const cellLeaf = (text: string) =>
  inlineRuns(
    decodeEntities(text).replace(
      DISPLAY_MATH,
      (_, tex: string) => `$${tex.trim().replace(/\s+/g, ' ')}$`,
    ),
  )

interface PassageOptions extends HtmlOptions {
  /** the chunk is the note's markdown slice (`m` in the text sidecar) */
  markdown?: boolean
  /** the chunk is a note's rendered text: every line is a paragraph */
  lines?: boolean
}

const mathSpan = (tex: string, display: boolean) =>
  h(display ? 'span.math.math-display' : 'span.math.math-inline', tex)
const mdLeaf = (text: string) => mdInline(decodeEntities(text), mathSpan)
const mdCellLeaf = (text: string) => mdInline(decodeEntities(text), tex => mathSpan(tex, false))

// Raw html figures in notes (an inline svg pictogram, a figure wrapper) are pictures, not text;
// `{{phrase}}` marks a phrase to hear in the French notes and reads as the phrase (TeX's `{{` sits
// after a letter, `^`, `_` or a backslash, and holds no `$`, so it is left alone).
const SVG = /<svg\b[\s\S]*?(?:<\/svg>|$)/g
const MARKED = /(?<![\\\w^_{])\{\{([^{}\n$\\]+)\}\}(?!\})/g
const gardenText = (text: string) => text.replace(SVG, '').replace(MARKED, '$1')

const mdHooks = (options: PassageOptions): MdHooks => ({
  inline: text => htmlRuns(gardenText(text), mdLeaf, options),
  pipe: block => pipeTable(block, cell => htmlRuns(gardenText(cell), mdCellLeaf, options)),
  display: tex => h('div.math.math-display', decodeEntities(tex)),
  fence: (body, info) => fenceView(body, info, options),
})

const TRACK_LINE = /^[ \t]*([\w-]+):[ \t]*(.*)$/

/**
 * A fence in a note's markdown slice. Garden components render as the page shows them: `quotes`
 * is a quote whose last paragraph is its attribution, `poetry` and `poem` keep their line breaks,
 * `sms` is a message, `tracking` (the triathlon log) one metadata line. Anything else is code, as
 * is every fence outside a markdown slice.
 */
function fenceView(body: string, info: string | undefined, options: PassageOptions): Element {
  const kind = (info ?? '').split(/\s/)[0].toLowerCase()
  if (!options.markdown) return fenceBlock(body)
  const hooks = mdHooks(options)
  const stanzas = body
    .replace(/^\n+|\s+$/g, '')
    .split(/\n[ \t]*\n+/)
    .filter(s => s.trim())
  if (kind === 'quotes' && stanzas.length > 0) {
    const parts = stanzas.map(s => s.trim().replace(/--/g, '—'))
    const cite = parts.length > 1 ? parts.pop()! : null
    return h('blockquote.lat-quote.lat-quotes', [
      ...mdBlocks(parts.join('\n\n'), hooks),
      cite
        ? h('p.lat-cite', ['\u2014\u2009', ...hooks.inline(cite.replace(/\s*\n\s*/g, ' '))])
        : null,
    ])
  }
  if (kind === 'poetry' || kind === 'poem')
    return h(
      'div.lat-verse',
      stanzas.map(s => h('p', hooks.inline(s))),
    )
  if (kind === 'sms')
    return h(
      'div.lat-sms',
      stanzas.map(s => h('p', hooks.inline(s.trim()))),
    )
  if (kind === 'tracking') {
    const fields = body
      .split('\n')
      .map(line => TRACK_LINE.exec(line))
      .filter((m): m is RegExpExecArray => m !== null)
    const date = fields.find(m => m[1] === 'date')?.[2]
    const rest = fields
      .filter(m => m[1] !== 'date')
      .map(m => (m[2] === 'true' ? m[1] : `${m[1]} ${m[2]}`))
    return h('p.lat-meta', [
      date ? h('time', { dateTime: date }, date) : null,
      ...rest.flatMap((part, n) => (date || n > 0 ? [' · ', part] : [part])),
    ])
  }
  return fenceBlock(body)
}

function paragraphs(text: string, options: PassageOptions): Element[] {
  const out: Element[] = []
  for (const raw of text.split(options.lines ? /\n+/ : /\n[ \t]*\n+/)) {
    const block = raw.trim()
    if (!block) continue
    const pipe = pipeTable(block, cell => htmlRuns(cell, cellLeaf, options))
    if (pipe) {
      out.push(...paragraphs(pipe.before, options), pipe.table, ...paragraphs(pipe.after, options))
      continue
    }
    const heading = /^#{1,6}\s+(.+)$/.exec(block)
    out.push(
      heading && !block.includes('\n')
        ? h('p.lat-h', htmlRuns(heading[1], leaf, options))
        : h('p', htmlRuns(block, leaf, options)),
    )
  }
  return out
}

/** display math on its own, blank lines split the rest into paragraphs */
function prose(source: string, options: PassageOptions): Element[] {
  if (options.markdown) return mdBlocks(source, mdHooks(options))
  const blocks: Element[] = []
  let last = 0
  for (const match of source.matchAll(DISPLAY_MATH)) {
    blocks.push(...paragraphs(source.slice(last, match.index), options))
    const tex = decodeEntities(match[1]).trim()
    if (tex) blocks.push(h('div.math.math-display', tex))
    last = match.index + match[0].length
  }
  blocks.push(...paragraphs(source.slice(last), options))
  return blocks
}

/**
 * A passage as blocks: fenced code and html tables first (both stay whole across blank lines),
 * then display math, paragraphs and pipe tables. A one-line `# heading` becomes a bold paragraph
 * (a p, so the heading plugins and the toc never pick it up). Inline html is allowlisted by
 * ./html; `code` turns backtick spans into code (notes only); a note's markdown slice goes
 * through ./md.
 */
function passageBlocks(raw: string, options: PassageOptions = {}): Element[] {
  // pdf passages are cut by length, sometimes inside a tag
  const text = options.code ? raw : repairCuts(raw)
  const blocks: Element[] = []
  let last = 0
  for (const span of blockSpans(text, options)) {
    blocks.push(...prose(text.slice(last, span.start), options))
    if (span.kind === 'table')
      blocks.push(htmlTable(text.slice(span.start, span.end), cellLeaf, span.continued ?? false))
    else if (span.body?.trim()) blocks.push(fenceView(span.body, span.info, options))
    last = span.end
  }
  blocks.push(...prose(text.slice(last), options))
  return blocks.length > 0 ? blocks : [h('p.lat-empty', 'no text for this row')]
}

export type ReaderMeta = Pick<LatentText, 'd' | 'store' | 'total' | 'part'>

const SAME_LEAD: Record<NonNullable<LatentChunk['same']>[2], string> = {
  repeat: 'the page text of ',
  figure: 'the text drawn by the figure in ',
  code: 'the runtime data of the code cell in ',
}

/** A chunk whose text another chunk already shows: one line pointing there. */
function samePointer(same: NonNullable<LatentChunk['same']>): Element {
  const [row, chunk, why] = same
  return h('p.lat-empty.lat-same', [
    SAME_LEAD[why] ?? SAME_LEAD.repeat,
    h(
      'a',
      { href: `?focus=${row}`, dataLatFocus: String(row), dataNoPopover: true },
      `§${chunk + 1}`,
    ),
  ])
}

/** `article.lat-chunk` for each chunk; `chunk.c` is its index in the document. */
export function chunkArticles(
  meta: Pick<LatentText, 'store'>,
  doc: Pick<LatentDoc, 'id' | 's'>,
  chunks: LatentChunk[],
  focus: number,
): Element[] {
  return chunks.map(chunk =>
    h(
      'article.lat-chunk',
      {
        id: `lat-r${chunk.r}`,
        dataRow: String(chunk.r),
        dataPage: chunk.p ? String(chunk.p) : undefined,
        dataBox: chunk.box ? chunk.box.join(',') : undefined,
        dataFocus: chunk.r === focus ? '' : undefined,
      },
      [
        h('span.lat-locus', [
          meta.store === 'pdf' && chunk.p
            ? h(
                'a',
                { href: rowHref(doc, chunk.p), dataNoPopover: true },
                `p. ${pageLabel(chunk.l, chunk.p)} ¶${chunk.c + 1}`,
              )
            : h(
                'a',
                { href: `?focus=${chunk.r}`, dataLatFocus: String(chunk.r), dataNoPopover: true },
                `§${chunk.c + 1}`,
              ),
        ]),
        // an image row is the image: its embedding has no text to show
        ...(meta.store === 'image'
          ? [imageFrame(doc)]
          : chunk.same
            ? [samePointer(chunk.same)]
            : passageBlocks(chunk.text ?? '', {
                code: meta.store === 'note',
                markdown: chunk.m === 1,
                lines: meta.store === 'note' && chunk.m !== 1,
              })),
      ],
    ),
  )
}

/**
 * An image row in the reader. The frame has a fixed height so the image arriving does not move
 * the passages under it; the image is contained in it, top-left.
 */
function imageFrame(doc: Pick<LatentDoc, 'id' | 's'>): Element {
  const href = rowHref(doc)
  return h('a.lat-image', { href, dataNoPopover: true }, [
    h('img', {
      src: href,
      alt: doc.id.split('/').pop() ?? doc.id,
      loading: 'lazy',
      decoding: 'async',
    }),
  ])
}

/**
 * The gap left by parts not yet loaded, before or after the loaded run. The client replaces it
 * with the adjacent part when it scrolls near; `data-part` is the part that fills it next.
 */
export function moreSentinel(
  meta: ReaderMeta,
  side: 'before' | 'after',
  count: number,
  part: number,
): Element {
  const unit = meta.store === 'pdf' ? 'passages' : 'chunks'
  return h(
    'div.lat-more',
    { dataDoc: String(meta.d), dataSide: side, dataPart: String(part), role: 'status' },
    `⋯ ${count} ${side === 'before' ? 'earlier' : 'later'} ${unit}`,
  )
}

type HeadDoc = Pick<LatentDoc, 'id' | 's' | 't' | 'date' | 'n' | 'pages'>

/**
 * A document's head line. `mirror` is the reader's one frozen head: it sits at the top of the
 * reader and is retitled for the document under the reading line, so the top band never moves.
 */
export function docHead(doc: HeadDoc, mirror = false): Element {
  const href = rowHref(doc)
  const unit = doc.s === 'pdf' ? 'passages' : doc.s === 'note' ? 'chunks' : 'rows'
  return h(mirror ? 'header.lat-doc-head.lat-reader-head' : 'header.lat-doc-head', [
    h('div.lat-doc-kicker', [
      mark(doc.s),
      h('a.lat-doc-title', { href, dataNoPopover: true, title: doc.t }, doc.t),
      doc.date ? h('time.lat-doc-date', { dateTime: doc.date }, doc.date.slice(0, 4)) : null,
    ]),
    h('div.lat-doc-meta', [
      h('span', `${doc.n} ${unit}`),
      doc.s === 'pdf' && doc.pages ? h('span', `${doc.pages} pages`) : null,
      h('a.lat-doc-open', { href, dataNoPopover: true }, doc.s === 'pdf' ? '/read →' : 'note →'),
    ]),
  ])
}

/**
 * One document in the reader: its head and a run of chunk articles. `window` is the loaded
 * run (`chunks` starting at chunk index `offset`); a stub renders the head only. The run is a
 * whole number of parts so the client can grow it part by part.
 */
export function readerDocument(
  text: LatentText,
  doc: HeadDoc,
  focus: number,
  options: { stub?: boolean; offset?: number } = {},
): Element {
  const head = docHead(doc)
  if (options.stub) return h('section.lat-doc.lat-doc-stub', { dataDoc: String(text.d) }, [head])
  const offset = options.offset ?? 0
  const part = text.part || Math.max(1, text.total)
  const lo = Math.floor(offset / part)
  const end = offset + text.chunks.length
  const hi = Math.max(lo, Math.ceil(end / part) - 1)
  const after = text.total - end
  return h('section.lat-doc', { dataDoc: String(text.d), dataLo: String(lo), dataHi: String(hi) }, [
    head,
    offset > 0 ? moreSentinel(text, 'before', offset, lo - 1) : null,
    ...chunkArticles(text, doc, text.chunks, focus),
    after > 0 ? moreSentinel(text, 'after', after, hi + 1) : null,
  ])
}

export interface LinkItem {
  preview: LatentPreview
  score: number
  doc: LatentDoc
  inScope: boolean
}

export function linksPane(
  focus: LatentPreview,
  items: LinkItem[],
  kinds: LatentKind[],
  label = 'links',
): Element {
  const docs = new Set(items.map(i => i.doc.d))
  const groups: ElementContent[] = []
  for (const kind of kinds) {
    const own = items.filter(i => i.preview.s === kind)
    if (own.length === 0) continue
    groups.push(
      h('h3.lat-group', [mark(kind), h('span', `${own.length}`)]),
      h(
        'ol.lat-list',
        { dataKind: kind },
        own.map(item => {
          const sameDoc = item.doc.d === focus.d
          const href = rowHref(item.doc, item.preview.p)
          return h(
            'li.lat-link',
            {
              dataRow: String(item.preview.r),
              dataDoc: String(item.doc.d),
              dataOut: item.inScope ? undefined : '',
              dataSame: sameDoc ? '' : undefined,
              dataSha: item.doc.sha,
              dataPage: item.preview.p ? String(item.preview.p) : undefined,
            },
            [
              h('span.lat-score', [
                h('i', { style: `--score:${(item.score / 255).toFixed(3)}` }),
                h('b', (item.score / 255).toFixed(3)),
              ]),
              h(
                'a.lat-link-title',
                { href: `?focus=${item.preview.r}`, dataLatFocus: String(item.preview.r) },
                [item.doc.t],
              ),
              h('span.lat-link-locus', [
                locus(item.preview),
                sameDoc ? h('em', 'same document') : null,
                item.inScope ? null : h('em', 'out of scope'),
              ]),
              item.preview.x ? h('p.lat-extract', extractRuns(item.preview.x)) : null,
              item.preview.s === 'pdf'
                ? h('a.lat-link-read', { href, dataNoPopover: true }, '/read →')
                : null,
              // the map card: a pdf's page boxes, or a note's chunks as lines of text; a click
              // on it opens the row as the title does
              item.preview.s === 'pdf' && item.doc.sha && item.preview.p
                ? h('span.lat-silhouette', {
                    dataSha: item.doc.sha,
                    dataPage: String(item.preview.p),
                    dataLatFocus: String(item.preview.r),
                    ariaHidden: 'true',
                  })
                : item.preview.s === 'note'
                  ? h('span.lat-silhouette', {
                      dataNote: '',
                      dataLatFocus: String(item.preview.r),
                      ariaHidden: 'true',
                    })
                  : null,
            ],
          )
        }),
      ),
    )
  }
  return h('section.lat-links', { dataFocus: String(focus.r) }, [
    h('header.lat-pane-head', [
      h('span', label),
      h('span.lat-count', `${items.length} from ${docs.size} documents`),
    ]),
    ...groups,
  ])
}

export interface RailModel {
  docs: LatentTrackDoc[]
  titles: (d: number) => string
  open: number
  focus: number
  /** neighbour doc → best score */
  neighbours: Map<number, number>
  /** neighbour doc → how many of the focus row's neighbours it holds; ribbon width */
  weights?: Map<number, number>
  focusChunkIndex: number
  scopeFlags?: Uint8Array
  /** viewBox height in rail units; RAIL_HEIGHT unless the client has measured its pane */
  height?: number
}

/** The rail as an inline SVG. Collapsed documents are one path per store. */
export const RAIL_BIN = 4
const BAND_WIDTH = 7

export function railSvg(model: RailModel): Element {
  const height = model.height ?? RAIL_HEIGHT
  const layout = layoutRail(model.docs, height)
  const children: ElementContent[] = []
  children.push(h('line.lat-axis', { x1: TRACK_X, x2: TRACK_X, y1: 10, y2: height - 8 }))
  for (const tick of thinYears(layout.years)) {
    children.push(
      h('text.lat-year', { x: TRACK_X - 5, y: tick.y.toFixed(1) }, String(tick.year)),
      h('line.lat-tick', {
        x1: TRACK_X - 3,
        x2: TRACK_X + 3,
        y1: tick.y.toFixed(1),
        y2: tick.y.toFixed(1),
      }),
    )
  }
  // dated documents: y is rank, so time density is already in the year ticks' spacing; the band
  // shows which kinds a slice of time holds, one thin stacked segment per 4-unit bin
  const bins = Math.ceil(height / RAIL_BIN)
  const counts = new Map<LatentKind, Uint16Array>()
  const undated = new Map<LatentKind, number>()
  const totals = new Uint16Array(bins)
  for (let i = 0; i < model.docs.length; i++) {
    const doc = model.docs[i]
    if (doc.day === 0) {
      undated.set(doc.kind, (undated.get(doc.kind) ?? 0) + 1)
      continue
    }
    let row = counts.get(doc.kind)
    if (!row) {
      row = new Uint16Array(bins)
      counts.set(doc.kind, row)
    }
    const b = Math.min(bins - 1, Math.floor(layout.y[i] / RAIL_BIN))
    row[b]++
    totals[b]++
  }
  const segments = new Map<LatentKind, string[]>()
  for (let b = 0; b < bins; b++) {
    if (!totals[b]) continue
    let x = TRACK_X + 3
    for (const kind of KIND_ORDER) {
      const count = counts.get(kind)?.[b] ?? 0
      if (!count) continue
      const w = (BAND_WIDTH * count) / totals[b]
      const list = segments.get(kind) ?? []
      list.push(`M${x.toFixed(1)} ${b * RAIL_BIN}h${w.toFixed(1)}v${RAIL_BIN}h-${w.toFixed(1)}z`)
      segments.set(kind, list)
      x += w
    }
  }
  for (const [kind, list] of segments) {
    children.push(h('path.lat-density', { dataKind: kind, d: list.join('') }))
  }
  // undated documents: one row per kind, a bar as long as its share and the count
  if (undated.size > 0) {
    const total = [...undated.values()].reduce((a, b) => a + b, 0)
    children.push(
      h('line.lat-tick', {
        x1: TRACK_X - 3,
        x2: TRACK_X + 3,
        y1: layout.undatedTop.toFixed(1),
        y2: layout.undatedTop.toFixed(1),
      }),
      h(
        'text.lat-year.lat-undated',
        { x: TRACK_X + 6, y: layout.undatedTop.toFixed(1) },
        `undated · ${total.toLocaleString('en-US')}`,
      ),
    )
    for (const kind of layout.undatedKinds) {
      const count = undated.get(kind) ?? 0
      const y = layout.undatedTop + UNDATED_FIRST + layout.undatedKinds.indexOf(kind) * UNDATED_ROW
      const w = Math.max(2, (count / total) * 36)
      children.push(
        h('rect.lat-density', {
          dataKind: kind,
          x: TRACK_X + 3,
          y: (y - 2.5).toFixed(1),
          width: w.toFixed(1),
          height: 5,
        }),
        h(
          'text.lat-undated-count',
          { x: (TRACK_X + 3 + w + 4).toFixed(1), y: y.toFixed(1) },
          `${kind} ${count.toLocaleString('en-US')}`,
        ),
      )
    }
  }
  // the open document: a bracket at its slot, its chunk rows fanned to the right. The fan and the
  // neighbour labels stay above the undated band, even when the open document sits in it.
  const floor = undated.size > 0 ? layout.undatedTop - 16 : height - 12
  // undated documents share band rows with the kind counts, so brackets and leaders for them
  // anchor just above the band header instead of crossing the counts
  const anchorY = (i: number) => (model.docs[i].day === 0 ? layout.undatedTop - 8 : layout.y[i])
  const open = model.docs[model.open]
  if (open) {
    const oy = anchorY(model.open)
    const rows = Math.min(open.n, 48)
    const rowH = 3
    const first = Math.max(0, Math.min(open.n - rows, model.focusChunkIndex - Math.floor(rows / 2)))
    // neighbour labels share the fan's column: those above the open slot go above the fan, the
    // rest below it, and the fan leaves each group its rows
    const targets = [...model.neighbours.entries()]
      .filter(([d]) => d !== model.open)
      .sort((a, b) => anchorY(a[0]) - anchorY(b[0]))
    const aboveCount = targets.filter(([d]) => anchorY(d) < oy).length
    const belowCount = targets.length - aboveCount
    const top = Math.max(
      14 + (aboveCount > 0 ? 22 + (aboveCount - 1) * LABEL_ROW : 0),
      Math.min(
        floor - 6 - rows * rowH - (belowCount > 0 ? belowCount * LABEL_ROW : 0),
        oy - (model.focusChunkIndex - first) * rowH,
      ),
    )
    const chunkMarks: string[] = []
    for (let i = 0; i < rows; i++) {
      const y = (top + i * rowH).toFixed(1)
      chunkMarks.push(`M${LEADER_X} ${y}h${Math.min(OPEN_WIDTH, 6 + ((i * 37) % 23))}`)
    }
    const focusY = top + (model.focusChunkIndex - first) * rowH
    children.push(
      h('path.lat-bracket', {
        d: `M${LANE_X[open.kind]} ${oy.toFixed(1)}H${LEADER_X - 6}V${top.toFixed(1)}M${LEADER_X - 6} ${oy.toFixed(1)}V${(top + rows * rowH).toFixed(1)}`,
      }),
      h('path.lat-open', { dataKind: open.kind, d: chunkMarks.join('') }),
      h('line.lat-cursor', {
        x1: LEADER_X - 10,
        x2: RAIL_WIDTH,
        y1: focusY.toFixed(1),
        y2: focusY.toFixed(1),
      }),
      h(
        'text.lat-open-title',
        { x: LEADER_X, y: (top - 4).toFixed(1) },
        truncate(model.titles(model.open), 26),
      ),
    )
    // ribbons to neighbour documents: width is how many of the focus row's neighbours the
    // document holds, opacity its best score against the others; labels de-overlapped
    const fanTop = top - 22
    const fanBottom = top + rows * rowH + 6
    const sources = targets.map(([d]) => anchorY(d))
    const labelY = [
      ...placeLabels(sources.slice(0, aboveCount), LABEL_ROW, 12, fanTop),
      ...placeLabels(sources.slice(aboveCount), LABEL_ROW, fanBottom, floor),
    ]
    let lo = 255
    let hi = 0
    for (const [, score] of targets) {
      lo = Math.min(lo, score)
      hi = Math.max(hi, score)
    }
    targets.forEach(([d, score], i) => {
      const doc = model.docs[d]
      const ty = anchorY(d)
      const ly = labelY[i]
      const out = model.scopeFlags && model.scopeFlags[d] === 0
      const count = model.weights?.get(d) ?? 1
      const width = Math.min(5, 0.6 + 0.9 * (count - 1))
      const opacity = 0.35 + 0.55 * (hi > lo ? (score - lo) / (hi - lo) : 1)
      const title = model.titles(d)
      children.push(
        h('path.lat-leader', {
          dataKind: doc.kind,
          dataOut: out ? '' : undefined,
          strokeWidth: width.toFixed(1),
          opacity: (out ? opacity * 0.5 : opacity).toFixed(2),
          d: `M${LANE_X[doc.kind] + 4} ${ty.toFixed(1)}C${LEADER_X - 20} ${ty.toFixed(1)} ${LEADER_X - 20} ${ly.toFixed(1)} ${LABEL_X - 4} ${ly.toFixed(1)}`,
        }),
        h(
          'text.lat-label',
          {
            x: LABEL_X,
            y: (ly + 3.5).toFixed(1),
            dataDoc: String(d),
            dataKind: doc.kind,
            dataOut: out ? '' : undefined,
          },
          [
            h(
              'title',
              `${title} · ${(score / 255).toFixed(2)}${count > 1 ? ` · ${count} passages` : ''}`,
            ),
            truncate(title, count > 1 ? 21 : 25),
            ...(count > 1 ? [h('tspan.lat-label-count', { dx: 3 }, `×${count}`)] : []),
          ],
        ),
      )
    })
  }
  return h(
    'svg.lat-track',
    {
      viewBox: `0 0 ${RAIL_WIDTH} ${height}`,
      preserveAspectRatio: 'xMinYMin meet',
      role: 'img',
      ariaLabel: 'documents in time order',
    },
    children,
  )
}

/**
 * Hover and focus notes for the probe's terms. The text rides in data-tip and latent-tips.scss
 * draws it from attr(), so it stays out of textContent (the axis search, the page text) and
 * costs no layout until shown; aria-description hands it to screen readers.
 */
function tipProps(note: string | null, at?: 'end' | 'up'): Record<string, string | number> {
  if (!note) return {}
  const props: Record<string, string | number> = {
    tabIndex: 0,
    dataTip: note,
    'aria-description': note,
  }
  if (at) props.dataTipAt = at
  return props
}

function axisNote(axis: LatentAxis): string | null {
  const pc = /^pc(\d+) · ([\d.]+) %$/.exec(axis.name)
  if (pc) {
    const n = Number(pc[1])
    const share = Number(pc[2])
    const spread =
      n === 1
        ? 'the direction of widest spread in the 256-d space'
        : `the widest spread left at right angles to ${n === 2 ? 'pc1' : n === 3 ? 'pc1 and pc2' : `pc1–pc${n - 1}`}`
    // an even split over 256 dimensions gives each 1/256 of the variance
    const ratio = Math.round(share / (100 / 256))
    return `principal component ${n}: ${spread}. it holds ${share} % of the variance, ${ratio}× the share of an average dimension. the sign is arbitrary.`
  }
  if (axis.id === 'store')
    return 'the line from the average note vector to the average pdf-passage vector: whether a passage reads more like a note or like a book.'
  if (axis.id === 'lang')
    return 'the line from the average fr/ note to the average english note: how french a passage reads.'
  return null
}

function bcNote(bc: number, shape: string): string {
  const reading =
    shape === 'flat'
      ? 'one hump, so the axis ranks passages without splitting them'
      : shape === 'skewed'
        ? 'one hump with a heavy tail or shoulder'
        : 'two humps, so the axis splits the space in two'
  return `bimodality coefficient, (skew² + 1) / kurtosis, of where every passage falls on this axis. a bell scores 0.33, an even spread 0.555, two clusters more. ${bc.toFixed(2)}: ${reading}.`
}

function poleNote(axis: LatentAxis, side: 'a' | 'b'): string {
  const end = /^pc\d+$/.test(axis.id)
    ? `the ${side === 'a' ? 'low' : 'high'} end of ${axis.id}`
    : `the ${side === 'a' ? axis.a : axis.b} end`
  return `${end}. italics: the words most over-represented among the 200 passages furthest this way. rows: the three furthest; click one to read it.`
}

const SOURCES_NOTE =
  'every store, embedded by one encoder into one space. counts are rows, one vector each: a note chunk, a pdf passage, an image, a base entry. documents count the wholes.'

function modelNote(index: LatentIndex, model: string, scoped: boolean): string {
  const scope = scoped
    ? ' the scope picks the opening passage and greys links that leave it; nothing outside is dropped.'
    : ''
  return `${model} wrote every vector at ${index.dims} dimensions; the probe and the 3-d view read the first 256 (matryoshka). each passage keeps its ${index.k} nearest by cosine, at most 4 from its own document.${scope}`
}

export function axisCard(axis: LatentAxis, focusValue: number | null): Element {
  const flat = axis.bc < 0.45 ? 'flat' : axis.bc < 0.555 ? 'skewed' : 'two modes'
  const pole = (side: 'a' | 'b') =>
    h('div.lat-pole', { dataSide: side }, [
      h(
        'span.lat-pole-name',
        tipProps(poleNote(axis, side), side === 'b' ? 'end' : undefined),
        side === 'a' ? axis.a : axis.b,
      ),
      h('span.lat-pole-terms', axis.terms[side].slice(0, 4).join('\u00a0· ')),
      h(
        'ol.lat-pole-rows',
        axis.poles[side]
          .slice(0, 3)
          .map(p =>
            h('li', [
              mark(p.s),
              h('a', { href: `?focus=${p.r}`, dataLatFocus: String(p.r) }, truncate(p.t, 44)),
            ]),
          ),
      ),
    ])
  const max = Math.max(1, ...axis.hist)
  const bars = axis.hist
    .map((v, i) => `M${(i * 4).toFixed(0)} 16V${(16 - (v / max) * 16).toFixed(1)}`)
    .join('')
  return h('section.lat-axis', { dataAxis: axis.id }, [
    h('header', [
      h('span.lat-axis-name', tipProps(axisNote(axis)), axis.name),
      h(
        'span.lat-chip',
        { dataFlat: flat === 'flat' ? '' : undefined, ...tipProps(bcNote(axis.bc, flat), 'end') },
        `bc ${axis.bc.toFixed(2)} · ${flat}`,
      ),
    ]),
    h('svg.lat-hist', { viewBox: '0 0 128 18', preserveAspectRatio: 'none', ariaHidden: 'true' }, [
      h('path', { d: bars }),
      focusValue === null
        ? null
        : h('line.lat-focus-tick', {
            x1: (64 + (focusValue / 127) * 64).toFixed(1),
            x2: (64 + (focusValue / 127) * 64).toFixed(1),
            y1: 0,
            y2: 18,
          }),
    ]),
    h('div.lat-poles', [pole('a'), pole('b')]),
  ])
}

const count = (n: number) => n.toLocaleString('en-US')

/**
 * The stores behind the space, one row each: store and rows. An excluded store says why in
 * place of a count.
 */
export function ledger(index: LatentIndex, scope: string[]): Element {
  const rows: ElementContent[] = index.stores.map(store => {
    const name = store.id.replace(/^embeddings\/?/, '') || 'notes'
    return h('li', { dataIncluded: store.included ? '' : undefined }, [
      h('span.lat-ledger-id', name),
      h('span.lat-ledger-rows', store.included ? count(store.rows ?? 0) : 'excluded'),
      store.included ? null : h('span.lat-ledger-detail', store.reason ?? ''),
    ])
  })
  const model = index.model.replace(/^[^/]+\//, '')
  return h('section.lat-ledger', [
    h('header.lat-pane-head', [
      h('span', tipProps(SOURCES_NOTE), 'sources'),
      h('span.lat-count', `${count(index.docs)} documents`),
    ]),
    h('ul', rows),
    h('p.lat-ledger-note', tipProps(modelNote(index, model, scope.length > 0), 'up'), [
      `${model}, ${index.dims} dimensions, ${index.k} neighbours/passage`,
      scope.length > 0 ? `; scoped to ${scope.join(', ')}` : '',
    ]),
  ])
}

export function scopeFlagsFor(docs: LatentDoc[], prefixes: string[]): Uint8Array {
  const flags = new Uint8Array(docs.length)
  for (const doc of docs) flags[doc.d] = inScope(doc.id, prefixes) ? 1 : 0
  return flags
}
