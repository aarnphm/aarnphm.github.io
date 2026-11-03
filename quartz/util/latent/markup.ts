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
  LABEL_ROW,
  LABEL_X,
  LANE_X,
  LEADER_X,
  OPEN_WIDTH,
  RAIL_HEIGHT,
  RAIL_WIDTH,
  TRACK_X,
  layoutRail,
  placeLabels,
  thinYears,
  truncate,
} from './rail'
import { inScope, rowHref } from './types'

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

export function locus(preview: Pick<LatentPreview, 'c' | 'p' | 'l' | 's'>, total?: number): string {
  // `c` counts from 0 (per page for pdfs, per document for notes); both read from 1
  if (preview.s === 'pdf' && preview.p)
    return `p. ${preview.l || preview.p}${total ? ` of ${total}` : ''} ¶${preview.c + 1}`
  if (preview.s === 'note') return total ? `§${preview.c + 1} of ${total}` : `§${preview.c + 1}`
  return ''
}

// the embedding text keeps markdown headings and html-escaped entities from the source; hast
// escapes again on output, so decode the five xml entities and drop heading hashes for display
const ENTITY: Record<string, string> = { amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'" }

function decodeEntities(text: string): string {
  return text.replace(/&(amp|lt|gt|quot|#39);/g, (_, name: string) => ENTITY[name])
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
  let text = decodeEntities(x).replace(/(^|\s)#{1,6}\s+/g, '$1')
  const display = text.split('$$').length - 1
  if (display % 2 === 1) text = `${text.slice(0, text.lastIndexOf('$$')).trimEnd()} …`
  text = text.replace(DISPLAY_MATH, (_, tex: string) => `$${tex.trim().replace(/\s+/g, ' ')}$`)
  const dollars = text.replace(/\\\$/g, '').split('$').length - 1
  if (dollars % 2 === 1) text = `${text.slice(0, text.lastIndexOf('$')).trimEnd()} …`
  return inlineRuns(text)
}

function paragraphs(text: string): Element[] {
  return text
    .split(/\n[ \t]*\n+/)
    .map(block => block.trim())
    .filter(Boolean)
    .map(block => {
      const heading = /^#{1,6}\s+(.+)$/.exec(block)
      return heading && !block.includes('\n')
        ? h('p.lat-h', inlineRuns(heading[1]))
        : h('p', inlineRuns(block))
    })
}

/**
 * A passage as blocks: display math on its own, blank lines split the rest into paragraphs, and
 * a one-line `# heading` becomes a bold paragraph (a p, so the heading plugins and the toc never
 * pick it up).
 */
function passageBlocks(text: string): Element[] {
  const source = decodeEntities(text)
  const blocks: Element[] = []
  let last = 0
  for (const match of source.matchAll(DISPLAY_MATH)) {
    blocks.push(...paragraphs(source.slice(last, match.index)))
    if (match[1].trim()) blocks.push(h('div.math.math-display', match[1].trim()))
    last = match.index + match[0].length
  }
  blocks.push(...paragraphs(source.slice(last)))
  return blocks.length > 0 ? blocks : [h('p.lat-empty', 'no text for this row')]
}

export type ReaderMeta = Pick<LatentText, 'd' | 'store' | 'total' | 'part'>

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
                `p. ${chunk.l || chunk.p} ¶${chunk.c + 1}`,
              )
            : `§${chunk.c + 1}`,
        ]),
        ...passageBlocks(chunk.text ?? ''),
      ],
    ),
  )
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

/**
 * One document in the reader: a sticky head and a run of chunk articles. `window` is the loaded
 * run (`chunks` starting at chunk index `offset`); a stub renders the head only. The run is a
 * whole number of parts so the client can grow it part by part.
 */
export function readerDocument(
  text: LatentText,
  doc: Pick<LatentDoc, 'id' | 's' | 't' | 'date' | 'n' | 'pages'>,
  focus: number,
  options: { stub?: boolean; offset?: number } = {},
): Element {
  const href = rowHref(doc)
  const unit = doc.s === 'pdf' ? 'passages' : doc.s === 'note' ? 'chunks' : 'rows'
  const head = h('header.lat-doc-head', [
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

export function linksPane(focus: LatentPreview, items: LinkItem[], kinds: LatentKind[]): Element {
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
              item.preview.s === 'pdf' && item.doc.sha && item.preview.p
                ? h('span.lat-silhouette', {
                    dataSha: item.doc.sha,
                    dataPage: String(item.preview.p),
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
      h('span', 'links'),
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
  focusChunkIndex: number
  scopeFlags?: Uint8Array
}

/** The rail as an inline SVG. Collapsed documents are one path per store. */
export function railSvg(model: RailModel): Element {
  const layout = layoutRail(model.docs)
  const lanes = new Map<LatentKind, string[]>()
  for (let i = 0; i < model.docs.length; i++) {
    if (i === model.open) continue
    const doc = model.docs[i]
    const x = LANE_X[doc.kind]
    const y = layout.y[i].toFixed(1)
    const list = lanes.get(doc.kind) ?? []
    list.push(`M${x - 4} ${y}h8`)
    lanes.set(doc.kind, list)
  }
  const children: ElementContent[] = []
  children.push(h('line.lat-axis', { x1: TRACK_X, x2: TRACK_X, y1: 10, y2: RAIL_HEIGHT - 8 }))
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
  if (layout.datedCount < model.docs.length) {
    children.push(
      h(
        'text.lat-year.lat-undated',
        { x: TRACK_X - 5, y: (layout.undatedTop + 10).toFixed(1) },
        'undated',
      ),
      h('line.lat-tick', {
        x1: TRACK_X - 3,
        x2: RAIL_WIDTH,
        y1: layout.undatedTop.toFixed(1),
        y2: layout.undatedTop.toFixed(1),
        dataBand: '',
      }),
    )
  }
  for (const [kind, segments] of lanes) {
    children.push(h('path.lat-lane', { dataKind: kind, d: segments.join('') }))
  }
  // the open document: a bracket at its slot, its chunk rows fanned to the right
  const open = model.docs[model.open]
  if (open) {
    const oy = layout.y[model.open]
    const rows = Math.min(open.n, 48)
    const rowH = 3
    const first = Math.max(0, Math.min(open.n - rows, model.focusChunkIndex - Math.floor(rows / 2)))
    const top = Math.max(
      14,
      Math.min(RAIL_HEIGHT - 20 - rows * rowH, oy - (model.focusChunkIndex - first) * rowH),
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
    // leaders to neighbour documents, labels de-overlapped
    const targets = [...model.neighbours.entries()]
      .filter(([d]) => d !== model.open)
      .sort((a, b) => layout.y[a[0]] - layout.y[b[0]])
    const sources = targets.map(([d]) => layout.y[d])
    const labelY = placeLabels(sources, LABEL_ROW, 12, RAIL_HEIGHT - 12)
    targets.forEach(([d, score], i) => {
      const doc = model.docs[d]
      const ty = layout.y[d]
      const ly = labelY[i]
      const out = model.scopeFlags && model.scopeFlags[d] === 0
      children.push(
        h('path.lat-leader', {
          dataKind: doc.kind,
          dataOut: out ? '' : undefined,
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
            truncate(model.titles(d), 22),
            h('tspan.lat-label-score', { dx: 4 }, (score / 255).toFixed(2)),
          ],
        ),
      )
    })
  }
  return h(
    'svg.lat-track',
    {
      viewBox: `0 0 ${RAIL_WIDTH} ${RAIL_HEIGHT}`,
      preserveAspectRatio: 'xMinYMin meet',
      role: 'img',
      ariaLabel: 'documents in time order',
    },
    children,
  )
}

export function axisCard(axis: LatentAxis, focusValue: number | null): Element {
  const flat = axis.bc < 0.45 ? 'flat' : axis.bc < 0.555 ? 'skewed' : 'two modes'
  const pole = (side: 'a' | 'b') =>
    h('div.lat-pole', { dataSide: side }, [
      h('span.lat-pole-name', side === 'a' ? axis.a : axis.b),
      h('span.lat-pole-terms', axis.terms[side].slice(0, 4).join(' · ')),
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
      h('span.lat-axis-name', axis.name),
      h(
        'span.lat-chip',
        {
          dataFlat: flat === 'flat' ? '' : undefined,
          title: 'bimodality coefficient: above 0.555 the axis splits the space in two',
        },
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
 * The stores behind the space, one row each: kind, store, rows. A pdf store adds its coverage
 * underneath; an excluded store says why in place of a count.
 */
export function ledger(index: LatentIndex, scope: string[]): Element {
  const rows: ElementContent[] = index.stores.map(store => {
    const name = store.id.replace(/^embeddings\/?/, '') || 'notes'
    const coverage = store.included && store.coverage
    return h('li', { dataIncluded: store.included ? '' : undefined }, [
      store.kind ? mark(store.kind) : h('span'),
      h('span.lat-ledger-id', name),
      h('span.lat-ledger-rows', store.included ? count(store.rows ?? 0) : 'excluded'),
      coverage
        ? h(
            'span.lat-ledger-detail',
            `${count(coverage.indexedDocuments)} of ${count(coverage.eligibleDocuments)} pdfs, ${count(coverage.indexedPages)} pages`,
          )
        : store.included
          ? null
          : h('span.lat-ledger-detail', store.reason ?? ''),
    ])
  })
  const model = index.model.replace(/^[^/]+\//, '')
  return h('section.lat-ledger', [
    h('header.lat-pane-head', [
      h('span', 'ledger'),
      h('span.lat-count', `${count(index.docs)} documents`),
    ]),
    h('ul', rows),
    h('p.lat-ledger-note', [
      `${model}, ${index.dims} dimensions, ${index.k} neighbours a passage`,
      scope.length > 0 ? `; scoped to ${scope.join(', ')}` : '',
    ]),
  ])
}

export function scopeFlagsFor(docs: LatentDoc[], prefixes: string[]): Uint8Array {
  const flags = new Uint8Array(docs.length)
  for (const doc of docs) flags[doc.d] = inScope(doc.id, prefixes) ? 1 : 0
  return flags
}
