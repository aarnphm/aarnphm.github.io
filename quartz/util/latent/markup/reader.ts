import type { Element } from 'hast'
import { h } from 'hastscript'
import type { LatentChunk, LatentDoc, LatentText } from '../types'
import { folioToggle, pdfSource } from '../pdf'
import { titleRuns, titleText } from '../title'
import { rowHref } from '../types'
import { mark, pageLabel, passageBlocks, DISPLAY_MATH } from './passage'

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

// OCR sets each line of a display equation as a paragraph, and the pdf store cuts passages at
// paragraphs, so one equation can arrive as several rows: `$$`, the array in pieces, `\tag …
// \end{array}`, `$$`. A row that leaves `$$` open reads together with the rows up to the one that
// closes it. JOIN_MAX bounds the run; past it, the dollars were text.
const JOIN_MAX = 8
const fences = (text: string) => text.split('$$').length - 1

/** Index of the last chunk read with `chunks[i]`; `i` when it stands alone. */
function joinedRun(chunks: LatentChunk[], texts: string[], i: number): number {
  let open = fences(texts[i])
  if (chunks[i].same || open % 2 === 0) return i
  let j = i
  while (open % 2 === 1) {
    j += 1
    const next = chunks[j]
    if (
      !next ||
      j - i >= JOIN_MAX ||
      next.same ||
      next.m === 1 ||
      next.j !== undefined ||
      next.r !== chunks[j - 1].r + 1
    )
      return i
    open += fences(texts[j])
  }
  // a lone `$`, a table or a heading inside any of the run's display math means the dollars
  // pair up wrong (text that happens to use them), so the run stays as rows
  for (const [, tex] of texts
    .slice(i, j + 1)
    .join('\n\n')
    .matchAll(DISPLAY_MATH))
    if (/(?<!\\)\$|<t[dhr]\b|\n#/.test(tex)) return i
  return j
}

// The pdf chunker carries the end of a passage into the start of the next (a heading, a line, or
// half an equation row), so two rows read in order repeat that seam. OVERLAP_MIN keeps a shared
// `$$` or a word from counting.
const OVERLAP_MIN = 12

/** A pdf row's text without the start it repeats from the row before. */
function ownText(prev: LatentChunk | undefined, chunk: LatentChunk): string {
  const text = chunk.text ?? ''
  const before = prev?.text ?? ''
  if (!prev || prev.same || chunk.same || prev.m === 1 || prev.r + 1 !== chunk.r) return text
  const probe = text.length < OVERLAP_MIN ? '\u0000' : text.slice(0, OVERLAP_MIN)
  // the earliest place the row's start recurs in the tail of the one before is the longest seam
  for (
    let at = before.indexOf(probe, Math.max(0, before.length - text.length));
    at >= 0;
    at = before.indexOf(probe, at + 1)
  ) {
    if (text.startsWith(before.slice(at))) {
      const own = text.slice(before.length - at)
      return own.trim() ? own.replace(/^\s+/, '') : text
    }
  }
  // a short seam (`Hence,`) counts when it is a whole paragraph on both sides
  const cut = text.indexOf('\n\n')
  const head = cut > 0 ? text.slice(0, cut) : ''
  if (head.trim() && before.endsWith(`\n\n${head}`)) return text.slice(cut).replace(/^\s+/, '')
  return text
}

/** The page box around a run's rows on its first page (a lone `$$` row has none). */
function runBox(run: LatentChunk[]): LatentChunk['box'] {
  const boxes = run.filter(c => c.box && c.p === run[0].p).map(c => c.box!)
  if (boxes.length === 0) return undefined
  return [
    Math.min(...boxes.map(b => b[0])),
    Math.min(...boxes.map(b => b[1])),
    Math.max(...boxes.map(b => b[2])),
    Math.max(...boxes.map(b => b[3])),
  ]
}

function chunkLocus(
  meta: Pick<LatentText, 'store'>,
  doc: Pick<LatentDoc, 'id' | 's'>,
  chunk: LatentChunk,
  last: LatentChunk,
  box?: LatentChunk['box'],
): Element {
  if (meta.store === 'pdf' && chunk.p) {
    const to =
      last === chunk
        ? ''
        : last.p === chunk.p
          ? `\u2013${last.c + 1}`
          : `\u2013p. ${pageLabel(last.l, last.p ?? chunk.p)} ¶${last.c + 1}`
    return h('span.lat-locus', [
      h(
        'a',
        { href: rowHref(doc, chunk.p), dataNoPopover: true },
        `p. ${pageLabel(chunk.l, chunk.p)} ¶${chunk.c + 1}${to}`,
      ),
      // the page itself, in the site's pdf viewer under the passage (latent-client/folio.ts)
      doc.s === 'pdf'
        ? folioToggle(pdfSource(doc.id), chunk.p, box, `p. ${pageLabel(chunk.l, chunk.p)}`)
        : null,
    ])
  }
  return h('span.lat-locus', [
    h(
      'a',
      { href: `?focus=${chunk.r}`, dataLatFocus: String(chunk.r), dataNoPopover: true },
      `§${chunk.c + 1}`,
    ),
  ])
}

/** An arXiv slice and the passages right after it whose text it shows (`j` naming it). */
function sourceRun(chunks: LatentChunk[], i: number): number {
  let j = i
  while (chunks[j + 1]?.j === chunks[i].r && chunks[j + 1].r === chunks[j].r + 1) j += 1
  return j
}

const joinedMarker = (c: LatentChunk, to: number): Element =>
  h('article.lat-chunk.lat-joined', {
    id: `lat-r${c.r}`,
    dataRow: String(c.r),
    dataPage: c.p ? String(c.p) : undefined,
    dataJoined: String(to),
  })

/**
 * `article.lat-chunk` for each chunk; `chunk.c` is its index in the document. A run of pdf rows
 * holding one display equation (see `joinedRun`) is one article on its first row; each row after
 * it keeps an empty `article.lat-joined` marker (`data-joined` names the first row), so focus,
 * the rail and goTo still find every row. An arXiv paper's passage read from its TeX source
 * (`m`) is markdown, as a note's slice is; a passage whose text another slice shows (`j`) is a
 * marker joined to that slice's row, wherever it sits.
 */
export function chunkArticles(
  meta: Pick<LatentText, 'store'>,
  doc: Pick<LatentDoc, 'id' | 's'>,
  chunks: LatentChunk[],
  focus: number,
): Element[] {
  const pdf = meta.store === 'pdf'
  const texts = chunks.map((chunk, i) =>
    pdf && chunk.m !== 1 ? ownText(chunks[i - 1], chunk) : (chunk.text ?? ''),
  )
  const out: Element[] = []
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i]
    if (chunk.j !== undefined) {
      out.push(joinedMarker(chunk, chunk.j))
      continue
    }
    const j = !pdf ? i : chunk.m === 1 ? sourceRun(chunks, i) : joinedRun(chunks, texts, i)
    const run = chunks.slice(i, j + 1)
    out.push(
      h(
        // a pdf passage is set like its paper (latent-folio.scss); its hyphenation is English,
        // as the pdf shelf is, not the page's language
        pdf ? 'article.lat-chunk.lat-pdf' : 'article.lat-chunk',
        {
          id: `lat-r${chunk.r}`,
          lang: pdf ? 'en' : undefined,
          dataRow: String(chunk.r),
          dataPage: chunk.p ? String(chunk.p) : undefined,
          dataBox: runBox(run)?.join(','),
          dataFocus: run.some(c => c.r === focus) ? '' : undefined,
        },
        [
          chunkLocus(meta, doc, chunk, run[run.length - 1], runBox(run)),
          // an image row is the image: its embedding has no text to show
          ...(meta.store === 'image'
            ? [imageFrame(doc)]
            : chunk.same
              ? [samePointer(chunk.same)]
              : passageBlocks(
                  (chunk.m === 1 ? texts.slice(i, i + 1) : texts.slice(i, j + 1)).join('\n\n'),
                  {
                    code: meta.store === 'note',
                    markdown: chunk.m === 1,
                    lines: meta.store === 'note' && chunk.m !== 1,
                    page: meta.store === 'note' ? rowHref(doc) : undefined,
                    seed: doc.id,
                  },
                )),
        ],
      ),
      ...run.slice(1).map(c => joinedMarker(c, chunk.r)),
    )
    i = j
  }
  return out
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
      h(
        'a.lat-doc-title',
        { href, dataNoPopover: true, title: titleText(doc.t) },
        titleRuns(doc.t),
      ),
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
