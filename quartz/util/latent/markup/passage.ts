import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import type { LatentKind, LatentPreview } from '../types'
import { dropcapSpecimen } from '../../dropcap'
import { poetryChildren } from '../../poetry'
import { telescopeHast, telescopicContent } from '../../telescopic'
import {
  type HtmlOptions,
  blockSpans,
  decodeEntities,
  fenceBlock,
  htmlRuns,
  htmlTable,
  pipeTable,
  repairCuts,
} from '../html'
import {
  type MdHooks,
  codeText,
  embedHref,
  embedName,
  mdBlocks,
  mdInline,
  mdLink,
  mdSource,
} from '../md'
import { pseudocodeBlock } from '../pseudo'
import { svgDrawings } from '../svg'
import { displayTex, inlineTex } from '../tex'

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
export function pageLabel(label: string | undefined, page: number): string {
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
export const DISPLAY_MATH = /\$\$([\s\S]+?)\$\$/g
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
    out.push(
      match[1] !== undefined
        ? h('span.math.math-inline', inlineTex(match[1]))
        : h('strong', match[2]),
    )
    last = match.index + match[0].length
  }
  if (last < text.length) out.push({ type: 'text', value: text.slice(last) })
  return out
}

// text between tags: entities decoded (hast escapes again on output), then math and strong
const leaf = (text: string) => inlineRuns(decodeEntities(text))

// a table cell is one line, so display math folds inline there; OCR pads a cell's math as
// `$ \|F\|_{F}$`, which the inline rule reads as a price, so a dollar followed by space and a
// control word opens math
const cellLeaf = (text: string) =>
  inlineRuns(
    decodeEntities(text)
      .replace(DISPLAY_MATH, (_, tex: string) => `$${tex.trim().replace(/\s+/g, ' ')}$`)
      .replace(/(^|[^\\$])\$\s+(?=\\[a-zA-Z|{])/g, '$1$$'),
  )

interface PassageOptions extends HtmlOptions {
  /** the chunk is the note's markdown slice (`m` in the text sidecar) */
  markdown?: boolean
  /** the chunk is a note's rendered text: every line is a paragraph */
  lines?: boolean
  /** the note's page (`/slug`): its tikz figures are fetched from there */
  page?: string
  /** the note's slug, the seed of a `procedural` dropcap without its own */
  seed?: string
}

const mathSpan = (tex: string, display: boolean) =>
  h(display ? 'span.math.math-display' : 'span.math.math-inline', tex)
const mdLeaf = (text: string) => mdInline(decodeEntities(text), mathSpan)
const mdCellLeaf = (text: string) => mdInline(decodeEntities(text), tex => mathSpan(tex, false))

// An inline svg pictogram inside a sentence is a picture, not text (a drawing on its own lines is
// a figure block, see `figure` below); nested svgs count, so a drop never stops at an inner close.
// `{{phrase}}` marks a phrase to hear in the French notes and reads as the phrase (TeX's `{{` sits
// after a letter, `^`, `_` or a backslash, and holds no `$`, so it is left alone).
const SVG_TAG = /<(\/?)svg\b[^>]*?(\/?)>/g
const MARKED = /(?<![\\\w^_{])\{\{([^{}\n$\\]+)\}\}(?!\})/g
function dropSvg(text: string): string {
  let out = ''
  let depth = 0
  let last = 0
  for (const m of text.matchAll(SVG_TAG)) {
    if (depth === 0) out += text.slice(last, m.index)
    if (m[1]) depth = Math.max(0, depth - 1)
    else if (!m[2]) depth++
    last = m.index + m[0].length
  }
  return depth === 0 ? out + text.slice(last) : out
}
const gardenText = (text: string) => dropSvg(text).replace(MARKED, '$1')

// Backtick spans in a markdown slice are code, but they sit inside emphasis
// (`**Parse tree for `x`:**`), so they are held out as placeholders while the html allowlist and
// mdInline run over the rest, and put back as `code` after: html inside code stays text and the
// strong run still spans the code.
// A link whose text holds html (`[<span style=…>Space Grotesk</span>](https://…)`) is held the
// same way: the allowlist would cut its text at the tags and leave `[` and `](url)` as text.
const CODE_SPAN = /(`+)(?!`)([^\n]*?[^`\n])\1(?!`)/g
const HTML_LINK =
  /(?<!!)\[((?:[^[\]\n]|\[[^[\]\n]*\])*<[a-zA-Z](?:[^[\]\n]|\[[^[\]\n]*\])*)\]\(\s*<?([^()\s<>]*(?:\([^()\s]*\)[^()\s<>]*)*)>?(?:\s+"[^"\n]*")?\s*\)/g
const HELD = /(\d+)/g
function withCode(text: string, runs: (masked: string) => ElementContent[]): ElementContent[] {
  const held: ElementContent[][] = []
  const hold = (nodes: ElementContent[]) => {
    held.push(nodes)
    return `${held.length - 1}`
  }
  const restore = (nodes: ElementContent[]): ElementContent[] =>
    nodes.flatMap(node => {
      if (node.type === 'element') {
        node.children = restore(node.children as ElementContent[])
        return [node]
      }
      if (node.type !== 'text' || !node.value.includes('')) return [node]
      const out: ElementContent[] = []
      let last = 0
      for (const m of node.value.matchAll(HELD)) {
        if (m.index > last) out.push({ type: 'text', value: node.value.slice(last, m.index) })
        out.push(...(held[Number(m[1])] ?? []))
        last = m.index + m[0].length
      }
      if (last < node.value.length) out.push({ type: 'text', value: node.value.slice(last) })
      return out
    })
  const masked = text
    .replace(CODE_SPAN, (_, _ticks: string, body: string) =>
      hold([h('code', decodeEntities(codeText(body)))]),
    )
    .replace(HTML_LINK, (_, label: string, href: string) =>
      hold(mdLink(decodeEntities(href), restore(runs(label)))),
    )
  if (!held.length) return runs(text)
  return restore(runs(masked))
}

/**
 * A drawing a note writes as html (nltk parse trees under `<figure class="svg-diagram">`): the
 * svg rebuilt through ./svg's allowlist, labelled like the note page's figure. A block with no
 * drawing in it (a figure around an image) drops, as images do.
 */
function figureBlock(html: string, inline: MdHooks['inline']): Element[] {
  const paper = paperFigure(html, inline)
  if (paper) return paper
  const drawings = svgDrawings(html)
  if (!drawings.length) return []
  const label = /<figure\b[^>]*?\baria-label\s*=\s*"([^"]*)"/.exec(html)?.[1]
  return [
    h(
      'figure.lat-figure',
      { role: 'img', ariaLabel: label ? decodeEntities(label) : undefined },
      drawings,
    ),
  ]
}

// An arXiv paper's figure (quartz/scripts/latent/arxiv.py writes it, one line): the source's
// graphics, drawn at build under /embeddings/latent/fig/, and the caption as markdown. Only that
// exact shape and folder pass; any other figure html goes through the svg allowlist.
const PAPER_FIGURE =
  /^<figure class="lat-paper-figure">((?:<img [^<>]*>)*)<figcaption>([\s\S]*?)<\/figcaption><\/figure>\s*$/
const PAPER_IMAGE =
  /<img src="(\/embeddings\/latent\/fig\/[\w.-]+\/[\w.-]+\.(?:svg|webp))" alt="">/g

function paperFigure(html: string, inline: MdHooks['inline']): Element[] | null {
  const m = PAPER_FIGURE.exec(html.trim())
  if (!m) return null
  const plates = [...m[1].matchAll(PAPER_IMAGE)].map(image =>
    h('img', { src: image[1], alt: '', loading: 'lazy', decoding: 'async' }),
  )
  return [
    h('figure.lat-paper-figure', [
      plates.length
        ? h('div.lat-paper-plates', { dataCount: String(plates.length) }, plates)
        : null,
      h('figcaption', inline(m[2])),
    ]),
  ]
}

const mdHooks = (options: PassageOptions): MdHooks => {
  const runs = { ...options, code: false }
  const inline: MdHooks['inline'] = text =>
    withCode(gardenText(text), masked => htmlRuns(masked, mdLeaf, runs))
  return {
    inline,
    pipe: block =>
      pipeTable(block, cell =>
        withCode(gardenText(cell), masked => htmlRuns(masked, mdCellLeaf, runs)),
      ),
    display: tex => h('div.math.math-display', decodeEntities(tex)),
    fence: (body, info) => fenceView(body, info, options),
    figure: html => figureBlock(html, inline),
    triathlon: count => triathlonCard(count, options.page),
    signature: () => signature(options.page),
    embed: (href, alias) => embed(href, alias, options.page),
  }
}

const TRACK_LINE = /^[ \t]*([\w-]+):[ \t]*(.*)$/
// eslint-disable-next-line no-control-regex
const ANSI = new RegExp(`${String.fromCharCode(27)}\\[[0-?]*[ -/]*[@-~]`, 'g')

/**
 * A fence in a note's markdown slice. Garden components render as the page shows them: `quotes`
 * is a quote whose last paragraph is its attribution, `poetry` and `poem` keep their line breaks,
 * `sms` is a message, `tracking` (the triathlon log) one metadata line, `pseudo` an algorithm
 * the way the note page sets it, `telescopic` the page's telescopic text, `dropcap` its specimen
 * and `tikz` its figure. Anything else is code, as is every fence outside a markdown slice.
 */
function fenceView(body: string, info: string | undefined, options: PassageOptions): Element {
  const [kind = '', name = ''] = (info ?? '').toLowerCase().split(/\s+/)
  if (!options.markdown) return fenceBlock(body, info)
  // a notebook cell's output (the builder writes `output stdout|stderr|result|error`), set as
  // the notebook page sets it: mono on its own tint, the stream named above it
  if (kind === 'output')
    return h('pre.lat-code.lat-output', { dataOutputName: name || undefined }, [
      h('samp', decodeEntities(body.replace(/\n+$/, ''))),
    ])
  // the note page's pseudocode block (`.ps-root`); source the library cannot parse stays code
  if (kind === 'pseudo') {
    const algorithm = pseudocodeBlock(decodeEntities(body), info)
    if (algorithm) return algorithm
  }
  // a drop-cap specimen, as the note page's Profile transformer sets one
  if (kind === 'dropcap') {
    const specimen = dropcapSpecimen(body, info ?? '', options.seed ?? '')
    if (specimen) return specimen
  }
  if (kind === 'tikz') return tikzFigure(decodeEntities(body), info ?? '', options.page)
  const hooks = mdHooks(options)
  // telescopic text (telescopic.inline.ts on the page opens it; links arrive as markdown)
  if (kind === 'telescopic')
    return h('div.telescopic-container', [
      telescopeHast(telescopicContent(decodeEntities(body)), text =>
        text ? hooks.inline(text) : [],
      ),
    ])
  // the Aarnphm transformer's fences (transformers/aarnphm/codeblocks.ts), in its markup so the
  // site's styles set them: a `quotes` block is a blockquote.quotes of markdown whose last
  // paragraph is the attribution (quotes.scss opens and closes the quote marks and dashes the
  // attribution), `poetry` is pre.poetry, `sms` the italic p.text between rules
  const trimmed = body.replace(/^\n+|\s+$/g, '')
  if (kind === 'quotes' && trimmed) return h('blockquote.quotes', dashes(mdBlocks(trimmed, hooks)))
  if (kind === 'poetry' || kind === 'poem')
    return h('pre.poetry', { dataCodeblock: kind }, poetryChildren(decodeEntities(trimmed)))
  if (kind === 'sms')
    return h('p.text', { dataCodeblock: 'sms' }, hooks.inline(trimmed.replace(/\s+/g, ' ')))
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
      // a field's value can link (`wahoo [26-09-06.fit](/triathlon/wahoo/…)`)
      ...rest.flatMap((part, n) => [...(date || n > 0 ? [' · '] : []), ...hooks.inline(part)]),
    ])
  }
  return fenceBlock(body, info)
}

// quoteDisplayDashes: a quote's `--` reads as a dash in its text, never in a link's address
function dashes(nodes: ElementContent[]): ElementContent[] {
  for (const node of nodes)
    if (node.type === 'text') node.value = node.value.replace(/--/g, '\u2014')
    else if (node.type === 'element') dashes(node.children)
  return nodes
}

/**
 * A tikz fence. The build typesets tikz on the note page (TikzJax); the reader holds the source
 * and the client swaps in that page's rendered figure the first time the passage shows. A fence
 * carrying its own picture (`alt="data:image/svg+xml;base64,…"`) shows it at once.
 */
function tikzFigure(source: string, info: string, page: string | undefined): Element {
  const alt = /\balt\s*=\s*"(data:image\/svg\+xml;base64,[A-Za-z0-9+/=]+)"/.exec(info)?.[1]
  if (alt)
    return h('figure.tikz', { dataRemarkTikz: '' }, [
      h('img', { src: alt, alt: 'tikz diagram', loading: 'lazy', decoding: 'async' }),
    ])
  return h('figure.tikz.lat-tikz', { dataPage: page }, [
    h('pre.lat-code.lat-tikz-source', [h('code', source.replace(/\n+$/, ''))]),
  ])
}

/**
 * A note's latest-activities card. The build fills the note page's marker from the Strava store
 * (quartz/components/strava-activities.ts); the reader holds the link the page shows without
 * synced data, and the client swaps in the page's card the first time the passage nears the
 * screen. `count` picks that card when a page has more than one.
 */
function triathlonCard(count: string | undefined, page: string | undefined): Element {
  return h('div.lat-triathlon', { dataPage: page, dataCount: count }, [
    h('a.internal', { href: '/triathlon' }, 'latest activities in the training log'),
  ])
}

/**
 * A note's handwritten signature (the Aarnphm transformer draws `[^sign]` from its letter banks,
 * in the text the page's frontmatter names). The reader holds an empty line of the signature's
 * height and the client copies the page's drawing in the first time the passage nears the screen.
 */
function signature(page: string | undefined): Element {
  return h('p.signature.lat-signature', { dataPage: page })
}

const IMAGE = /\.(?:avif|gif|jpe?g|png|svg|webp)$/i
const VIDEO = /\.(?:mp4|webm|mov)$/i
const AUDIO = /\.(?:m4a|mp3|wav|ogg|flac)$/i

/**
 * An embed (`![[…]]`, its target resolved by the builder). An image, a video or a voice memo is drawn here as
 * the note page draws one (ofm.ts: a captioned image is `figure[data-img-w-caption]`, `|300` or
 * `|300x200` sizes it). Anything else is what the note page transcluded there: the reader holds a
 * link to it and the client copies in the page's `blockquote.transclude` for it.
 */
function embed(raw: string, alias: string | undefined, page: string | undefined): Element {
  const href = embedHref(raw)
  // a web embed (a YouTube video, a tweet) reads as its link
  if (/^https?:\/\//.test(href))
    return h('p', mdLink(href, [{ type: 'text', value: embedName(href, alias) }]))
  const path = href.split('#')[0]
  const label = alias?.trim() ?? ''
  const size = /^(\d+)(?:x(\d+))?$/.exec(label)
  if (IMAGE.test(path)) {
    const img = h('img', {
      src: path,
      alt: size ? '' : label,
      width: size?.[1],
      height: size?.[2],
      loading: 'lazy',
      decoding: 'async',
    })
    return label && !size
      ? h('figure', { dataImgWCaption: 'true' }, [
          img,
          h('figcaption', [h('span.figure-caption', label)]),
        ])
      : h('p.lat-embed-media', [img])
  }
  if (AUDIO.test(path))
    return h('p.lat-embed-media', [h('audio', { src: path, controls: true, preload: 'none' })])
  if (VIDEO.test(path))
    return h('p.lat-embed-media', [
      h('video', { src: path, controls: true, preload: 'metadata', width: size?.[1] }),
    ])
  return h('div.lat-embed', { dataPage: page, dataEmbed: href }, [
    h('a.internal', { href }, embedName(href, alias)),
  ])
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
  if (options.markdown) return mdSource(source, mdHooks(options))
  const blocks: Element[] = []
  let last = 0
  for (const match of source.matchAll(DISPLAY_MATH)) {
    blocks.push(...paragraphs(source.slice(last, match.index), options))
    const tex = decodeEntities(match[1]).trim()
    if (tex) blocks.push(h('div.math.math-display', displayTex(tex)))
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
export function passageBlocks(raw: string, options: PassageOptions = {}): Element[] {
  // pdf passages are cut by length, sometimes inside a tag; notebook output keeps its ANSI colours
  const text = (options.code ? raw : repairCuts(raw)).replace(ANSI, '')
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
