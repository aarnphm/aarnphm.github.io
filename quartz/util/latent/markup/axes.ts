import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import type { LatentAxis, LatentAxisPole, LatentDoc, LatentIndex, LatentKind } from '../types'
import { kindsBar } from '../document-links'
import { truncate } from '../rail'
import { titleText } from '../title'
import { inScope } from '../types'
import { mark } from './passage'

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
  return `${end}. italics: the words most over-represented among the 200 passages furthest this way, at most 8 from one document. rows: the furthest ${POLE_ROWS} documents, one passage each, with the furthest notes this way after them; click one to read it.`
}

const POLE_ROWS = 5
const POLE_NOTES = 2
const EDITION =
  / (?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) ed(?:ition)?\b.*$/

/** build.py `title_key`: two editions, or a notebook copied into eleven folders, read as one */
const titleKey = (title: string) =>
  titleText(title)
    .toLowerCase()
    .replace(/[^0-9a-z]+/g, ' ')
    .trim()
    .replace(EDITION, '')

/** The furthest documents of one end, then notes until POLE_NOTES of the rows are notes. Sidecars
 * built before build.py deduplicated the ends list a document many times, so this drops repeats. */
function poleExemplars(
  axis: LatentAxis,
  side: 'a' | 'b',
): { pole: LatentAxisPole; pick: boolean }[] {
  const docs = new Set<number>()
  const titles = new Set<string>()
  // only rows that are shown claim their document, so a note past the cut can still be picked
  const take = (list: LatentAxisPole[], n: number) => {
    const out: LatentAxisPole[] = []
    for (const p of list) {
      if (out.length >= n) break
      const key = titleKey(p.t)
      // a book's base entry and its pdf are one work; a note titled 'linear algebra' is not the book
      const t = key && p.s === 'note' ? `note\t${key}` : key
      if (docs.has(p.d) || (t && titles.has(t))) continue
      docs.add(p.d)
      if (t) titles.add(t)
      out.push(p)
    }
    return out
  }
  const top = take(axis.poles[side], POLE_ROWS)
  const want = POLE_NOTES - top.filter(p => p.s === 'note').length
  const extra = want > 0 ? take(axis.notes?.[side] ?? [], want) : []
  // room for the notes comes off the inner end of the list, never from a note already in it
  for (let i = top.length - 1; i >= 0 && top.length + extra.length > POLE_ROWS; i--) {
    if (top[i].s !== 'note') top.splice(i, 1)
  }
  return [...top.map(pole => ({ pole, pick: false })), ...extra.map(pole => ({ pole, pick: true }))]
}

const SOURCES_NOTE =
  'every store, embedded by one encoder into one space. counts are rows, one vector each: a note chunk, a pdf passage, an image, a base entry. documents count the wholes.'

// the circuit and selection tabs' transcoders, as models/latent-plt/latest.json (v1) states them;
// a retrain that changes the shape or the validation FVU updates these
const TRANSCODERS = { model: 'gemma 4 e4b', layers: 42, width: 8192, k: 64 }
const TRANSCODER_NOTE = [
  'Gemma 4 E4B · 4-bit · mlx',
  'skip transcoder per layer: MLP + PLE',
  '42 layers · d 2560',
  '8192 features · top 64',
  'val FVU 0.32 · worst layer 0.54',
  'v1 · trained on this sidecar',
].join('\n')

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
        poleExemplars(axis, side).map(({ pole: p, pick }) =>
          h('li', { dataPick: pick ? 'note' : undefined }, [
            mark(p.s),
            h(
              'a',
              { href: `?focus=${p.r}`, dataLatFocus: String(p.r) },
              truncate(titleText(p.t), 44),
            ),
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

const axisLabel = (axis: LatentAxis) =>
  /^pc\d+$/.test(axis.id) ? axis.id : axis.name.replace(/\s*↔\s*/g, '↔')

/**
 * The axis cards under the 3-d view: one card a row, scrolled sideways and snapped, pc1 first
 * and the named axes (store, frontmatter) after the principal ones. `data-axis-index` keeps the
 * probe.json order, which the kNN blocks' per-row projections follow (search.ts focus ticks).
 * The pager names every card; latent-client/axes-carousel.ts scrolls to a pick and marks the
 * card in view.
 */
export function axesCarousel(axes: LatentAxis[], focusValues: number[]): Element {
  const order = axes
    .map((axis, i) => ({ axis, i }))
    .sort((x, y) => Number(!/^pc\d+$/.test(x.axis.id)) - Number(!/^pc\d+$/.test(y.axis.id)))
  return h('div.lat-axes', [
    h(
      'nav.lat-axes-pager',
      { ariaLabel: 'axes' },
      order.map(({ axis }, n) =>
        h(
          'button',
          { type: 'button', dataAxis: axis.id, ariaCurrent: n === 0 ? 'true' : undefined },
          axisLabel(axis),
        ),
      ),
    ),
    h(
      'div.lat-axes-track',
      { tabindex: 0, ariaLabel: 'probe axes, scroll sideways' },
      order.map(({ axis, i }) => {
        const card = axisCard(axis, focusValues[i] ?? null)
        card.properties.dataAxisIndex = String(i)
        return card
      }),
    ),
  ])
}

const count = (n: number) => n.toLocaleString('en-US')

/**
 * The stores behind the space, one row each: store and rows. An excluded store says why in
 * place of a count.
 */
export function ledger(index: LatentIndex, scope: string[], shown?: LatentKind[]): Element {
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
    shown ? kindsBar(index.kinds, shown) : null,
    h('p.lat-ledger-note', tipProps(modelNote(index, model, scope.length > 0), 'up'), [
      `${model}, ${index.dims} dimensions, ${index.k} neighbours/passage`,
      scope.length > 0 ? `; scoped to ${scope.join(', ')}` : '',
    ]),
    h(
      'p.lat-ledger-note',
      { ...tipProps(TRANSCODER_NOTE, 'up'), dataTipWide: '' },
      `${TRANSCODERS.model} transcoders, ${TRANSCODERS.layers} × ${count(TRANSCODERS.width)}, top ${TRANSCODERS.k}`,
    ),
  ])
}

export function scopeFlagsFor(docs: LatentDoc[], prefixes: string[]): Uint8Array {
  const flags = new Uint8Array(docs.length)
  for (const doc of docs) flags[doc.d] = inScope(doc.id, prefixes) ? 1 : 0
  return flags
}
