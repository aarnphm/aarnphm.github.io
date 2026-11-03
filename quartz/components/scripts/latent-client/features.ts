import type { Element, ElementContent } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import { linksPane, mark, type LinkItem } from '../../../util/latent/markup'
import { truncate } from '../../../util/latent/rail'
import { inScope, type LatentKind } from '../../../util/latent/types'
import {
  docOfRow,
  fetchBytes,
  fetchJson,
  knnOf,
  metaOf,
  previewOf,
  requestVec,
  type Runtime,
} from './runtime'
import { renderLinks, renderRail } from './search'

/**
 * Sparse features of the latent space (Prism's move): quartz/scripts/latent/features.py trains a
 * Matryoshka TopK sparse autoencoder per family of rows (text passages, images, base entries) and
 * writes `feat/<name>/`; label.py names the features with a local Gemma. The probe's features
 * tab lists the features the focused chunk fires, a click pins a feature's strongest rows to the
 * links pane and opens its card (ancestry, children, notes over time), ± steers the focus vector
 * along a feature's decoder direction and asks for its neighbours there, and a typed query is
 * encoded into features in the browser. The tag probes (`feat/tags.json`) add the tags a document
 * reads as but does not carry. Nothing here loads until the tab opens or the search box is used.
 *
 * Steering is score fusion (docs/latent-steering.md): a row ranks by its cosine to the base plus
 * Σ weight × channel, where a feature's channel is its detector (relu of the encoder
 * pre-activation over the max, which finds rows that fire the feature far better than the decoder
 * direction) and a tag's or an axis's is the projection on its difference-in-means direction. The
 * worker keeps the channels, so a slider drag is one pass over cached arrays. A term's unit comes
 * from `feat/steer.json`: the weight at which half the steered links carry it.
 */

interface FeatureDictMeta {
  name: string
  kinds: LatentKind[]
  features: number
  k: number
  dims: number
  rows: number
  trained: number
  fvu: number
  /** Matryoshka prefixes: features [0, prefixes[0]) are level 0, and so on */
  prefixes?: number[]
  levels?: { prefix: number; fvu: number }[]
  /** rows per codes/ block */
  codeBlock: number
  /** dated notes per month (yyyymm) */
  months?: Record<string, number>
  norm: number
  mean: number[]
  bias: number[]
  encScale: number[]
  decScale: number[]
}

interface FeatureRow {
  f: number
  /** a written name from latent/labels.json, when one exists */
  label?: string
  /** label.py's one-sentence description */
  about?: string
  /** detection score of the label: balanced accuracy on held-out rows, 0.5 = chance */
  score?: number
  /** Matryoshka level (0 = coarsest) and the coarser feature that fires with it most */
  level?: number
  parent?: number
  share?: number
  terms: string[]
  n: number
  /** firing rows by kind: note, pdf, image, base */
  kinds: [number, number, number, number]
  docs: number
  max: number
  near: [number, number][]
  /** [row, activation / max × 255] */
  top: [number, number][]
  cross: [number, number][]
  /** dated notes that fire it, by month: [yyyymm, notes] */
  time?: [number, number][]
}

interface CodeBlock {
  ids: Uint16Array
  acts: Uint8Array
}

interface FeatureDict {
  meta: FeatureDictMeta
  table: FeatureRow[]
  children: Map<number, number[]>
  blocks: Map<number, Promise<CodeBlock>>
  enc: Int8Array | null
  dec: Int8Array | null
}

interface TagProbes {
  tags: { t: string; n: number; auc: number }[]
  /** [doc, [[tag index, p × 100]]] */
  pred: [number, [number, number][]][]
}

/** feat/steer.json (quartz/scripts/latent/steering.py) */
interface SteerTables {
  topn: number
  grid: number
  /** per dictionary, per feature: [unit, share of links that fire it at one unit, weak] */
  dicts: Record<string, [number, number, number][]>
  tags: {
    t: string
    dir: number[]
    unit: number
    p: number
    weak: number
    /** the 5-feature sparse probe: [text feature, coefficient] */
    explain: [number, number][]
    auc5: number
    auc: number
  }[]
  axes: { id: string; name: string; unit: number }[]
}

/** feat/text/words.json (quartz/scripts/latent/words.py): words that make a feature fire */
interface WordTable {
  features: Record<string, { w: [string, number][]; r: [number, [string, number][]][] }>
}

type SteerTerm =
  | { kind: 'feature'; dict: string; f: number; w: number }
  | { kind: 'tag'; id: string; w: number }
  | { kind: 'axis'; id: string; w: number }

interface FeatureState {
  names: string[]
  dicts: Map<string, Promise<FeatureDict>>
  tags: Promise<Map<number, [string, number][]> | null> | null
  selected: { dict: string; f: number } | null
  query: { text: string; vector: Float32Array } | null
  needle: string
  /** the steering set: term key (`text:12`, `tag:philosophy`, `axis:fm1`) → signed units */
  steer: Map<string, SteerTerm>
  steerSeq: number
  tables: Promise<SteerTables | null> | null
  tablesNow: SteerTables | null
  words: Promise<WordTable | null> | null
  /** channel keys the worker holds for the rows matrix */
  channels: Set<string>
  /** the "why these links" group is open */
  explain: boolean
  /** the rows steering last pinned to the links pane, which "why these links" explains */
  pinned: number[] | null
  dragFrame: number | null
  panel: HTMLElement
}

/** units when feat/steer.json is missing: the medians of the first calibration */
const DEFAULT_UNIT = { feature: 0.45, tag: 0.6, axis: 1 }
const RANGE = 2

const KIND_ORDER: LatentKind[] = ['note', 'pdf', 'image', 'base']
const states = new WeakMap<Runtime, FeatureState>()

function loadDict(rt: Runtime, st: FeatureState, name: string): Promise<FeatureDict> {
  let dict = st.dicts.get(name)
  if (!dict) {
    const base = `${rt.base}/feat/${name}`
    dict = Promise.all([
      fetchJson<FeatureDictMeta>(rt, `${base}/meta.json`),
      fetchJson<FeatureRow[]>(rt, `${base}/features.json`),
    ]).then(([meta, table]) => {
      const children = new Map<number, number[]>()
      for (const row of table) {
        if (row.parent === undefined || row.parent < 0) continue
        const list = children.get(row.parent) ?? []
        list.push(row.f)
        children.set(row.parent, list)
      }
      for (const list of children.values()) list.sort((a, b) => table[b].n - table[a].n)
      return { meta, table, children, blocks: new Map(), enc: null, dec: null }
    })
    dict.catch(() => st.dicts.delete(name))
    st.dicts.set(name, dict)
  }
  return dict
}

async function weights(rt: Runtime, dict: FeatureDict, which: 'enc' | 'dec'): Promise<Int8Array> {
  const have = dict[which]
  if (have) return have
  const bytes = new Int8Array(
    await fetchBytes(rt, `${rt.base}/feat/${dict.meta.name}/${which}-int8.bin`),
  )
  dict[which] = bytes
  return bytes
}

/** codes/NNNNN.bin: a block of rows, ids (u16) then activations (u8), fetched on first use. */
function codeBlock(rt: Runtime, dict: FeatureDict, b: number): Promise<CodeBlock> {
  let block = dict.blocks.get(b)
  if (!block) {
    const { k, rows, codeBlock: size } = dict.meta
    const n = Math.min(size, rows - b * size) * k
    block = fetchBytes(
      rt,
      `${rt.base}/feat/${dict.meta.name}/codes/${String(b).padStart(5, '0')}.bin`,
    ).then(bytes => ({ ids: new Uint16Array(bytes, 0, n), acts: new Uint8Array(bytes, n * 2, n) }))
    block.catch(() => dict.blocks.delete(b))
    dict.blocks.set(b, block)
  }
  return block
}

/** The features the row fires in one dictionary, strongest first: [feature, activation 0..1]. */
async function codesOf(rt: Runtime, dict: FeatureDict, row: number): Promise<[number, number][]> {
  const { k, codeBlock: size } = dict.meta
  const block = await codeBlock(rt, dict, Math.floor(row / size))
  const at = (row % size) * k
  const out: [number, number][] = []
  for (let j = 0; j < k; j++) {
    const a = block.acts[at + j]
    if (a > 0) out.push([block.ids[at + j], a / 255])
  }
  return out
}

function tagsOf(rt: Runtime, st: FeatureState): Promise<Map<number, [string, number][]> | null> {
  st.tags ??= fetchJson<TagProbes>(rt, `${rt.base}/feat/tags.json`)
    .then(
      probes =>
        new Map(
          probes.pred.map(([d, got]) => [
            d,
            got.map(([j, p]) => [probes.tags[j].t, p] as [string, number]),
          ]),
        ),
    )
    .catch(() => null)
  return st.tags
}

function steerTables(rt: Runtime, st: FeatureState): Promise<SteerTables | null> {
  st.tables ??= fetchJson<SteerTables>(rt, `${rt.base}/feat/steer.json`)
    .then(t => (st.tablesNow = t))
    .catch(() => null)
  return st.tables
}

function wordsOf(rt: Runtime, st: FeatureState): Promise<WordTable | null> {
  st.words ??= fetchJson<WordTable>(rt, `${rt.base}/feat/text/words.json`).catch(() => null)
  return st.words
}

/** Encode a unit 256-d vector into the dictionary's features, as the build did: top-k of z. */
async function encodeVector(
  rt: Runtime,
  dict: FeatureDict,
  vector: Float32Array,
): Promise<[number, number][]> {
  const enc = await weights(rt, dict, 'enc')
  const { features: M, dims: D, k, mean, norm, bias, encScale } = dict.meta
  const x = new Float32Array(D)
  for (let d = 0; d < D; d++) x[d] = (vector[d] - mean[d]) / norm
  const z: [number, number][] = []
  for (let f = 0; f < M; f++) {
    let s = 0
    const off = f * D
    for (let d = 0; d < D; d++) s += enc[off + d] * x[d]
    const v = s * encScale[f] + bias[f]
    if (v > 0) z.push([f, v])
  }
  z.sort((a, b) => b[1] - a[1])
  return z.slice(0, k).map(([f, v]) => [f, Math.min(1, v / (dict.table[f].max || 1))])
}

// ---- markup --------------------------------------------------------------------------------

function featureName(row: FeatureRow): string {
  if (row.label) return row.label
  return row.terms.length ? row.terms.slice(0, 3).join(' · ') : `#${row.f}`
}

function share(dict: FeatureDict, row: FeatureRow): string {
  const own = dict.meta.kinds.reduce((sum, kind) => sum + row.kinds[KIND_ORDER.indexOf(kind)], 0)
  const pct = (own / Math.max(1, dict.meta.trained)) * 100
  return pct >= 1 ? `${pct.toFixed(1)} %` : `${pct.toFixed(2)} %`
}

function featureItem(
  dict: FeatureDict,
  f: number,
  a: number,
  selected: boolean,
  w = 0,
  weak = false,
): Element {
  const row = dict.table[f]
  return h(
    'li.lat-feat',
    {
      dataDict: dict.meta.name,
      dataF: String(f),
      dataSelected: selected ? '' : undefined,
      dataSteered: w > 0 ? 'up' : w < 0 ? 'down' : undefined,
      dataWeak: weak ? '' : undefined,
    },
    [
      h('span.lat-feat-bar', { ariaHidden: 'true' }, [h('i', { style: `--a:${a.toFixed(3)}` })]),
      h(
        'button.lat-feat-name',
        { type: 'button', title: row.about ?? (row.terms.join(' · ') || undefined) },
        featureName(row),
      ),
      h(
        'span.lat-feat-meta',
        { dataLevel: row.level === undefined ? undefined : String(row.level) },
        `${dict.meta.name === 'text' ? '' : `${dict.meta.name} `}#${f}`,
      ),
      steerPair(weak),
    ],
  )
}

function steerPair(weak: boolean): Element {
  const note = weak ? ' (a weak steerer: under half the links carry it at full strength)' : ''
  return h('span.lat-feat-steer', [
    h('button', { type: 'button', dataSteer: '1', title: `steer toward it${note}` }, '+'),
    h('button', { type: 'button', dataSteer: '-1', title: `steer away from it${note}` }, '−'),
  ])
}

function isWeak(st: FeatureState, dict: string, f: number): boolean {
  return Boolean(st.tablesNow?.dicts[dict]?.[f]?.[2])
}

function featureList(
  rows: { dict: FeatureDict; f: number; a: number }[],
  st: FeatureState,
): Element {
  return h(
    'ol.lat-feat-list',
    rows.map(({ dict, f, a }) =>
      featureItem(
        dict,
        f,
        a,
        st.selected?.dict === dict.meta.name && st.selected.f === f,
        st.steer.get(steerKey(dict.meta.name, f))?.w ?? 0,
        isWeak(st, dict.meta.name, f),
      ),
    ),
  )
}

function section(label: string, count: string, body: ElementContent[]): Element {
  return h('section.lat-feat-group', [
    h('header', [h('span.lat-feat-head', label), h('span.lat-count', count)]),
    ...body,
  ])
}

function chip(dict: FeatureDict, g: number, title?: string): Element {
  return h(
    'button.lat-feat-chip',
    { type: 'button', dataDict: dict.meta.name, dataF: String(g), title },
    featureName(dict.table[g]),
  )
}

/** Matryoshka ancestry, coarsest first: the parents' chain up to level 0. */
function ancestry(dict: FeatureDict, f: number): number[] {
  const out: number[] = []
  let p = dict.table[f].parent ?? -1
  while (p >= 0 && out.length < 4) {
    out.unshift(p)
    p = dict.table[p].parent ?? -1
  }
  return out
}

/**
 * Notes over time: per quarter, the share of that quarter's dated notes that fire the feature,
 * from the first quarter with a few notes. A bar per quarter, so a topic's arrival and fade read
 * at a glance.
 */
function timeline(dict: FeatureDict, row: FeatureRow): Element | null {
  const months = dict.meta.months
  if (!months || !row.time?.length) return null
  const quarter = (ym: number) => Math.floor(ym / 100) * 4 + Math.floor(((ym % 100) - 1) / 3)
  const totals = new Map<number, number>()
  for (const [ym, n] of Object.entries(months))
    totals.set(quarter(Number(ym)), (totals.get(quarter(Number(ym))) ?? 0) + n)
  const live = [...totals.entries()].filter(([, n]) => n >= 5).map(([q]) => q)
  if (live.length < 2) return null
  const q0 = Math.min(...live)
  const q1 = Math.max(...totals.keys())
  const hits = new Map<number, number>()
  for (const [ym, n] of row.time) hits.set(quarter(ym), (hits.get(quarter(ym)) ?? 0) + n)
  const shares: number[] = []
  for (let q = q0; q <= q1; q++) shares.push((hits.get(q) ?? 0) / Math.max(1, totals.get(q) ?? 0))
  const top = Math.max(...shares) || 1
  const W = 7
  const H = 28
  const bars = shares.map((s, i) => {
    const y = Math.round((s / top) * (H - 2))
    const q = q0 + i
    return h(
      'rect',
      {
        x: i * W,
        y: H - Math.max(1, y),
        width: W - 1,
        height: Math.max(1, y),
        dataEmpty: s === 0 ? '' : undefined,
      },
      [h('title', `${Math.floor(q / 4)} q${(q % 4) + 1}: ${(s * 100).toFixed(0)} % of notes`)],
    )
  })
  const notes = row.time.reduce((sum, [, n]) => sum + n, 0)
  return h('div.lat-feat-time', [
    h('span', 'notes'),
    h(
      'svg',
      {
        viewBox: `0 0 ${shares.length * W} ${H}`,
        width: shares.length * W,
        height: H,
        role: 'img',
        ariaLabel: `share of notes per quarter firing ${featureName(row)}`,
      },
      bars,
    ),
    h('span.lat-feat-span', `${Math.floor(q0 / 4)}–${Math.floor(q1 / 4)} · ${notes} notes`),
  ])
}

async function featureCard(
  rt: Runtime,
  st: FeatureState,
  dict: FeatureDict,
  f: number,
): Promise<Element> {
  const row = dict.table[f]
  const up = ancestry(dict, f)
  const kids = dict.children.get(f) ?? []
  const [cross, words] = await Promise.all([
    Promise.all(row.cross.slice(0, 4).map(([r]) => previewOf(rt, r))),
    dict.meta.name === 'text' ? wordsOf(rt, st) : Promise.resolve(null),
  ])
  const unit = st.tablesNow?.dicts[dict.meta.name]?.[f]
  const said = words?.features[String(f)]
  const inTags = (dict.meta.name === 'text' ? (st.tablesNow?.tags ?? []) : []).filter(t =>
    t.explain.some(([g]) => g === f),
  )
  const kinds = KIND_ORDER.map((kind, i) =>
    row.kinds[i] ? h('span', [mark(kind), ` ${row.kinds[i].toLocaleString('en-US')}`]) : null,
  ).filter(Boolean) as Element[]
  return h('section.lat-feat-card', { dataDict: dict.meta.name, dataF: String(f) }, [
    h('header', [
      h('span.lat-feat-head', featureName(row)),
      h(
        'span.lat-chip',
        `${dict.meta.name} #${f}${row.level === undefined ? '' : ` · L${row.level}`} · ${share(dict, row)}`,
      ),
    ]),
    up.length
      ? h('p.lat-feat-path', [
          h('span', 'in'),
          ...up.map((g, i) =>
            chip(
              dict,
              g,
              i === up.length - 1 && row.share
                ? `fires in ${(row.share * 100).toFixed(0)} % of this feature's rows`
                : undefined,
            ),
          ),
        ])
      : null,
    row.about ? h('p.lat-feat-about', row.about) : null,
    h('p.lat-feat-terms', [
      row.terms.join(' · '),
      row.score === undefined
        ? null
        : h(
            'span.lat-feat-score',
            {
              dataTip:
                'detection score: a second prompt picks, from 5 held-out rows that fire the feature and 5 that do not, the ones the label fits. 0.5 is chance.',
              dataTipAt: 'up',
              tabIndex: 0,
            },
            ` · label ${row.score.toFixed(2)}`,
          ),
    ]),
    said?.w.length
      ? h(
          'p.lat-feat-near.lat-feat-words',
          {
            dataTip:
              'integrated gradients from the encoder’s input tokens to this feature, on its strongest rows: the words that make it fire, with their share of the attribution',
            dataTipAt: 'up',
            tabIndex: 0,
          },
          [
            h('span', 'words'),
            ...said.w.map(([w, s]) =>
              h('span.lat-feat-word', { style: `--a:${Math.min(1, s * 4).toFixed(2)}` }, w),
            ),
          ],
        )
      : null,
    h('p.lat-feat-kinds', kinds),
    timeline(dict, row),
    unit
      ? h(
          'p.lat-feat-near.lat-feat-unit',
          {
            dataTip:
              'steering ranks rows by cosine plus weight × this feature’s detector; one unit is the weight at which half of a focus’s links fire it',
            dataTipAt: 'up',
            tabIndex: 0,
          },
          [
            h('span', 'steers'),
            h(
              'span',
              unit[2]
                ? `weakly: under half the links fire it at ${st.tablesNow!.grid}`
                : `1 unit = ${unit[0]} · ${Math.round(unit[1] * 100)} % of links fire it`,
            ),
          ],
        )
      : null,
    inTags.length
      ? h('p.lat-feat-near', [
          h('span', 'explains'),
          ...inTags.map(t =>
            h(
              'button.lat-feat-chip',
              {
                type: 'button',
                dataTerm: `tag:${t.t}`,
                dataSteer: '1',
                title: `one of the 5 features of the #${t.t} sparse probe; click to steer toward #${t.t}`,
              },
              `#${t.t}`,
            ),
          ),
        ])
      : null,
    kids.length
      ? h('p.lat-feat-near', [
          h('span', 'finer'),
          ...kids
            .slice(0, 8)
            .map(g => chip(dict, g, `${dict.table[g].n.toLocaleString('en-US')} rows`)),
          kids.length > 8 ? h('span.lat-count', `+${kids.length - 8}`) : null,
        ])
      : null,
    row.near.length
      ? h('p.lat-feat-near', [
          h('span', 'near'),
          ...row.near.map(([g, c]) => chip(dict, g, `decoder cosine ${c.toFixed(2)}`)),
        ])
      : null,
    cross.length
      ? h('div.lat-feat-cross', [
          h('span', dict.meta.name === 'text' ? 'also lights' : 'in text'),
          h(
            'ol',
            cross.map(p =>
              h('li', [
                mark(p.s),
                h('a', { href: `?focus=${p.r}`, dataLatFocus: String(p.r) }, truncate(p.t, 40)),
              ]),
            ),
          ),
        ])
      : null,
  ])
}

async function renderPanel(rt: Runtime, st: FeatureState): Promise<void> {
  const [dicts] = await Promise.all([
    Promise.all(st.names.map(name => loadDict(rt, st, name))),
    steerTables(rt, st),
  ])
  if (rt.signal.aborted) return
  const focus = rt.focus
  const groups: Element[] = []
  // the focused chunk: its own family's dictionary first, then what the others read in it
  const kind = docOfRow(rt, focus)?.s
  const ordered = [...dicts].sort(
    (a, b) => Number(!a.meta.kinds.includes(kind)) - Number(!b.meta.kinds.includes(kind)),
  )
  const own = ordered[0]
  const [codes, preview, tags] = await Promise.all([
    Promise.all(ordered.map(dict => codesOf(rt, dict, focus))),
    previewOf(rt, focus),
    tagsOf(rt, st),
  ])
  if (rt.signal.aborted || rt.focus !== focus) return
  const focusRows = codes[0].map(([f, a]) => ({ dict: own, f, a }))
  const read = preview.d >= 0 ? tags?.get(preview.d) : undefined
  groups.push(
    section('focus', `${focusRows.length} of ${own.meta.features.toLocaleString('en-US')}`, [
      h('p.lat-feat-where', truncate(preview.t, 48)),
      featureList(focusRows, st),
      ...(read?.length
        ? [
            h(
              'p.lat-feat-tags',
              {
                dataTip:
                  'tag probes: a logistic probe per frontmatter tag, trained on tagged notes, kept when it ranks held-out notes at AUC 0.85 or better. these are tags this document reads as and does not carry.',
                dataTipAt: 'up',
                tabIndex: 0,
              },
              [
                h('span', 'reads as'),
                ...read.map(([t, p]) =>
                  h('span.lat-feat-tag', { title: tagGloss(st, dicts, t) }, `#${t} ${p}`),
                ),
              ],
            ),
          ]
        : []),
    ]),
  )
  for (const [i, other] of ordered.slice(1).entries()) {
    const read = codes[i + 1].slice(0, 3).map(([f, a]) => ({ dict: other, f, a }))
    if (read.length)
      groups.push(section(`as ${other.meta.name}`, `${read.length}`, [featureList(read, st)]))
  }
  if (st.query) {
    const text = dicts.find(d => d.meta.name === 'text') ?? own
    const rows = (await encodeVector(rt, text, st.query.vector)).map(([f, a]) => ({
      dict: text,
      f,
      a,
    }))
    if (rt.signal.aborted) return
    groups.push(
      section(`query · ${truncate(st.query.text, 28)}`, `${rows.length}`, [featureList(rows, st)]),
    )
  }
  const why = await whySection(rt, st, dicts)
  if (rt.signal.aborted) return
  if (why) groups.splice(1, 0, why)
  if (st.needle) {
    const hits = findFeatures(dicts, st.needle)
    const terms = await findTerms(rt, st, st.needle)
    if (terms.length)
      groups.unshift(section('tags & axes', `${terms.length}`, [termList(st, dicts, terms)]))
    groups.unshift(
      section(`# ${st.needle}`, hits.length ? `${hits.length}` : 'no feature', [
        featureList(
          hits.slice(0, 12).map(({ dict, f, score }) => ({ dict, f, a: score })),
          st,
        ),
      ]),
    )
  }
  if (st.selected) {
    const dict = dicts.find(d => d.meta.name === st.selected!.dict)
    if (dict) groups.unshift(await featureCard(rt, st, dict, st.selected.f))
  }
  const steering = await steeringSection(rt, st, dicts)
  if (steering) groups.unshift(steering)
  // variance rebuilt by the coarsest prefix → the whole dictionary
  const fvu = dicts
    .map(d => {
      const levels = d.meta.levels ?? [{ prefix: d.meta.features, fvu: d.meta.fvu }]
      const pct = (x: number) => (100 * (1 - x)).toFixed(0)
      return levels.length > 1
        ? `${d.meta.name} ${pct(levels[0].fvu)}→${pct(levels[levels.length - 1].fvu)} %`
        : `${d.meta.name} ${pct(levels[0].fvu)} %`
    })
    .join(' · ')
  groups.push(
    h(
      'p.lat-feat-note',
      {
        dataTip:
          'a Matryoshka sparse autoencoder per family of rows splits each 256-d vector into k of M learned directions, the first few of which must rebuild it alone. the percentages are the variance the coarsest prefix and the whole dictionary rebuild.',
        dataTipAt: 'up',
        tabIndex: 0,
      },
      `sparse dictionaries · ${fvu}`,
    ),
  )
  if (rt.signal.aborted || (rt.focus !== focus && !st.selected)) return
  st.panel.innerHTML = toHtml(groups as never)
}

// ---- links pane ----------------------------------------------------------------------------

async function pinLinks(rt: Runtime, label: string, rows: [number, number][]): Promise<void> {
  const previews = await Promise.all(rows.map(([r]) => previewOf(rt, r)))
  if (rt.signal.aborted) return
  const items: LinkItem[] = previews.map((preview, i) => {
    const doc = rt.docs[preview.d]
    return { preview, score: rows[i][1], doc, inScope: inScope(doc.id, rt.scope) }
  })
  // the pinned list survives focus changes, like a typed query; Escape in the box clears it
  rt.query = { text: label, items }
  const pane = linksPane(
    { r: -1, d: -1, s: 'note', t: label, c: 0, p: null, l: '', x: '' },
    items,
    rt.kinds,
    label,
  )
  const body = rt.links.querySelector<HTMLElement>('.lat-links-body') ?? rt.links
  body.innerHTML = toHtml(pane.children as never, { allowDangerousHtml: false })
  rt.links.scrollTop = 0
  rt.links.dataset.query = label
  renderRail(rt, items)
}

/**
 * `#` search: a feature by id (`#99`, `#image 4`, `#base:12`) or by its label, description and
 * terms. A whole-word hit in the label ranks above a prefix, a prefix above the description and
 * terms; ties go to the feature that fires more often. `score` is 0..1 for the list's bar.
 */
function findFeatures(
  dicts: FeatureDict[],
  needle: string,
): { dict: FeatureDict; f: number; score: number }[] {
  const id = /^(?:(text|image|base)[\s:]*)?(\d+)$/.exec(needle)
  if (id) {
    const f = Number(id[2])
    return dicts
      .filter(d => (id[1] ? d.meta.name === id[1] : true) && f < d.table.length)
      .map(dict => ({ dict, f, score: 1 }))
  }
  const words = needle.split(/\s+/).filter(Boolean)
  const out: { dict: FeatureDict; f: number; score: number; n: number }[] = []
  for (const dict of dicts)
    for (const row of dict.table) {
      const label = row.label?.toLowerCase() ?? ''
      const about = row.about?.toLowerCase() ?? ''
      let score = 0
      for (const w of words) {
        const whole = new RegExp(`\\b${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`)
        const s = whole.test(label)
          ? 1
          : label.split(/\W+/).some(t => t.startsWith(w))
            ? 0.75
            : row.terms.some(t => t.startsWith(w))
              ? 0.5
              : about.includes(w)
                ? 0.4
                : 0
        if (s === 0) {
          score = 0
          break
        }
        score += s / words.length
      }
      if (score > 0) out.push({ dict, f: row.f, score, n: row.n })
    }
  return out.sort((a, b) => b.score - a.score || b.n - a.n)
}

/**
 * A feature and its neighbourhood: its Matryoshka parents, its strongest children, and the
 * features whose decoder directions lie closest. The links pane gets the feature's own rows at
 * full weight and two rows from each neighbour at reduced weight, so the cluster reads as one
 * field with the feature at its centre.
 */
function cluster(dict: FeatureDict, f: number): number[] {
  const row = dict.table[f]
  const out = [
    ...ancestry(dict, f),
    ...(dict.children.get(f) ?? []).slice(0, 4),
    ...row.near.map(([g]) => g),
  ]
  return [...new Set(out)].filter(g => g !== f)
}

async function selectFeature(
  rt: Runtime,
  st: FeatureState,
  name: string,
  f: number,
  withCluster = false,
) {
  st.selected = { dict: name, f }
  const dict = await loadDict(rt, st, name)
  const row = dict.table[f]
  void renderPanel(rt, st)
  const rows: [number, number][] = [...row.top, ...row.cross.slice(0, 4)]
  if (withCluster) {
    const seen = new Set(rows.map(([r]) => r))
    for (const g of cluster(dict, f))
      for (const [r, a] of dict.table[g].top.slice(0, 2))
        if (!seen.has(r)) {
          seen.add(r)
          rows.push([r, Math.round(a * 0.6)])
        }
  }
  const label = withCluster ? `# ${featureName(row)} + cluster` : `feature · ${featureName(row)}`
  await pinLinks(rt, label, rows)
}

/**
 * Steering, stacked, as score fusion. Each term is a feature, a tag or an axis with a signed
 * strength in its own unit; a row ranks by its cosine to the base (the typed query when there is
 * one, else the focused chunk) plus Σ strength × unit × channel. Renormalising a steered vector
 * never reorders a dot product, so this is the same ranking embedding arithmetic gives along the
 * channel's direction, and a slider only re-mixes arrays the worker already holds. A feature's
 * channel is its detector, relu((x − mean) / norm · enc_f + b_f) / max_f: 75–96 % of the links
 * fire the feature where the decoder direction stalls near 64 %. A negative strength excludes the
 * feature's rows and also pushes away along its decoder direction, so −2 goes further than −1.
 */
function steerKey(name: string, f: number): string {
  return `${name}:${f}`
}

function termOf(key: string, w: number): SteerTerm {
  if (key.startsWith('tag:')) return { kind: 'tag', id: key.slice(4), w }
  if (key.startsWith('axis:')) return { kind: 'axis', id: key.slice(5), w }
  const at = key.lastIndexOf(':')
  return { kind: 'feature', dict: key.slice(0, at), f: Number(key.slice(at + 1)), w }
}

function steerBump(st: FeatureState, key: string, sign: number): void {
  const old = st.steer.get(key)?.w ?? 0
  const w = Math.max(-RANGE, Math.min(RANGE, old + sign))
  // zero is skipped: − at +1 goes to −1 (contrastive, pushes away), + at −1 back to +1; × removes it
  st.steer.set(key, termOf(key, Math.abs(w) < 1e-6 ? sign : w))
}

function unitOf(term: SteerTerm, tables: SteerTables | null): number {
  if (term.kind === 'feature')
    return tables?.dicts[term.dict]?.[term.f]?.[0] ?? DEFAULT_UNIT.feature
  if (term.kind === 'tag') return tables?.tags.find(t => t.t === term.id)?.unit ?? DEFAULT_UNIT.tag
  return tables?.axes.find(a => a.id === term.id)?.unit ?? DEFAULT_UNIT.axis
}

async function termName(
  rt: Runtime,
  term: SteerTerm,
  dicts: Map<string, FeatureDict>,
): Promise<string> {
  if (term.kind === 'tag') return `#${term.id}`
  if (term.kind === 'axis') {
    const axis = (await metaOf(rt)).axes.find(a => a.id === term.id)
    return axis ? axis.name : term.id
  }
  const d = dicts.get(term.dict)
  const name = d ? featureName(d.table[term.f]) : `#${term.f}`
  return term.dict === 'text' ? name : `${name} · ${term.dict}`
}

function amount(w: number): string {
  return `${w > 0 ? '+' : w < 0 ? '−' : ''}${Math.abs(w).toFixed(Math.abs(w) % 1 ? 2 : 0)}`
}

async function steerLabel(rt: Runtime, st: FeatureState, dicts: Map<string, FeatureDict>) {
  const parts = await Promise.all(
    [...st.steer.values()]
      .filter(t => t.w !== 0)
      .map(async t => `${amount(t.w)} ${await termName(rt, t, dicts)}`),
  )
  return parts.join(' ')
}

/** The worker's channel for a term, computed on first use. Returns its key. */
async function ensureChannel(
  rt: Runtime,
  st: FeatureState,
  url: string,
  scale: number,
  key: string,
  make: () => Promise<{ vector: Float32Array; offset: number; gain: number; relu: boolean } | null>,
): Promise<string | null> {
  if (st.channels.has(key)) return key
  const spec = await make()
  if (!spec) return null
  await requestVec(rt, { type: 'channel', url, key, scale, ...spec }, [spec.vector.buffer])
  st.channels.add(key)
  return key
}

async function termChannel(
  rt: Runtime,
  st: FeatureState,
  url: string,
  scale: number,
  term: SteerTerm,
): Promise<string | null> {
  if (term.kind === 'feature') {
    return ensureChannel(rt, st, url, scale, `det:${term.dict}:${term.f}`, async () => {
      const dict = await loadDict(rt, st, term.dict)
      const enc = await weights(rt, dict, 'enc')
      const { dims: D, norm, mean, bias, encScale } = dict.meta
      const vector = new Float32Array(D)
      let offset = bias[term.f]
      for (let d = 0; d < D; d++) {
        vector[d] = (enc[term.f * D + d] * encScale[term.f]) / norm
        offset -= vector[d] * mean[d]
      }
      return { vector, offset, gain: 1 / (dict.table[term.f].max || 1), relu: true }
    })
  }
  if (term.kind === 'tag') {
    return ensureChannel(rt, st, url, scale, `tag:${term.id}`, async () => {
      const tag = (await steerTables(rt, st))?.tags.find(t => t.t === term.id)
      return tag ? { vector: Float32Array.from(tag.dir), offset: 0, gain: 1, relu: false } : null
    })
  }
  return ensureChannel(rt, st, url, scale, `axis:${term.id}`, async () => {
    const axis = (await metaOf(rt)).axes.find(a => a.id === term.id)
    if (!axis) return null
    const vector = Float32Array.from(axis.dir)
    let length = 0
    for (const v of vector) length += v * v
    length = Math.sqrt(length) || 1
    for (let d = 0; d < vector.length; d++) vector[d] /= length
    return { vector, offset: 0, gain: 1, relu: false }
  })
}

/**
 * A feature's decoder direction as a channel, scaled so one unit is the feature's largest
 * activation (the old embedding-arithmetic step): ⟨x, norm · max_f · d_f⟩.
 */
async function decoderChannel(
  rt: Runtime,
  st: FeatureState,
  url: string,
  scale: number,
  name: string,
  f: number,
): Promise<string | null> {
  return ensureChannel(rt, st, url, scale, `dec:${name}:${f}`, async () => {
    const dict = await loadDict(rt, st, name)
    const dec = await weights(rt, dict, 'dec')
    const { dims: D, norm, decScale } = dict.meta
    const vector = new Float32Array(D)
    const gain = norm * decScale[f] * dict.table[f].max
    for (let d = 0; d < D; d++) vector[d] = dec[f * D + d] * gain
    return { vector, offset: 0, gain: 1, relu: false }
  })
}

/** The base: the typed query's cosine, else the focused row's. */
async function baseChannel(
  rt: Runtime,
  st: FeatureState,
  url: string,
  scale: number,
): Promise<string | null> {
  if (st.query) {
    const vector = st.query.vector.slice()
    return ensureChannel(rt, st, url, scale, `query:${st.query.text}`, async () => ({
      vector,
      offset: 0,
      gain: 1,
      relu: false,
    }))
  }
  const focus = rt.focus
  return ensureChannel(rt, st, url, scale, `row:${focus}`, async () => {
    const got = await requestVec(rt, { type: 'row', url, row: focus, scale })
    return got.type === 'row' ? { vector: got.vector, offset: 0, gain: 1, relu: false } : null
  })
}

async function applySteer(rt: Runtime, st: FeatureState, drag = false, retry = true) {
  const seq = ++st.steerSeq
  const live = [...st.steer.values()].filter(t => t.w !== 0)
  if (live.length === 0 && !st.query) {
    // nothing left to steer: back to the focus's own links (a query re-scores unsteered below)
    st.pinned = null
    if (!drag) void renderPanel(rt, st)
    rt.query = null
    delete rt.links.dataset.query
    void renderLinks(rt, rt.focus)
    return
  }
  const meta = await metaOf(rt)
  const url = `${rt.base}/vec/rows-256-int8.bin`
  await requestVec(rt, { type: 'load', url, rows: meta.rows, dims: 256 })
  const tables = await steerTables(rt, st)
  let result
  try {
    const base = await baseChannel(rt, st, url, meta.scale)
    if (!base) return
    const terms = [{ key: base, weight: 1 }]
    for (const term of live) {
      const key = await termChannel(rt, st, url, meta.scale, term)
      if (key) terms.push({ key, weight: term.w * unitOf(term, tables) })
      // a negative feature also pushes away along its decoder direction: the detector alone only
      // drops the rows that fire it, which is done by −1 and leaves −2 looking like −1
      if (term.kind === 'feature' && term.w < 0) {
        const away = await decoderChannel(rt, st, url, meta.scale, term.dict, term.f)
        if (away) terms.push({ key: away, weight: term.w })
      }
    }
    result = await requestVec(rt, {
      type: 'fuse',
      url,
      terms,
      report: base,
      skip: st.query ? [] : [rt.focus],
      k: rt.k,
    })
  } catch (error) {
    // the worker dropped a channel (its cache is bounded): rebuild them once
    if (!retry) throw error
    st.channels.clear()
    return applySteer(rt, st, drag, false)
  }
  if (result.type !== 'fused' || rt.signal.aborted || seq !== st.steerSeq) return
  const rows: [number, number][] = []
  result.hits.forEach((r, i) => rows.push([r, Math.round(Math.max(0, result.scores[i]) * 255)]))
  const dicts = new Map(
    (await Promise.all(st.names.map(n => loadDict(rt, st, n)))).map(d => [d.meta.name, d]),
  )
  const label = live.length
    ? `${st.query ? 'query' : 'focus'} ${await steerLabel(rt, st, dicts)}`
    : st.query!.text
  if (seq !== st.steerSeq) return
  st.pinned = rows.map(([r]) => r)
  await pinLinks(rt, label, rows)
  // after the pin, so "why these links" explains the steered list
  if (!drag) void renderPanel(rt, st)
}

async function steeringSection(
  rt: Runtime,
  st: FeatureState,
  dicts: FeatureDict[],
): Promise<Element | null> {
  if (st.steer.size === 0) return null
  const byName = new Map(dicts.map(d => [d.meta.name, d]))
  const tables = st.tablesNow
  // one row per term: [strength slider] label ×; the last row adds another term
  const rows = await Promise.all(
    [...st.steer.entries()].map(async ([key, term]) => {
      const name = await termName(rt, term, byName)
      const unit = unitOf(term, tables)
      const weak =
        term.kind === 'feature'
          ? isWeak(st, term.dict, term.f)
          : term.kind === 'tag' && Boolean(tables?.tags.find(t => t.t === term.id)?.weak)
      const about =
        term.kind === 'feature'
          ? byName.get(term.dict)?.table[term.f]?.about
          : term.kind === 'tag'
            ? tagGloss(st, dicts, term.id)
            : 'difference-in-means direction between the axis poles; + leans toward the second pole'
      return h(
        'li.lat-steer-row',
        {
          dataTerm: key,
          dataKind: term.kind,
          dataW: term.w.toFixed(2),
          dataWeak: weak ? '' : undefined,
        },
        [
          h('span.lat-steer-amount', [
            h('input', {
              type: 'range',
              min: String(-RANGE),
              max: String(RANGE),
              step: '0.05',
              value: String(term.w),
              ariaLabel: `strength of ${name}`,
              dataSteerRange: '',
            }),
            h('output', amount(term.w)),
          ]),
          h(
            'span.lat-steer-name',
            {
              title: `${about ? `${about}\n` : ''}1 unit = ${unit} × ${term.kind === 'feature' ? 'detector' : 'projection'}${weak ? ' · weak: under half the links carry it at full strength' : ''}`,
            },
            name,
          ),
          h('button', { type: 'button', dataSteerDrop: '', title: 'stop steering by it' }, '×'),
        ],
      )
    }),
  )
  rows.push(
    h('li.lat-steer-row.lat-steer-add', [
      h('span.lat-steer-amount', { ariaHidden: 'true' }),
      h('span.lat-steer-name', 'add: # a feature, tag or axis, or ± below'),
      h(
        'button',
        { type: 'button', dataSteerAdd: '', title: 'search a feature, tag or axis to add' },
        '+',
      ),
    ]),
  )
  return h('section.lat-feat-group.lat-feat-steering', [
    h('header', [
      h('span.lat-feat-head', 'steering'),
      h('button.lat-feat-clear', { type: 'button', dataSteerClear: '' }, 'clear'),
    ]),
    h('ol.lat-steer-rows', rows),
  ])
}

// ---- tags and axes as steering terms ---------------------------------------------------------

/** "#philosophy ≈ metaphysics + aesthetics + …": the tag's 5-feature sparse probe, in words. */
function tagGloss(st: FeatureState, dicts: FeatureDict[], t: string): string | undefined {
  const tag = st.tablesNow?.tags.find(x => x.t === t)
  const text = dicts.find(d => d.meta.name === 'text')
  if (!tag || !text) return undefined
  const names = tag.explain.filter(([, c]) => c > 0).map(([f]) => featureName(text.table[f]))
  return `#${t} ≈ ${names.join(' + ')} (5-feature probe AUC ${tag.auc5.toFixed(2)}, full ${tag.auc.toFixed(2)})`
}

type TermHit = { key: string; name: string; kind: 'tag' | 'axis'; score: number }

async function findTerms(rt: Runtime, st: FeatureState, needle: string): Promise<TermHit[]> {
  const tables = await steerTables(rt, st)
  const words = needle.split(/\s+/).filter(Boolean)
  const hit = (name: string) =>
    words.every(w =>
      name
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .some(p => p.startsWith(w)),
    )
  const out: TermHit[] = []
  for (const tag of tables?.tags ?? [])
    if (hit(tag.t))
      out.push({ key: `tag:${tag.t}`, name: `#${tag.t}`, kind: 'tag', score: tag.auc })
  const axes = (await metaOf(rt)).axes.filter(a => !a.id.startsWith('pc'))
  for (const axis of axes)
    if (hit(axis.name))
      out.push({ key: `axis:${axis.id}`, name: axis.name, kind: 'axis', score: 1 })
  return out.slice(0, 8)
}

function termList(st: FeatureState, dicts: FeatureDict[], hits: TermHit[]): Element {
  const text = dicts.find(d => d.meta.name === 'text')
  return h(
    'ol.lat-feat-list',
    hits.flatMap(hit => {
      const w = st.steer.get(hit.key)?.w ?? 0
      const tag =
        hit.kind === 'tag' ? st.tablesNow?.tags.find(t => `tag:${t.t}` === hit.key) : undefined
      const item = h(
        'li.lat-feat.lat-term',
        {
          dataTerm: hit.key,
          dataSteered: w > 0 ? 'up' : w < 0 ? 'down' : undefined,
          dataWeak: tag?.weak ? '' : undefined,
        },
        [
          h('span.lat-feat-bar', { ariaHidden: 'true' }, [
            h('i', { style: `--a:${hit.score.toFixed(3)}` }),
          ]),
          h('span.lat-feat-name', hit.name),
          h('span.lat-feat-meta', tag ? `auc ${tag.auc.toFixed(2)}` : 'axis'),
          steerPair(Boolean(tag?.weak)),
        ],
      )
      if (!tag || !text) return [item]
      // the sparse probe under the tag: its features, clickable like any chip
      return [
        item,
        h(
          'li.lat-term-explain',
          {
            title: `a 5-feature logistic probe on document codes: held-out AUC ${tag.auc5.toFixed(2)} against ${tag.auc.toFixed(2)} for the full probe`,
          },
          [
            h('span', '≈'),
            ...tag.explain
              .filter(([, c]) => c > 0)
              .map(([f, c]) => chip(text, f, `coefficient ${c.toFixed(2)}`)),
          ],
        ),
      ]
    }),
  )
}

// ---- why these links -------------------------------------------------------------------------

/**
 * A link's cosine as an attribution graph. Each row is mean + norm · Σ a_f d_f + error in its
 * family's dictionary, so ⟨q, x⟩ splits into the dictionaries' mean terms (the space's shared
 * baseline), one edge per pair of active features (norm_q · norm_x · a_f · a_g · ⟨d_f, d_g⟩), and
 * error. Over the corpus the edges carry 86 % of what is above the baseline and the top three
 * edges 63 % of the edge mass, so three edges and the error share say why a link is there.
 */
interface WhyLink {
  r: number
  cos: number
  share: number
  edges: { q: [FeatureDict, number]; x: [FeatureDict, number]; v: number }[]
}

function dictFor(dicts: FeatureDict[], kind: LatentKind | undefined): FeatureDict | undefined {
  return (
    dicts.find(d => kind && d.meta.kinds.includes(kind)) ?? dicts.find(d => d.meta.name === 'text')
  )
}

async function reconstruct(
  rt: Runtime,
  dict: FeatureDict,
  codes: [number, number][],
): Promise<{ codes: [number, number][]; dirs: Float32Array[] }> {
  const dec = await weights(rt, dict, 'dec')
  const { dims: D, decScale } = dict.meta
  return {
    codes: codes.map(([f, a]) => [f, a * dict.table[f].max]),
    dirs: codes.map(([f]) => {
      const v = new Float32Array(D)
      for (let d = 0; d < D; d++) v[d] = dec[f * D + d] * decScale[f]
      return v
    }),
  }
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0
  for (let d = 0; d < a.length; d++) s += a[d] * b[d]
  return s
}

async function explainLinks(
  rt: Runtime,
  st: FeatureState,
  dicts: FeatureDict[],
): Promise<WhyLink[]> {
  const meta = await metaOf(rt)
  const url = `${rt.base}/vec/rows-256-int8.bin`
  await requestVec(rt, { type: 'load', url, rows: meta.rows, dims: 256 })
  const rowVec = async (r: number) => {
    const got = await requestVec(rt, { type: 'row', url, row: r, scale: meta.scale })
    return got.type === 'row' ? got.vector : new Float32Array(256)
  }
  // the list on screen: steering's pin, a typed query's or a feature's pin, else the focus's links
  const steered = st.pinned && [...st.steer.values()].some(t => t.w !== 0)
  const links = steered
    ? st.pinned!.slice(0, 5)
    : rt.query
      ? rt.query.items.slice(0, 5).map(i => i.preview.r)
      : (await knnOf(rt, rt.focus)).neighbours.slice(0, 5).map(n => n.r)
  const qDict = st.query
    ? dicts.find(d => d.meta.name === 'text')!
    : dictFor(dicts, docOfRow(rt, rt.focus)?.s)!
  const q = st.query ? st.query.vector : await rowVec(rt.focus)
  const qCodes = st.query ? await encodeVector(rt, qDict, q) : await codesOf(rt, qDict, rt.focus)
  const qr = await reconstruct(rt, qDict, qCodes)
  const out: WhyLink[] = []
  for (const r of links) {
    const xDict = dictFor(dicts, docOfRow(rt, r)?.s)!
    const [x, xCodes] = await Promise.all([rowVec(r), codesOf(rt, xDict, r)])
    const xr = await reconstruct(rt, xDict, xCodes)
    const cos = dot(q, x)
    const baseline =
      dot(qDict.meta.mean, x) + dot(q, xDict.meta.mean) - dot(qDict.meta.mean, xDict.meta.mean)
    const edges: WhyLink['edges'] = []
    let total = 0
    qr.codes.forEach(([f, a], i) =>
      xr.codes.forEach(([g, b], j) => {
        const v = qDict.meta.norm * xDict.meta.norm * a * b * dot(qr.dirs[i], xr.dirs[j])
        total += v
        edges.push({ q: [qDict, f], x: [xDict, g], v })
      }),
    )
    edges.sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    const above = cos - baseline
    out.push({
      r,
      cos,
      share: above > 1e-6 ? Math.max(0, Math.min(1, total / above)) : 0,
      edges: edges.slice(0, 3),
    })
  }
  return out
}

async function whySection(rt: Runtime, st: FeatureState, dicts: FeatureDict[]): Promise<Element> {
  const head = h('header', [
    h('span.lat-feat-head', 'why these links'),
    h(
      'button.lat-feat-clear',
      { type: 'button', dataWhy: '', ariaExpanded: st.explain ? 'true' : 'false' },
      st.explain ? 'hide' : 'explain',
    ),
  ])
  if (!st.explain) return h('section.lat-feat-group.lat-why', [head])
  const links = await explainLinks(rt, st, dicts)
  const previews = await Promise.all(links.map(l => previewOf(rt, l.r)))
  return h('section.lat-feat-group.lat-why', [
    head,
    h(
      'ol.lat-why-links',
      links.map((link, i) =>
        h('li', [
          h('p.lat-why-head', [
            mark(previews[i].s),
            h(
              'a',
              { href: `?focus=${link.r}`, dataLatFocus: String(link.r) },
              truncate(previews[i].t, 34),
            ),
            h(
              'span.lat-why-share',
              {
                style: `--a:${link.share.toFixed(3)}`,
                title: `cosine ${link.cos.toFixed(3)}; feature pairs carry ${Math.round(link.share * 100)} % of what is above the shared baseline, the rest is the dictionaries' error`,
              },
              `${link.cos.toFixed(2)} · ${Math.round(link.share * 100)} %`,
            ),
          ]),
          h(
            'ol.lat-why-edges',
            link.edges.map(({ q: [qd, f], x: [xd, g], v }) =>
              h('li', { dataBoth: qd === xd && f === g ? '' : undefined }, [
                h('span.lat-why-pair', [
                  chip(qd, f, qd === xd && f === g ? 'both rows fire it' : 'fires in the focus'),
                  qd === xd && f === g ? null : h('span.lat-why-to', '⟷'),
                  qd === xd && f === g ? null : chip(xd, g, 'fires in the link'),
                ]),
                h('span.lat-why-v', `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`),
              ]),
            ),
          ),
        ]),
      ),
    ),
  ])
}

// ---- wiring --------------------------------------------------------------------------------

/** Hydrate the features tab when the sidecar carries dictionaries; a no-op otherwise. */
export function bindFeatures(rt: Runtime): void {
  const panel = rt.probe.querySelector<HTMLElement>('.lat-panel[data-panel="features"]')
  const names = (rt.root.dataset.latentFeatures ?? '').split(',').filter(Boolean)
  if (!panel || names.length === 0) return
  const st: FeatureState = {
    names,
    dicts: new Map(),
    tags: null,
    selected: null,
    query: null,
    needle: '',
    steer: new Map(),
    steerSeq: 0,
    tables: null,
    tablesNow: null,
    words: null,
    channels: new Set(),
    explain: false,
    pinned: null,
    dragFrame: null,
    panel,
  }
  states.set(rt, st)
  const open = () => !panel.hidden
  const refresh = () => {
    if (open()) void renderPanel(rt, st).catch(error => console.warn('[latent] features', error))
  }
  rt.probe.addEventListener(
    'click',
    event => {
      const target = event.target as HTMLElement
      if (target.closest('.lat-tab[data-tab="features"]')) {
        // the root's tab handler un-hides the panel after this listener; render once it has
        setTimeout(refresh, 0)
        return
      }
      if (target.closest('.lat-steer-add') && panel.contains(target)) {
        // the add row: start a `#` feature search in the box; its results carry ±
        const box = rt.links.querySelector<HTMLInputElement>('form.lat-search input')
        if (box) {
          if (!box.value.trim().startsWith('#')) box.value = '#'
          box.focus()
        }
        return
      }
      if (target.closest('button[data-steer-clear]') && panel.contains(target)) {
        st.steer.clear()
        void applySteer(rt, st)
        return
      }
      if (target.closest('button[data-why]') && panel.contains(target)) {
        st.explain = !st.explain
        refresh()
        return
      }
      if (target.closest('input[data-steer-range]')) return
      const steerButton = target.closest<HTMLButtonElement>('button[data-steer]')
      // a steering row, a tag or axis row, or a chip that steers: all carry the term key
      const term = target.closest<HTMLElement>('[data-term]')
      if (
        term &&
        panel.contains(term) &&
        (steerButton || target.closest('button[data-steer-drop]'))
      ) {
        event.preventDefault()
        const key = term.dataset.term!
        if (target.closest('button[data-steer-drop]')) st.steer.delete(key)
        else steerBump(st, key, Number(steerButton!.dataset.steer))
        void applySteer(rt, st)
        return
      }
      const item = target.closest<HTMLElement>('[data-dict][data-f]')
      if (!item || !panel.contains(item)) return
      event.preventDefault()
      const name = item.dataset.dict!
      const f = Number(item.dataset.f)
      if (steerButton) {
        steerBump(st, steerKey(name, f), Number(steerButton.dataset.steer))
        if (!st.steer.size || !st.selected) st.selected = { dict: name, f }
        void applySteer(rt, st)
      } else if (target.closest('button')) void selectFeature(rt, st, name, f)
    },
    { signal: rt.signal },
  )
  // a strength slider: the links follow the drag (one fused pass a frame, the panel stays put so
  // the drag is not cut), and the panel re-renders on release
  panel.addEventListener(
    'input',
    event => {
      const input = (event.target as HTMLElement).closest<HTMLInputElement>(
        'input[data-steer-range]',
      )
      const row = input?.closest<HTMLElement>('[data-term]')
      if (!input || !row) return
      const key = row.dataset.term!
      const w = Number(input.value)
      st.steer.set(key, termOf(key, w))
      row.dataset.w = w.toFixed(2)
      const out = row.querySelector('output')
      if (out) out.textContent = amount(w)
      if (st.dragFrame !== null) return
      st.dragFrame = requestAnimationFrame(() => {
        st.dragFrame = null
        void applySteer(rt, st, true)
      })
    },
    { signal: rt.signal },
  )
  panel.addEventListener(
    'change',
    event => {
      if (!(event.target as HTMLElement).closest('input[data-steer-range]')) return
      if (st.dragFrame !== null) cancelAnimationFrame(st.dragFrame)
      st.dragFrame = null
      void applySteer(rt, st)
    },
    { signal: rt.signal },
  )
  // a new focus (renderLinks) and a settled typed query (runQuery) announce themselves
  // with a steering set on the focus, a new focus is steered the same way
  rt.root.addEventListener(
    'latent:focus',
    () => {
      if (st.steer.size && !st.query) void applySteer(rt, st)
      else refresh()
    },
    { signal: rt.signal },
  )
  rt.root.addEventListener(
    'latent:query',
    event => {
      const detail = (event as CustomEvent<{ text: string; vector: Float32Array } | null>).detail
      st.query = detail
      if (!detail) st.selected = null
      if (st.steer.size) void applySteer(rt, st)
      else refresh()
    },
    { signal: rt.signal },
  )
  // `#` in the search box searches features instead of passages (search.ts leaves it alone):
  // the tab opens, matches list as you type, and after a pause (or Enter) the best match is
  // selected with its cluster pinned to the links pane
  const input = rt.links.querySelector<HTMLInputElement>('form.lat-search input')
  const tab = rt.probe.querySelector<HTMLButtonElement>('.lat-tab[data-tab="features"]')
  let timer: number | null = null
  const commit = async () => {
    timer = null
    if (!st.needle) return
    const dicts = await Promise.all(st.names.map(name => loadDict(rt, st, name)))
    const best = findFeatures(dicts, st.needle)[0]
    if (best && rt.wanted !== `#${st.needle}`) {
      rt.wanted = `#${st.needle}`
      if (st.steer.size) {
        // steering owns the links pane: `#` only finds features to add to it
        st.selected = { dict: best.dict.meta.name, f: best.f }
        refresh()
      } else await selectFeature(rt, st, best.dict.meta.name, best.f, true)
    }
  }
  input?.addEventListener(
    'input',
    () => {
      if (timer !== null) window.clearTimeout(timer)
      const value = input.value.trim()
      const was = st.needle
      st.needle = value.startsWith('#') ? value.slice(1).trim().toLowerCase() : ''
      if (!st.needle) {
        if (was) refresh()
        return
      }
      if (!open()) tab?.click()
      else refresh()
      timer = window.setTimeout(() => void commit(), 350)
    },
    { signal: rt.signal },
  )
  rt.root.addEventListener(
    'latent:feature-search',
    () => {
      if (timer !== null) window.clearTimeout(timer)
      rt.wanted = ''
      void commit()
    },
    { signal: rt.signal },
  )
}
