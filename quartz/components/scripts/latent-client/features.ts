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

interface FeatureState {
  names: string[]
  dicts: Map<string, Promise<FeatureDict>>
  tags: Promise<Map<number, [string, number][]> | null> | null
  selected: { dict: string; f: number } | null
  query: { text: string; vector: Float32Array } | null
  needle: string
  /** the steering set: dictionary:feature → signed units */
  steer: Map<string, { dict: string; f: number; w: number }>
  steerSeq: number
  panel: HTMLElement
}

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

function featureItem(dict: FeatureDict, f: number, a: number, selected: boolean, w = 0): Element {
  const row = dict.table[f]
  return h(
    'li.lat-feat',
    {
      dataDict: dict.meta.name,
      dataF: String(f),
      dataSelected: selected ? '' : undefined,
      dataSteered: w > 0 ? 'up' : w < 0 ? 'down' : undefined,
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
      h('span.lat-feat-steer', [
        h('button', { type: 'button', dataSteer: '1', title: 'more of this feature' }, '+'),
        h('button', { type: 'button', dataSteer: '-1', title: 'less of this feature' }, '−'),
      ]),
    ],
  )
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

async function featureCard(rt: Runtime, dict: FeatureDict, f: number): Promise<Element> {
  const row = dict.table[f]
  const up = ancestry(dict, f)
  const kids = dict.children.get(f) ?? []
  const cross = await Promise.all(row.cross.slice(0, 4).map(([r]) => previewOf(rt, r)))
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
    h('p.lat-feat-kinds', kinds),
    timeline(dict, row),
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
  const dicts = await Promise.all(st.names.map(name => loadDict(rt, st, name)))
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
                ...read.map(([t, p]) => h('span.lat-feat-tag', `#${t} ${p}`)),
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
  if (st.needle) {
    const hits = findFeatures(dicts, st.needle)
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
    if (dict) groups.unshift(await featureCard(rt, dict, st.selected.f))
  }
  const steering = steeringSection(st, dicts)
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
 * Embedding arithmetic, stacked (Prism's edit without a decoder). Each + or − adds one unit of a
 * feature to the steering set; the base vector (the typed query when there is one, else the
 * focused chunk) moves along every feature's decoder direction by its weight, is renormalised,
 * and every row is scored again. + adds the feature's largest activation per unit; − takes out
 * what the base already has of it plus half its largest activation per unit, so a base that
 * barely fires it still moves.
 */
function steerKey(name: string, f: number): string {
  return `${name}:${f}`
}

function steerBump(st: FeatureState, name: string, f: number, sign: number): void {
  const key = steerKey(name, f)
  const w = (st.steer.get(key)?.w ?? 0) + sign
  if (w === 0) st.steer.delete(key)
  else st.steer.set(key, { dict: name, f, w })
}

function steerLabel(st: FeatureState, dicts: Map<string, FeatureDict>): string {
  return [...st.steer.values()]
    .map(({ dict, f, w }) => {
      const d = dicts.get(dict)
      const name = d ? featureName(d.table[f]) : `#${f}`
      return `${w > 0 ? '+' : '−'}${Math.abs(w) > 1 ? Math.abs(w) : ''} ${name}`
    })
    .join(' ')
}

async function applySteer(rt: Runtime, st: FeatureState): Promise<void> {
  const seq = ++st.steerSeq
  if (st.steer.size === 0 && !st.query) {
    // nothing left to steer: back to the focus's own links (a query re-scores unsteered below)
    void renderPanel(rt, st)
    rt.query = null
    delete rt.links.dataset.query
    void renderLinks(rt, rt.focus)
    return
  }
  const meta = await metaOf(rt)
  const url = `${rt.base}/vec/rows-256-int8.bin`
  await requestVec(rt, { type: 'load', url, rows: meta.rows, dims: 256 })
  let v: Float32Array
  const base = st.query ? 'query' : 'focus'
  if (st.query) v = st.query.vector.slice()
  else {
    const got = await requestVec(rt, { type: 'row', url, row: rt.focus, scale: meta.scale })
    if (got.type !== 'row') return
    v = got.vector
  }
  const used = new Map<string, FeatureDict>()
  for (const { dict: name, f, w } of st.steer.values()) {
    const dict = await loadDict(rt, st, name)
    used.set(name, dict)
    const dec = await weights(rt, dict, 'dec')
    const { dims: D, norm, decScale } = dict.meta
    const row = dict.table[f]
    const now = st.query
      ? ((await encodeVector(rt, dict, st.query.vector)).find(([g]) => g === f)?.[1] ?? 0)
      : ((await codesOf(rt, dict, rt.focus)).find(([g]) => g === f)?.[1] ?? 0)
    // in the dictionary's units, then into the 256-d space through the input scale
    const amount = w > 0 ? w * row.max : -(now * row.max + -w * 0.5 * row.max)
    const step = amount * norm * decScale[f]
    for (let d = 0; d < D; d++) v[d] += step * dec[f * D + d]
  }
  let length = 0
  for (let d = 0; d < v.length; d++) length += v[d] * v[d]
  length = Math.sqrt(length) || 1
  for (let d = 0; d < v.length; d++) v[d] /= length
  const result = await requestVec(
    rt,
    { type: 'query', url, vector: v, scale: meta.scale, k: rt.k + 1 },
    [v.buffer],
  )
  if (result.type !== 'queried' || rt.signal.aborted || seq !== st.steerSeq) return
  const rows: [number, number][] = []
  result.hits.forEach((r, i) => {
    if ((st.query || r !== rt.focus) && rows.length < rt.k)
      rows.push([r, Math.round(Math.max(0, result.scores[i]) * 255)])
  })
  void renderPanel(rt, st)
  await pinLinks(rt, st.steer.size ? `${base} ${steerLabel(st, used)}` : st.query!.text, rows)
}

function steeringSection(st: FeatureState, dicts: FeatureDict[]): Element | null {
  if (st.steer.size === 0) return null
  const byName = new Map(dicts.map(d => [d.meta.name, d]))
  return h('section.lat-feat-group.lat-feat-steering', [
    h('header', [
      h('span.lat-feat-head', `steering ${st.query ? 'the query' : 'the focus'}`),
      h('button.lat-feat-clear', { type: 'button', dataSteerClear: '' }, 'clear'),
    ]),
    h(
      'p.lat-feat-near',
      [...st.steer.values()].map(({ dict, f, w }) => {
        const d = byName.get(dict)
        return h('span.lat-feat-pin', { dataDict: dict, dataF: String(f), dataW: String(w) }, [
          h('button', { type: 'button', dataSteer: '-1', title: 'one unit less' }, '−'),
          h(
            'span',
            `${w > 0 ? '+' : '−'}${Math.abs(w) > 1 ? Math.abs(w) : ''} ${d ? featureName(d.table[f]) : `#${f}`}`,
          ),
          h('button', { type: 'button', dataSteer: '1', title: 'one unit more' }, '+'),
          h('button', { type: 'button', dataSteerDrop: '', title: 'remove' }, '×'),
        ])
      }),
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
      if (target.closest('button[data-steer-clear]') && panel.contains(target)) {
        st.steer.clear()
        void applySteer(rt, st)
        return
      }
      const steerButton = target.closest<HTMLButtonElement>('button[data-steer]')
      const item = target.closest<HTMLElement>('[data-dict][data-f]')
      if (!item || !panel.contains(item)) return
      event.preventDefault()
      const name = item.dataset.dict!
      const f = Number(item.dataset.f)
      if (target.closest('button[data-steer-drop]')) {
        st.steer.delete(steerKey(name, f))
        void applySteer(rt, st)
      } else if (steerButton) {
        steerBump(st, name, f, Number(steerButton.dataset.steer))
        if (!st.steer.size || !st.selected) st.selected = { dict: name, f }
        void applySteer(rt, st)
      } else if (target.closest('button')) void selectFeature(rt, st, name, f)
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
