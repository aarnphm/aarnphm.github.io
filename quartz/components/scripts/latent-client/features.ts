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
import { renderRail } from './search'

/**
 * Sparse features of the latent space (Prism's move): quartz/scripts/latent/features.py trains a
 * TopK sparse autoencoder per family of rows (text passages, images, base entries) and writes
 * `feat/<name>/`. The probe's features tab lists the features the focused chunk fires, a click
 * pins a feature's strongest rows to the links pane, ± steers the focus vector along a feature's
 * decoder direction and asks for its neighbours there, and a typed query is encoded into
 * features in the browser. Nothing here loads until the tab opens or the search box is used.
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
}

interface FeatureDict {
  meta: FeatureDictMeta
  table: FeatureRow[]
  ids: Uint16Array
  acts: Uint8Array
  enc: Int8Array | null
  dec: Int8Array | null
}

interface FeatureState {
  names: string[]
  dicts: Map<string, Promise<FeatureDict>>
  selected: { dict: string; f: number } | null
  query: { text: string; vector: Float32Array } | null
  needle: string
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
      fetchBytes(rt, `${base}/codes.bin`),
    ]).then(([meta, table, codes]) => {
      const n = meta.rows * meta.k
      return {
        meta,
        table,
        ids: new Uint16Array(codes, 0, n),
        acts: new Uint8Array(codes, n * 2, n),
        enc: null,
        dec: null,
      }
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

/** The features the row fires in one dictionary, strongest first: [feature, activation 0..1]. */
function codesOf(dict: FeatureDict, row: number): [number, number][] {
  const k = dict.meta.k
  const out: [number, number][] = []
  for (let j = 0; j < k; j++) {
    const a = dict.acts[row * k + j]
    if (a > 0) out.push([dict.ids[row * k + j], a / 255])
  }
  return out
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

function featureItem(dict: FeatureDict, f: number, a: number, selected: boolean): Element {
  const row = dict.table[f]
  return h(
    'li.lat-feat',
    { dataDict: dict.meta.name, dataF: String(f), dataSelected: selected ? '' : undefined },
    [
      h('span.lat-feat-bar', { ariaHidden: 'true' }, [h('i', { style: `--a:${a.toFixed(3)}` })]),
      h(
        'button.lat-feat-name',
        { type: 'button', title: row.terms.join(' · ') || undefined },
        featureName(row),
      ),
      h('span.lat-feat-meta', `${dict.meta.name === 'text' ? '' : `${dict.meta.name} `}#${f}`),
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
      featureItem(dict, f, a, st.selected?.dict === dict.meta.name && st.selected.f === f),
    ),
  )
}

function section(label: string, count: string, body: ElementContent[]): Element {
  return h('section.lat-feat-group', [
    h('header', [h('span.lat-feat-head', label), h('span.lat-count', count)]),
    ...body,
  ])
}

async function featureCard(rt: Runtime, dict: FeatureDict, f: number): Promise<Element> {
  const row = dict.table[f]
  const cross = await Promise.all(row.cross.slice(0, 4).map(([r]) => previewOf(rt, r)))
  const kinds = KIND_ORDER.map((kind, i) =>
    row.kinds[i] ? h('span', [mark(kind), ` ${row.kinds[i].toLocaleString('en-US')}`]) : null,
  ).filter(Boolean) as Element[]
  return h('section.lat-feat-card', { dataDict: dict.meta.name, dataF: String(f) }, [
    h('header', [
      h('span.lat-feat-head', featureName(row)),
      h('span.lat-chip', `${dict.meta.name} #${f} · ${share(dict, row)}`),
    ]),
    row.terms.length > 3 ? h('p.lat-feat-terms', row.terms.join(' · ')) : null,
    h('p.lat-feat-kinds', kinds),
    row.near.length
      ? h('p.lat-feat-near', [
          h('span', 'near'),
          ...row.near.map(([g, c]) =>
            h(
              'button.lat-feat-chip',
              {
                type: 'button',
                dataDict: dict.meta.name,
                dataF: String(g),
                title: `decoder cosine ${c.toFixed(2)}`,
              },
              featureName(dict.table[g]),
            ),
          ),
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
  const focusRows = codesOf(own, focus).map(([f, a]) => ({ dict: own, f, a }))
  const preview = await previewOf(rt, focus)
  if (rt.signal.aborted || rt.focus !== focus) return
  groups.push(
    section('focus', `${focusRows.length} of ${own.meta.features.toLocaleString('en-US')}`, [
      h('p.lat-feat-where', truncate(preview.t, 48)),
      featureList(focusRows, st),
    ]),
  )
  for (const other of ordered.slice(1)) {
    const read = codesOf(other, focus)
      .slice(0, 3)
      .map(([f, a]) => ({ dict: other, f, a }))
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
  if (st.needle.length >= 3) {
    const hits: { dict: FeatureDict; f: number; a: number }[] = []
    for (const dict of dicts)
      for (const row of dict.table)
        if (row.terms.some(t => t.startsWith(st.needle)))
          hits.push({ dict, f: row.f, a: Math.min(1, row.n / 2000) })
    hits.sort((a, b) => b.dict.table[b.f].n - a.dict.table[a.f].n)
    if (hits.length)
      groups.push(
        section(`terms · ${st.needle}`, `${hits.length}`, [featureList(hits.slice(0, 12), st)]),
      )
  }
  if (st.selected) {
    const dict = dicts.find(d => d.meta.name === st.selected!.dict)
    if (dict) groups.unshift(await featureCard(rt, dict, st.selected.f))
  }
  const fvu = dicts.map(d => `${d.meta.name} ${(100 * (1 - d.meta.fvu)).toFixed(0)} %`).join(' · ')
  groups.push(
    h(
      'p.lat-feat-note',
      {
        dataTip:
          'a sparse autoencoder per family of rows splits each 256-d vector into k of M learned directions. the percentages are the variance the k directions rebuild.',
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

async function selectFeature(rt: Runtime, st: FeatureState, name: string, f: number) {
  st.selected = { dict: name, f }
  const dict = await loadDict(rt, st, name)
  const row = dict.table[f]
  void renderPanel(rt, st)
  await pinLinks(rt, `feature · ${featureName(row)}`, [...row.top, ...row.cross.slice(0, 4)])
}

/**
 * Embedding arithmetic: move the focus vector along the feature's decoder direction by the
 * feature's largest activation (or take its activation out), renormalise, and score every row.
 */
async function steer(rt: Runtime, st: FeatureState, name: string, f: number, sign: number) {
  const dict = await loadDict(rt, st, name)
  const dec = await weights(rt, dict, 'dec')
  const meta = await metaOf(rt)
  const url = `${rt.base}/vec/rows-256-int8.bin`
  await requestVec(rt, { type: 'load', url, rows: meta.rows, dims: 256 })
  const got = await requestVec(rt, { type: 'row', url, row: rt.focus, scale: meta.scale })
  if (got.type !== 'row' || rt.signal.aborted) return
  const v = got.vector
  const { dims: D, norm, decScale } = dict.meta
  const now = codesOf(dict, rt.focus).find(([g]) => g === f)?.[1] ?? 0
  const row = dict.table[f]
  // in the dictionary's units: +max adds the feature at full strength; − removes what is there
  // and as much again, so a chunk that barely fires it still moves
  const amount = sign > 0 ? row.max : -Math.max(now * row.max * 2, row.max * 0.5)
  const step = amount * norm * decScale[f]
  let length = 0
  for (let d = 0; d < D; d++) {
    v[d] += step * dec[f * D + d]
    length += v[d] * v[d]
  }
  length = Math.sqrt(length) || 1
  for (let d = 0; d < D; d++) v[d] /= length
  const result = await requestVec(
    rt,
    { type: 'query', url, vector: v, scale: meta.scale, k: rt.k + 1 },
    [v.buffer],
  )
  if (result.type !== 'queried' || rt.signal.aborted) return
  const rows: [number, number][] = []
  result.hits.forEach((r, i) => {
    if (r !== rt.focus && rows.length < rt.k)
      rows.push([r, Math.round(Math.max(0, result.scores[i]) * 255)])
  })
  st.selected = { dict: name, f }
  void renderPanel(rt, st)
  await pinLinks(rt, `focus ${sign > 0 ? '+' : '−'} ${featureName(row)}`, rows)
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
    selected: null,
    query: null,
    needle: '',
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
      const steerButton = target.closest<HTMLButtonElement>('button[data-steer]')
      const item = target.closest<HTMLElement>('[data-dict][data-f]')
      if (!item || !panel.contains(item)) return
      event.preventDefault()
      const name = item.dataset.dict!
      const f = Number(item.dataset.f)
      if (steerButton) void steer(rt, st, name, f, Number(steerButton.dataset.steer))
      else if (target.closest('button')) void selectFeature(rt, st, name, f)
    },
    { signal: rt.signal },
  )
  // a new focus (renderLinks) and a settled typed query (runQuery) announce themselves
  rt.root.addEventListener('latent:focus', refresh, { signal: rt.signal })
  rt.root.addEventListener(
    'latent:query',
    event => {
      const detail = (event as CustomEvent<{ text: string; vector: Float32Array } | null>).detail
      st.query = detail
      if (!detail) st.selected = null
      refresh()
    },
    { signal: rt.signal },
  )
  const input = rt.links.querySelector<HTMLInputElement>('form.lat-search input')
  input?.addEventListener(
    'input',
    () => {
      st.needle = input.value.trim().toLowerCase()
      if (st.needle.length >= 3 || !st.needle) refresh()
    },
    { signal: rt.signal },
  )
}
