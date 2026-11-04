import { type Element, type ElementContent } from 'hast'
import { h } from 'hastscript'
import { KIND_ORDER, MAX, DEFAULT_UNIT } from './features-data'
import {
  type FeatureRow,
  type FeatureDict,
  type FeatureState,
  type SteerTerm,
  type SteerTables,
} from './features-types'
import { type Runtime, metaOf } from './runtime'

// ---- markup --------------------------------------------------------------------------------

export function featureName(row: FeatureRow): string {
  if (row.label) return row.label
  return row.terms.length ? row.terms.slice(0, 3).join(' · ') : `#${row.f}`
}

export function share(dict: FeatureDict, row: FeatureRow): string {
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
        'span.lat-feat-meta.lat-feat-id',
        {
          dataLevel: row.level === undefined ? undefined : String(row.level),
          title: dict.meta.name === 'text' ? undefined : `${dict.meta.name} feature`,
        },
        `#${f}`,
      ),
      steerPair(weak),
    ],
  )
}

export function steerPair(weak: boolean): Element {
  const note = weak ? ' (a weak steerer: under half the links carry it at full strength)' : ''
  return h('span.lat-feat-steer', [
    h('button', { type: 'button', dataSteer: '1', title: `steer toward it${note}` }, '+'),
    h('button', { type: 'button', dataSteer: '-1', title: `steer away from it${note}` }, '−'),
  ])
}

export function isWeak(st: FeatureState, dict: string, f: number): boolean {
  return Boolean(st.tablesNow?.dicts[dict]?.[f]?.[2])
}

export function featureList(
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

export function section(label: string, count: string, body: ElementContent[]): Element {
  return h('section.lat-feat-group', [
    h('header', [h('span.lat-feat-head', label), h('span.lat-count', count)]),
    ...body,
  ])
}

export function chip(dict: FeatureDict, g: number, title?: string): Element {
  return h(
    'button.lat-feat-chip',
    { type: 'button', dataDict: dict.meta.name, dataF: String(g), title },
    featureName(dict.table[g]),
  )
}

/** Matryoshka ancestry, coarsest first: the parents' chain up to level 0. */
export function ancestry(dict: FeatureDict, f: number): number[] {
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
export function timeline(dict: FeatureDict, row: FeatureRow): Element | null {
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
export function steerKey(name: string, f: number): string {
  return `${name}:${f}`
}

export function termOf(key: string, w: number): SteerTerm {
  if (key.startsWith('tag:')) return { kind: 'tag', id: key.slice(4), w }
  if (key.startsWith('axis:')) return { kind: 'axis', id: key.slice(5), w }
  const at = key.lastIndexOf(':')
  return { kind: 'feature', dict: key.slice(0, at), f: Number(key.slice(at + 1)), w }
}

export function steerBump(st: FeatureState, key: string, sign: number): void {
  const old = st.steer.get(key)?.w ?? 0
  const w = Math.max(-MAX, Math.min(MAX, old + sign))
  // zero is skipped: − at +1 goes to −1 (contrastive, pushes away), + at −1 back to +1; × removes it
  st.steer.set(key, termOf(key, Math.abs(w) < 1e-6 ? sign : w))
}

export function unitOf(term: SteerTerm, tables: SteerTables | null): number {
  if (term.kind === 'feature')
    return tables?.dicts[term.dict]?.[term.f]?.[0] ?? DEFAULT_UNIT.feature
  if (term.kind === 'tag') return tables?.tags.find(t => t.t === term.id)?.unit ?? DEFAULT_UNIT.tag
  return tables?.axes.find(a => a.id === term.id)?.unit ?? DEFAULT_UNIT.axis
}

export async function termName(
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

export function amount(w: number): string {
  return `${w > 0 ? '+' : w < 0 ? '−' : ''}${Math.abs(w).toFixed(Math.abs(w) % 1 ? 2 : 0)}`
}

export async function steerLabel(rt: Runtime, st: FeatureState, dicts: Map<string, FeatureDict>) {
  const parts = await Promise.all(
    [...st.steer.values()]
      .filter(t => t.w !== 0)
      .map(async t => `${amount(t.w)} ${await termName(rt, t, dicts)}`),
  )
  return parts.join(' ')
}

// ---- tags and axes as steering terms ---------------------------------------------------------

/** "#philosophy ≈ metaphysics + aesthetics + …": the tag's 5-feature sparse probe, in words. */
export function tagGloss(st: FeatureState, dicts: FeatureDict[], t: string): string | undefined {
  const tag = st.tablesNow?.tags.find(x => x.t === t)
  const text = dicts.find(d => d.meta.name === 'text')
  if (!tag || !text) return undefined
  const names = tag.explain.filter(([, c]) => c > 0).map(([f]) => featureName(text.table[f]))
  return `#${t} ≈ ${names.join(' + ')} (5-feature probe AUC ${tag.auc5.toFixed(2)}, full ${tag.auc.toFixed(2)})`
}
