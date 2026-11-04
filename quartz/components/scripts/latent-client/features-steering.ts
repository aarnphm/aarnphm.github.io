import { type Element } from 'hast'
import { h } from 'hastscript'
import { loadDict, weights, steerTables, RANGE } from './features-data'
import {
  unitOf,
  steerLabel,
  termName,
  isWeak,
  tagGloss,
  amount,
  steerPair,
  chip,
} from './features-markup'
import { type FeatureState, type SteerTerm, type FeatureDict, type TermHit } from './features-types'
import { type Runtime, requestVec, metaOf } from './runtime'
import { filterMask, renderLinks } from './search'

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

export async function applySteer(rt: Runtime, st: FeatureState, drag = false, retry = true) {
  const seq = ++st.steerSeq
  const live = [...st.steer.values()].filter(t => t.w !== 0)
  if (live.length === 0 && !st.query) {
    // nothing left to steer: back to the focus's own links (a query re-scores unsteered below)
    st.pinned = null
    if (!drag) st.refresh()
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
      only: await filterMask(rt),
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
  await st.pin(label, rows)
  // after the pin, so "reasoning" explains the steered list
  if (!drag) st.refresh()
}

export async function steeringSection(
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
              min: String(-Math.max(RANGE, Math.ceil(Math.abs(term.w)))),
              max: String(Math.max(RANGE, Math.ceil(Math.abs(term.w)))),
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

export async function findTerms(rt: Runtime, st: FeatureState, needle: string): Promise<TermHit[]> {
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

export function termList(st: FeatureState, dicts: FeatureDict[], hits: TermHit[]): Element {
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
