import { type Element } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import { truncate } from '../../../util/latent/rail'
import { featureCard } from './features-card'
import { loadDict, steerTables, codesOf, tagsOf, encodeVector } from './features-data'
import { section, featureList, tagGloss } from './features-markup'
import { whySection } from './features-reasoning'
import { findTerms, termList, steeringSection } from './features-steering'
import { type FeatureState, type FeatureDict } from './features-types'
import { type Runtime, docOfRow, previewOf } from './runtime'

export async function renderPanel(rt: Runtime, st: FeatureState): Promise<void> {
  const seq = ++st.renderSeq
  const focus = rt.focus
  const query = st.query
  const stale = () =>
    rt.signal.aborted || st.renderSeq !== seq || rt.focus !== focus || st.query !== query
  const [dicts] = await Promise.all([
    Promise.all(st.names.map(name => loadDict(rt, st, name))),
    steerTables(rt, st),
  ])
  if (stale()) return
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
  if (stale()) return
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
      groups.push(section(other.meta.name, `${read.length}`, [featureList(read, st)]))
  }
  // reasoning sits under what the other dictionaries read in the focus
  const whyAt = groups.length
  if (query) {
    const text = dicts.find(d => d.meta.name === 'text') ?? own
    const rows = (await encodeVector(rt, text, query.vector)).map(([f, a]) => ({
      dict: text,
      f,
      a,
    }))
    if (stale()) return
    groups.push(
      section(`query · ${truncate(query.text, 28)}`, `${rows.length}`, [featureList(rows, st)]),
    )
  }
  const why = await whySection(rt, st, dicts)
  if (stale()) return
  if (why) groups.splice(whyAt, 0, why)
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
          'a Matryoshka sparse autoencoder per family of rows splits each 256-d vector into k of M learned directions. the percentages are the variance between the coarsest prefix versus the whole dictionary.',
        dataTipAt: 'up',
        tabIndex: 0,
      },
      `${fvu}`,
    ),
  )
  if (stale()) return
  st.panel.innerHTML = toHtml(groups as never)
  if (st.panel.querySelector('.lat-why[data-why-pending]')) fillWhy(rt, st, dicts, seq)
}

/** Build "reasoning" once the browser is idle after a paint, then swap it into the same panel. */
function fillWhy(rt: Runtime, st: FeatureState, dicts: FeatureDict[], seq: number): void {
  const run = async () => {
    if (rt.signal.aborted || st.renderSeq !== seq) return
    const section = await whySection(rt, st, dicts, true)
    if (rt.signal.aborted || st.renderSeq !== seq) return
    st.whyReady = true
    const slot = st.panel.querySelector('.lat-why[data-why-pending]')
    if (slot) slot.outerHTML = toHtml(section as never)
  }
  const go = () => void run().catch(error => console.warn('[latent] reasoning', error))
  if ('requestIdleCallback' in window) requestIdleCallback(go, { timeout: 1500 })
  else setTimeout(go, 200)
}

/**
 * `#` search: a feature by id (`#99`, `#image 4`, `#base:12`) or by its label, description and
 * terms. A whole-word hit in the label ranks above a prefix, a prefix above the description and
 * terms; ties go to the feature that fires more often. `score` is 0..1 for the list's bar.
 */
export function findFeatures(
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
