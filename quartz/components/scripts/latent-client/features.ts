import { type SharedFeatures } from '../../../util/latent/share'
import { loadDict, weights, codeBlock, steerTables } from './features-data'
import { pinLinks } from './features-links'
import { featureName, ancestry, steerBump, steerKey, termOf, amount } from './features-markup'
import { renderPanel, findFeatures } from './features-panel'
import { annotateChunk } from './features-reasoning'
import { applySteer } from './features-steering'
import { type FeatureState, type FeatureDict } from './features-types'
import { type Runtime, metaOf } from './runtime'

const states = new WeakMap<Runtime, FeatureState>()

/** What the bench's features level reads (bench-feats.ts); null when the page has no dictionaries. */
export function benchFeatures(rt: Runtime) {
  const st = states.get(rt)
  if (!st) return null
  return {
    names: st.names,
    dict: (name: string) => loadDict(rt, st, name),
    dec: (dict: FeatureDict) => weights(rt, dict, 'dec'),
    block: (dict: FeatureDict, b: number) => codeBlock(rt, dict, b),
    name: featureName,
    select: (name: string, f: number) => selectFeature(rt, st, name, f),
  }
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
  // the bench rings it at the features level and lights its rows at the other two
  rt.root.dispatchEvent(new CustomEvent('latent:feature', { detail: { dict: name, f } }))
  await pinFeature(rt, st, name, f, withCluster)
}

async function pinFeature(
  rt: Runtime,
  st: FeatureState,
  name: string,
  f: number,
  withCluster: boolean,
): Promise<void> {
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

// ---- wiring --------------------------------------------------------------------------------

/** Hydrate the features tab when the sidecar carries dictionaries; a no-op otherwise. */
export function bindFeatures(rt: Runtime): void {
  const panel = rt.probe.querySelector<HTMLElement>('.lat-panel[data-panel="features"]')
  const names = (rt.root.dataset.latentFeatures ?? '').split(',').filter(Boolean)
  if (!panel || names.length === 0) return
  const st: FeatureState = {
    refresh: () => {
      void renderPanel(rt, st)
    },
    pin: (label, rows) => pinLinks(rt, label, rows),
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
    explain: true,
    whyReady: false,
    renderSeq: 0,
    whyCache: new Map(),
    whyKey: '',
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
        if (!st.explain)
          for (const slot of rt.links.querySelectorAll('.lat-link-why')) slot.replaceChildren()
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
      else if (open()) refresh()
    },
    { signal: rt.signal },
  )
  rt.root.addEventListener(
    'latent:chunk-preview',
    event => {
      const { row, slot } = (event as CustomEvent<{ row: number; slot: HTMLElement }>).detail
      void annotateChunk(rt, st, row, slot).catch(error => {
        if (!rt.signal.aborted) console.warn('[latent] reasoning', error)
      })
    },
    { signal: rt.signal },
  )
  // a chip in a card's reasoning opens that feature's card in the features tab
  rt.links.addEventListener(
    'click',
    event => {
      const item = (event.target as HTMLElement).closest<HTMLElement>(
        '.lat-link-why [data-dict][data-f]',
      )
      if (!item) return
      event.preventDefault()
      event.stopPropagation()
      const name = item.dataset.dict!
      const f = Number(item.dataset.f)
      if (!open()) tab?.click()
      void selectFeature(rt, st, name, f)
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
      // a bench showing features keeps the probe: the match lands on it as heat or a ring
      if (!open() && rt.probe.dataset.benchFeats === undefined) tab?.click()
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

/** The owning controller exposes values, keeping worker channels and caches private. */
export function snapshotFeatures(rt: Runtime): SharedFeatures | null {
  const st = states.get(rt)
  if (!st) return null
  const label = rt.query?.text ?? ''
  return {
    selected: st.selected ? { ...st.selected } : null,
    steer: [...st.steer].map(([key, term]) => [key, term.w]),
    explain: st.explain,
    query: st.query?.text ?? '',
    pin: label.startsWith('feature · ') || label.startsWith('# '),
    cluster: label.startsWith('# '),
  }
}

export async function restoreFeatures(rt: Runtime, saved: SharedFeatures): Promise<void> {
  const st = states.get(rt)
  if (!st) return
  const dicts = await Promise.all(st.names.map(name => loadDict(rt, st, name)))
  const tables = await steerTables(rt, st)
  const meta = await metaOf(rt)
  if (rt.signal.aborted) return
  const valid = (name: string, f: number) => dicts.some(d => d.meta.name === name && !!d.table[f])
  st.explain = saved.explain
  st.steer.clear()
  for (const [key, w] of saved.steer) {
    const term = termOf(key, w)
    if (
      term.kind === 'feature'
        ? valid(term.dict, term.f)
        : term.kind === 'tag'
          ? tables?.tags.some(tag => tag.t === term.id)
          : meta.axes.some(axis => axis.id === term.id)
    )
      st.steer.set(key, term)
  }
  const selected = saved.selected
  st.selected = selected && valid(selected.dict, selected.f) ? { ...selected } : null
  if (saved.pin && st.selected) {
    await pinFeature(rt, st, st.selected.dict, st.selected.f, saved.cluster)
  } else if (st.steer.size || st.query) {
    await applySteer(rt, st)
  }
  await renderPanel(rt, st)
}
export type { FeatureDict, FeatureRow } from './features-types'
