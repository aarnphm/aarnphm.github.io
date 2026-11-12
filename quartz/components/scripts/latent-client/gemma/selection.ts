import { bridgesOf, predicts } from '../bridges'
import { benchFeatures } from '../features'
import { docOfRow, type Runtime } from '../runtime'
import { bakeOf, glossOf, indexOf, labelOf, PORT, rowOf, type Bake, type BakeRow } from './bake'
import { clearHighlights, highlight } from './highlights'
import { bridgeSection, esc, ll, panelHtml, textChip, textNames } from './markup'
import { reads, tokensIn, tokensOfFeature } from './read'
import { align, chunkText, rawRange, type Aligned } from './text'

interface GemmaState {
  bake: Promise<Bake | null>
  panel: HTMLElement | null
  seq: number
  aligned: Aligned | null
  data: BakeRow | null
  picked: number[]
}

/** the selection tab's "elsewhere" box, made when a feature card asks for it */
function whereBox(rt: Runtime, st: GemmaState): HTMLElement {
  const panel = ensureTab(rt, st)
  let box = panel.querySelector<HTMLElement>('.lat-gm-where')
  if (!box) {
    box = document.createElement('section')
    box.className = 'lat-feat-group lat-gm-where'
    panel.append(box)
  }
  return box
}

/** POST the selection to plt_serve.py; the graph goes to the circuit tab (circuit.ts) */
async function traceNext(rt: Runtime, st: GemmaState, button: HTMLButtonElement): Promise<void> {
  const { aligned: al, data, picked } = st
  if (!al || !data || !picked.length || button.disabled) return
  button.disabled = true
  button.textContent = 'tracing…'
  let note = 'trace the next token'
  try {
    const response = await fetch(`http://localhost:${PORT}/graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        h8: (rt.root.dataset.latentHash ?? '').slice(0, 8),
        row: al.row,
        text: al.raw,
        end: data.spans[2 * picked[picked.length - 1] + 1],
      }),
      signal: rt.signal,
    })
    const body = await response.json()
    if (!response.ok || body.error) note = `not traced: ${body.error ?? response.status}`
    else rt.root.dispatchEvent(new CustomEvent('latent:graph', { detail: body }))
  } catch {
    note = 'not traced: plt_serve.py is not running'
  }
  if (!button.isConnected) return
  button.disabled = false
  button.textContent = note
}

async function elsewhere(rt: Runtime, st: GemmaState, l: number, f: number): Promise<void> {
  const bake = await st.bake
  if (!bake) return
  const box = whereBox(rt, st)
  box.hidden = false
  box.innerHTML = `<header><span class="lat-feat-head">${ll(l)} f${f} elsewhere</span><span class="lat-count">…</span></header>`
  const got = await Promise.all([
    indexOf(rt, bake, l),
    bake.labels,
    bridgesOf(rt),
    textNames(rt),
  ]).catch(() => null)
  if (!got) {
    box.innerHTML = `<header><span class="lat-feat-head">${ll(l)} f${f} elsewhere</span></header><p class="lat-gm-note">no index yet</p>`
    return
  }
  const [idx, labels, br, name] = got
  const best = new Map<number, number>()
  for (let i = idx.starts[f]; i < idx.starts[f + 1]; i++)
    if (idx.a[i] > (best.get(idx.r[i]) ?? 0)) best.set(idx.r[i], idx.a[i])
  const rows = [...best].sort((a, b) => b[1] - a[1]).slice(0, 8)
  const lab = labelOf(labels, l, f)
  const items = rows
    .map(([r, a]) => {
      const doc = docOfRow(rt, r)
      const title = doc ? doc.t : `row ${r}`
      return `<li class="lat-feat"><span class="lat-feat-meta">${a.toFixed(1)}</span><a class="lat-feat-name" href="?focus=${r}" data-lat-focus="${r}">${esc(title)}</a><span class="lat-feat-meta">#${doc ? r - doc.r0 + 1 : r}</span><span></span></li>`
    })
    .join('')
  const to = name ? predicts(br, l, f).slice(0, 4) : []
  const bridged = to.length
    ? `<p class="lat-feat-near"><span>predicts</span>${to.map(([j, w]) => textChip(j, name!(j), `text #${j} · bridge weight ${w.toFixed(2)}`)).join('')}</p>`
    : ''
  box.innerHTML = `<header><span class="lat-feat-head">${esc(lab?.label ?? `${ll(l)} f${f}`)}: elsewhere</span><span class="lat-count">${ll(l)} f${f}</span></header>${bridged}<ul class="lat-feat-list">${items}</ul>`
}

// ---- tab ----------------------------------------------------------------------------------------

/** a probe tab and its panel, made the first time something fills it */
export function probeTab(rt: Runtime, name: string, className: string): HTMLElement {
  const have = rt.probe.querySelector<HTMLElement>(`.lat-panel[data-panel="${name}"]`)
  if (have) return have
  const nav = rt.probe.querySelector<HTMLElement>('.lat-tabs')
  const tab = document.createElement('button')
  tab.type = 'button'
  tab.className = 'lat-tab'
  tab.dataset.tab = name
  tab.setAttribute('aria-selected', 'false')
  tab.textContent = name
  nav?.append(tab)
  const panel = document.createElement('div')
  panel.className = `lat-panel ${className}`
  panel.dataset.panel = name
  panel.hidden = true
  const panels = rt.probe.querySelectorAll('.lat-panel')
  const last = panels[panels.length - 1]
  if (last) last.after(panel)
  else rt.probe.append(panel)
  return panel
}

function ensureTab(rt: Runtime, st: GemmaState): HTMLElement {
  if (st.panel?.isConnected) return st.panel
  st.panel = probeTab(rt, 'selection', 'lat-features lat-gemma')
  return st.panel
}

function showTab(rt: Runtime): void {
  rt.probe.querySelector<HTMLButtonElement>('.lat-tab[data-tab="selection"]')?.click()
}

async function onSelection(rt: Runtime, st: GemmaState): Promise<void> {
  const sel = document.getSelection()
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return
  const range = sel.getRangeAt(0)
  const start = range.startContainer.parentElement?.closest<HTMLElement>('.lat-chunk[data-row]')
  if (!start || !rt.reader.contains(start)) return
  const row = Number(start.dataset.row)
  const seq = ++st.seq
  const bake = await st.bake
  if (!bake || seq !== st.seq) return
  const panel = ensureTab(rt, st)
  if (!bake.meta.rows[String(row)]) {
    panel.innerHTML = `<section class="lat-feat-group"><p class="lat-gm-note">not baked</p></section>`
    showTab(rt)
    return
  }
  const [raw, data, labels, glosses] = await Promise.all([
    chunkText(rt, row),
    rowOf(rt, bake, row),
    bake.labels,
    Promise.all(Array.from({ length: bake.meta.layers }, (_, l) => glossOf(rt, bake, l))),
  ])
  if (raw === null || seq !== st.seq) return
  const al =
    st.aligned?.row === row && st.aligned.nodes[0]?.node.isConnected
      ? st.aligned
      : align(start, row, raw)
  const span = rawRange(al, range)
  if (!span) return
  const picked = tokensIn(data, span[0], span[1])
  if (!picked.length || picked.length > 400) return
  const bridged = await bridgeSection(rt, data, picked).catch(() => '')
  if (seq !== st.seq) return
  st.aligned = al
  st.data = data
  st.picked = picked
  panel.innerHTML = panelHtml(bake, data, al, picked, labels, glosses, bridged)
  showTab(rt)
}

export function bindGemma(rt: Runtime): void {
  if (rt.mode === 'embed') return
  const st: GemmaState = {
    bake: bakeOf(rt),
    panel: null,
    seq: 0,
    aligned: null,
    data: null,
    picked: [],
  }
  let timer = 0
  document.addEventListener(
    'selectionchange',
    () => {
      window.clearTimeout(timer)
      timer = window.setTimeout(() => void onSelection(rt, st), 220)
    },
    { signal: rt.signal },
  )
  rt.probe.addEventListener(
    'pointerover',
    event => {
      const target = event.target as HTMLElement
      if (!st.aligned || !st.data || !st.panel?.contains(target)) return
      const f = target.closest<HTMLElement>('.lat-gm-f')
      if (f) {
        highlight(
          st.aligned,
          st.data,
          tokensOfFeature(st.data, Number(f.dataset.l), Number(f.dataset.f)),
        )
        return
      }
      const a = target.closest<HTMLElement>('.lat-gm-a')
      if (a) {
        const got = reads(st.data, st.picked, Number(a.dataset.a))
        highlight(
          st.aligned,
          st.data,
          [...got].map(([t, w]) => ({ t, a: w })),
        )
      }
    },
    { signal: rt.signal },
  )
  rt.probe.addEventListener(
    'pointerout',
    event => {
      const target = event.target as HTMLElement
      if (target.closest('.lat-gm-f, .lat-gm-a')) clearHighlights()
    },
    { signal: rt.signal },
  )
  rt.probe.addEventListener(
    'click',
    event => {
      const target = event.target as HTMLElement
      const trace = target.closest<HTMLButtonElement>('.lat-gm-trace')
      if (trace) {
        void traceNext(rt, st, trace)
        return
      }
      const text = target.closest<HTMLElement>('[data-open-text]')
      if (text) {
        // a bridge's text feature: open it in the features tab
        rt.probe.querySelector<HTMLButtonElement>('.lat-tab[data-tab="features"]')?.click()
        void benchFeatures(rt)?.select('text', Number(text.dataset.openText))
        return
      }
      const f = target.closest<HTMLElement>('.lat-gm-f, .lat-gm-bridge')
      if (!f) return
      const outside = !st.panel?.contains(f)
      void elsewhere(rt, st, Number(f.dataset.l), Number(f.dataset.f))
      // from a feature card: the answer is in the selection tab
      if (outside) showTab(rt)
    },
    { signal: rt.signal },
  )
  rt.signal.addEventListener('abort', clearHighlights)
}
