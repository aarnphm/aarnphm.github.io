import { docOfRow, fetchBytes, fetchJson, partOf, textOf, type Runtime } from './runtime'

// The probe's selection tab: select text in a passage and see it as Gemma 4 E4B reads it, from
// the bake (quartz/scripts/latent/plt_bake.py): per token, the strongest transcoder features at
// each of the 42 layers and the positions it attends to at the global-attention layers. Gemma
// never runs here; localhost reads the bake from plt_serve.py, the site from
// /models/latent-plt/bake/. Highlights use the CSS Custom Highlight API, so the reader's DOM is
// never written (site :has() rules make small DOM writes restyle the whole page).

interface BakeMeta {
  layers: number
  width: number
  top: number
  attnLayers: number[]
  attnTop: number
  rows: Record<string, [number, boolean]>
}

interface BakeRow {
  T: number
  L: number
  K: number
  A: number
  KA: number
  spans: Uint32Array
  ids: Uint16Array
  acts: Float32Array
  src: Uint16Array
  w: Float32Array
}

export interface Label {
  label: string
  score?: number
}

export interface Bake {
  base: string
  meta: BakeMeta
  labels: Promise<Record<string, Record<string, Label | string>>>
  rows: Map<number, Promise<BakeRow>>
  index: Map<
    number,
    Promise<{ starts: Uint32Array; r: Uint32Array; pos: Uint16Array; a: Float32Array }>
  >
  gloss: Map<number, Promise<Record<string, string>>>
}

/** a chunk's rendered text aligned to its raw text (both in UTF-16 units) */
interface Aligned {
  row: number
  raw: string
  nodes: { node: Text; off: number }[]
  rawToDom: Int32Array
  domToRaw: Int32Array
}

interface GemmaState {
  bake: Promise<Bake | null>
  panel: HTMLElement | null
  seq: number
  aligned: Aligned | null
  data: BakeRow | null
  picked: number[]
}

const STRONG = 0.75
const PORT = 8093

// ---- f16 ------------------------------------------------------------------------------------

let F16: Float32Array | null = null

function f16(bits: Uint16Array): Float32Array {
  if (!F16) {
    F16 = new Float32Array(65536)
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1
      const e = (h >> 10) & 0x1f
      const m = h & 0x3ff
      F16[h] =
        e === 0
          ? s * m * 2 ** -24
          : e === 31
            ? m
              ? NaN
              : s * Infinity
            : s * (1 + m / 1024) * 2 ** (e - 15)
    }
  }
  const out = new Float32Array(bits.length)
  for (let i = 0; i < bits.length; i++) out[i] = F16[bits[i]]
  return out
}

function decodeRow(buf: ArrayBuffer): BakeRow {
  const dv = new DataView(buf)
  if (dv.getUint32(0, false) !== 0x504c5442) throw new Error('not a bake row')
  const [T, L, K, A, KA] = [8, 12, 16, 20, 24].map(o => dv.getUint32(o, true))
  let o = 28
  const take = <A extends Uint16Array | Uint32Array>(
    Ctor: { new (b: ArrayBuffer, o: number, n: number): A; BYTES_PER_ELEMENT: number },
    n: number,
  ): A => {
    const view = new Ctor(buf.slice(o, o + n * Ctor.BYTES_PER_ELEMENT), 0, n)
    o += n * Ctor.BYTES_PER_ELEMENT
    return view
  }
  const spans = take(Uint32Array, T * 2)
  const ids = take(Uint16Array, T * L * K)
  const acts = f16(take(Uint16Array, T * L * K))
  const src = take(Uint16Array, T * A * KA)
  const w = f16(take(Uint16Array, T * A * KA))
  return { T, L, K, A, KA, spans, ids, acts, src, w }
}

// ---- bake -----------------------------------------------------------------------------------

const BAKES = new WeakMap<Runtime, Promise<Bake | null>>()

/** the page's bake, opened once and shared by the selection and circuit tabs */
export function bakeOf(rt: Runtime): Promise<Bake | null> {
  let got = BAKES.get(rt)
  if (!got) {
    got = openBake(rt)
    BAKES.set(rt, got)
  }
  return got
}

async function openBake(rt: Runtime): Promise<Bake | null> {
  const h8 = (rt.root.dataset.latentHash ?? '').slice(0, 8)
  if (!h8) return null
  const local = ['localhost', '127.0.0.1'].includes(window.location.hostname)
  const bases = [
    ...(local ? [`http://localhost:${PORT}/bake/${h8}`] : []),
    `/models/latent-plt/bake/${h8}`,
  ]
  for (const base of bases) {
    try {
      const meta = await fetchJson<BakeMeta>(rt, `${base}/meta.json`)
      const labels = fetchJson<Record<string, Record<string, Label | string>>>(
        rt,
        `${base}/labels.json`,
      ).catch(() => ({}))
      return { base, meta, labels, rows: new Map(), index: new Map(), gloss: new Map() }
    } catch {
      // next base
    }
  }
  return null
}

function rowOf(rt: Runtime, bake: Bake, row: number): Promise<BakeRow> {
  let got = bake.rows.get(row)
  if (!got) {
    got = fetchBytes(rt, `${bake.base}/rows/${row}.bin`).then(decodeRow)
    got.catch(() => bake.rows.delete(row))
    bake.rows.set(row, got)
  }
  return got
}

function indexOf(rt: Runtime, bake: Bake, layer: number) {
  let got = bake.index.get(layer)
  if (!got) {
    const M = bake.meta.width
    got = fetchBytes(rt, `${bake.base}/index/L${String(layer).padStart(2, '0')}.bin`).then(buf => {
      const counts = new Uint32Array(buf, 0, M)
      const starts = new Uint32Array(M + 1)
      for (let f = 0; f < M; f++) starts[f + 1] = starts[f] + counts[f]
      const n = starts[M]
      let o = 4 * M
      const r = new Uint32Array(buf.slice(o, o + 4 * n))
      o += 4 * n
      const pos = new Uint16Array(buf.slice(o, o + 2 * n))
      o += 2 * n
      const a = f16(new Uint16Array(buf.slice(o, o + 2 * n)))
      return { starts, r, pos, a }
    })
    got.catch(() => bake.index.delete(layer))
    bake.index.set(layer, got)
  }
  return got
}

/** a feature's strongest distinct tokens over the bake, for features with no label */
function glossOf(rt: Runtime, bake: Bake, layer: number): Promise<Record<string, string>> {
  let got = bake.gloss.get(layer)
  if (!got) {
    got = fetchJson<Record<string, string>>(
      rt,
      `${bake.base}/gloss/L${String(layer).padStart(2, '0')}.json`,
    ).catch(() => ({}))
    bake.gloss.set(layer, got)
  }
  return got
}

export function labelOf(
  labels: Record<string, Record<string, Label | string>>,
  l: number,
  f: number,
): Label | null {
  const v = labels[`L${String(l).padStart(2, '0')}`]?.[String(f)]
  if (!v) return null
  return typeof v === 'string' ? { label: v } : v
}

// ---- alignment ------------------------------------------------------------------------------

const SPACE = /\s/

/** the chunk's own text nodes: the locus (chunk number, page) is not passage text */
function textNodes(chunk: HTMLElement): Text[] {
  const out: Text[] = []
  const walker = document.createTreeWalker(chunk, NodeFilter.SHOW_TEXT, {
    acceptNode: node =>
      (node.parentElement?.closest('.lat-locus, .katex-mathml, button') ?? null)
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  })
  for (let n = walker.nextNode(); n; n = walker.nextNode()) out.push(n as Text)
  return out
}

/**
 * Raw markdown and rendered text, aligned by their non-space characters. A raw character maps
 * where the next three raw characters appear within a short window of the rendered text, so
 * markup that never renders (`**`, `[[target|`, `$`) is skipped instead of dragging the match
 * forward onto a later word.
 */
function align(chunk: HTMLElement, row: number, raw: string): Aligned {
  const nodes: { node: Text; off: number }[] = []
  let dom = ''
  for (const node of textNodes(chunk)) {
    const t = node.data
    for (let i = 0; i < t.length; i++) {
      nodes.push({ node, off: i })
      dom += t[i]
    }
  }
  const rawToDom = new Int32Array(raw.length).fill(-1)
  const domToRaw = new Int32Array(dom.length).fill(-1)
  const rawIdx: number[] = []
  const domIdx: number[] = []
  for (let i = 0; i < raw.length; i++) if (!SPACE.test(raw[i])) rawIdx.push(i)
  for (let i = 0; i < dom.length; i++) if (!SPACE.test(dom[i])) domIdx.push(i)
  const R = rawIdx.map(i => raw[i])
  const D = domIdx.map(i => dom[i])
  const WINDOW = 48
  let j = 0
  for (let i = 0; i < R.length && j < D.length; i++) {
    let hit = -1
    if (R[i] === D[j]) hit = j
    else
      for (let k = j + 1; k < Math.min(D.length, j + WINDOW); k++) {
        if (D[k] === R[i] && D[k + 1] === R[i + 1] && D[k + 2] === R[i + 2]) {
          hit = k
          break
        }
      }
    if (hit < 0) continue
    rawToDom[rawIdx[i]] = domIdx[hit]
    domToRaw[domIdx[hit]] = rawIdx[i]
    j = hit + 1
  }
  return { row, raw, nodes, rawToDom, domToRaw }
}

function domRange(al: Aligned, s: number, e: number): Range | null {
  let a = -1
  let b = -1
  for (let i = s; i < e; i++) {
    const d = al.rawToDom[i]
    if (d < 0) continue
    if (a < 0) a = d
    b = d
  }
  if (a < 0) return null
  const range = document.createRange()
  range.setStart(al.nodes[a].node, al.nodes[a].off)
  range.setEnd(al.nodes[b].node, al.nodes[b].off + 1)
  return range
}

/** dom (node, offset) -> index into al.nodes */
function domIndex(al: Aligned, node: Node, off: number, end: boolean): number {
  if (node.nodeType !== Node.TEXT_NODE) {
    const child = node.childNodes[Math.min(off, node.childNodes.length - 1)]
    if (!child) return end ? al.nodes.length : 0
    const walker = document.createTreeWalker(child, NodeFilter.SHOW_TEXT)
    const first = (child.nodeType === Node.TEXT_NODE ? child : walker.nextNode()) as Text | null
    if (!first) return end ? al.nodes.length : 0
    node = first
    off = 0
  }
  for (let i = 0; i < al.nodes.length; i++) {
    const n = al.nodes[i]
    if (n.node === node && n.off >= off) return i
  }
  return end ? al.nodes.length : 0
}

/** the raw range a dom selection covers, through the nearest aligned characters */
function rawRange(al: Aligned, range: Range): [number, number] | null {
  const a = domIndex(al, range.startContainer, range.startOffset, false)
  const b = domIndex(al, range.endContainer, range.endOffset, true)
  let s = -1
  let e = -1
  for (let i = a; i < Math.min(b, al.domToRaw.length); i++) {
    const r = al.domToRaw[i]
    if (r < 0) continue
    if (s < 0) s = r
    e = r + 1
  }
  return s < 0 ? null : [s, e]
}

// ---- reading the selection --------------------------------------------------------------------

function tokensIn(data: BakeRow, s: number, e: number): number[] {
  const out: number[] = []
  for (let t = 0; t < data.T; t++) {
    const a = data.spans[2 * t]
    const b = data.spans[2 * t + 1]
    if (b > s && a < e) out.push(t)
  }
  return out
}

/** per layer, the strongest features over the picked tokens (max-pooled), and each layer's max */
function pool(data: BakeRow, picked: number[]) {
  const { L, K, T } = data
  const layerMax = new Float32Array(L)
  for (let t = 0; t < T; t++)
    for (let l = 0; l < L; l++) layerMax[l] = Math.max(layerMax[l], data.acts[(t * L + l) * K])
  const layers: { f: number; a: number }[][] = []
  for (let l = 0; l < L; l++) {
    const best = new Map<number, number>()
    for (const t of picked)
      for (let k = 0; k < K; k++) {
        const i = (t * L + l) * K + k
        const a = data.acts[i]
        if (a <= 0) continue
        const f = data.ids[i]
        if (a > (best.get(f) ?? 0)) best.set(f, a)
      }
    layers.push([...best].map(([f, a]) => ({ f, a })).sort((x, y) => y.a - x.a))
  }
  return { layers, layerMax }
}

function tokensOfFeature(data: BakeRow, l: number, f: number): { t: number; a: number }[] {
  const out: { t: number; a: number }[] = []
  for (let t = 0; t < data.T; t++)
    for (let k = 0; k < data.K; k++) {
      const i = (t * data.L + l) * data.K + k
      if (data.ids[i] === f && data.acts[i] > 0) out.push({ t, a: data.acts[i] })
    }
  return out
}

/** what the picked tokens read at one global layer: sources outside the selection, by weight */
function reads(data: BakeRow, picked: number[], a: number): Map<number, number> {
  const inside = new Set(picked)
  const got = new Map<number, number>()
  for (const t of picked)
    for (let k = 0; k < data.KA; k++) {
      const i = (t * data.A + a) * data.KA + k
      const s = data.src[i]
      const w = data.w[i]
      if (w <= 0 || inside.has(s)) continue
      got.set(s, (got.get(s) ?? 0) + w / picked.length)
    }
  return got
}

// ---- highlights -------------------------------------------------------------------------------

const HIGHLIGHTS = ['lat-gemma-1', 'lat-gemma-2', 'lat-gemma-3']

export function clearHighlights(): void {
  if (!('highlights' in CSS)) return
  for (const name of HIGHLIGHTS) CSS.highlights.delete(name)
}

/** three strength buckets: the Highlight API has no per-range opacity */
function highlight(al: Aligned, data: BakeRow, items: { t: number; a: number }[]): void {
  clearHighlights()
  if (!('highlights' in CSS) || !items.length) return
  const max = Math.max(...items.map(i => i.a))
  const buckets: Range[][] = [[], [], []]
  for (const { t, a } of items) {
    const range = domRange(al, data.spans[2 * t], data.spans[2 * t + 1])
    if (!range) continue
    const x = a / max
    buckets[x > 0.66 ? 2 : x > 0.33 ? 1 : 0].push(range)
  }
  buckets.forEach((ranges, i) => {
    if (ranges.length) CSS.highlights.set(HIGHLIGHTS[i], new Highlight(...ranges))
  })
}

const MARKS = new WeakMap<HTMLElement, Aligned>()

/**
 * Highlight raw-text spans (UTF-16 offsets into the chunk's markdown) of a reader passage, each
 * in one of the three strengths (0 faint .. 2 strong). For the circuit tab: its context window,
 * traced token, and hovered node's token. Resolves false when the passage is not on screen.
 */
export async function markPassage(
  rt: Runtime,
  row: number,
  spans: [number, number, number][],
): Promise<boolean> {
  const chunk = rt.reader.querySelector<HTMLElement>(`.lat-chunk[data-row="${row}"]`)
  if (!chunk || !('highlights' in CSS)) return false
  let al = MARKS.get(chunk)
  if (!al || !al.nodes[0]?.node.isConnected) {
    const raw = await chunkText(rt, row)
    if (raw === null) return false
    al = align(chunk, row, raw)
    MARKS.set(chunk, al)
  }
  clearHighlights()
  const buckets: Range[][] = [[], [], []]
  for (const [s, e, level] of spans) {
    const range = domRange(al, s, e)
    if (range) buckets[Math.max(0, Math.min(2, level))].push(range)
  }
  buckets.forEach((ranges, i) => {
    if (ranges.length) CSS.highlights.set(HIGHLIGHTS[i], new Highlight(...ranges))
  })
  return true
}

// ---- markup -----------------------------------------------------------------------------------

export const esc = (s: string) =>
  s.replace(
    /[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!,
  )

export const ll = (l: number) => `L${String(l).padStart(2, '0')}`

function featureButton(
  l: number,
  f: number,
  a: number,
  label: Label | null,
  gloss?: string,
): string {
  const strong = (label?.score ?? 0) >= STRONG
  const name = label ? label.label : gloss || `f${f}`
  const score =
    label?.score !== undefined
      ? ` · detection ${label.score.toFixed(2)}`
      : gloss
        ? ' · fires on'
        : ''
  const kind = !label && gloss ? ' data-gloss' : ''
  return `<button type="button" class="lat-gm-f"${strong ? ' data-strong' : ''}${kind} data-l="${l}" data-f="${f}" style="--a:${a.toFixed(3)}" data-tip="${esc(`${ll(l)} f${f}${score}`)}"><span>${esc(name)}</span><i></i></button>`
}

function panelHtml(
  bake: Bake,
  data: BakeRow,
  al: Aligned,
  picked: number[],
  labels: Record<string, Record<string, Label | string>>,
  glosses: Record<string, string>[],
): string {
  const { layers, layerMax } = pool(data, picked)
  const s = data.spans[2 * picked[0]]
  const e = data.spans[2 * picked[picked.length - 1] + 1]
  const quote = al.raw.slice(s, e).replace(/\s+/g, ' ').trim()
  const top: { l: number; f: number; x: number; a: number }[] = []
  const strip = layers
    .map((feats, l) => {
      const cells = feats
        .slice(0, 3)
        .map(({ f, a }) => {
          const x = layerMax[l] > 0 ? a / layerMax[l] : 0
          top.push({ l, f, x, a })
          return featureButton(l, f, x, labelOf(labels, l, f), glosses[l]?.[String(f)])
        })
        .join('')
      return `<li><span class="lat-gm-l">${ll(l)}</span><span class="lat-gm-cells">${cells}</span></li>`
    })
    .reverse()
    .join('')
  top.sort((p, q) => q.x - p.x)
  const strongest = top
    .filter(({ l, f }) => (labelOf(labels, l, f)?.score ?? 0) >= STRONG)
    .slice(0, 10)
    .map(({ l, f, x }) => {
      const lab = labelOf(labels, l, f)!
      return `<li class="lat-feat"><span class="lat-feat-bar"><i style="--a:${x.toFixed(3)}"></i></span><button type="button" class="lat-feat-name lat-gm-f" data-l="${l}" data-f="${f}">${esc(lab.label)}</button><span class="lat-feat-meta">${ll(l)} f${f}</span><span class="lat-feat-meta">${(lab.score ?? 0).toFixed(2)}</span></li>`
    })
    .join('')
  const attn = bake.meta.attnLayers
    .map((layer, i) => `<button type="button" class="lat-gm-a" data-a="${i}">${ll(layer)}</button>`)
    .join('')
  const cut = bake.meta.rows[String(al.row)]?.[1]
  return [
    `<section class="lat-feat-group lat-gm-head"><header><span class="lat-feat-head">“${esc(quote.length > 80 ? `${quote.slice(0, 79)}…` : quote)}”</span><span class="lat-count">${picked.length} tok</span></header>`,
    `${cut ? '<p class="lat-gm-note">past the 1024-token window: the tail is not baked</p>' : ''}</section>`,
    strongest
      ? `<section class="lat-feat-group"><header><span class="lat-feat-head">reads as</span><span class="lat-count">detection ≥ ${STRONG}</span></header><ul class="lat-feat-list">${strongest}</ul></section>`
      : '',
    `<section class="lat-feat-group"><header><span class="lat-feat-head">attends to</span><span class="lat-count">global layers</span></header><nav class="lat-gm-attn">${attn}</nav></section>`,
    `<section class="lat-feat-group"><header><span class="lat-feat-head">depth</span><span class="lat-count">top 3 a layer, L41 first</span></header><ol class="lat-gm-strip">${strip}</ol></section>`,
    `<section class="lat-feat-group lat-gm-where" hidden></section>`,
  ].join('')
}

async function elsewhere(rt: Runtime, st: GemmaState, l: number, f: number): Promise<void> {
  const bake = await st.bake
  const box = st.panel?.querySelector<HTMLElement>('.lat-gm-where')
  if (!bake || !box) return
  box.hidden = false
  box.innerHTML = `<header><span class="lat-feat-head">${ll(l)} f${f} elsewhere</span><span class="lat-count">…</span></header>`
  const got = await Promise.all([indexOf(rt, bake, l), bake.labels]).catch(() => null)
  if (!got) {
    box.innerHTML = `<header><span class="lat-feat-head">${ll(l)} f${f} elsewhere</span></header><p class="lat-gm-note">no index yet</p>`
    return
  }
  const [idx, labels] = got
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
  box.innerHTML = `<header><span class="lat-feat-head">${esc(lab?.label ?? `${ll(l)} f${f}`)}: elsewhere</span><span class="lat-count">${ll(l)} f${f}</span></header><ul class="lat-feat-list">${items}</ul>`
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

async function chunkText(rt: Runtime, row: number): Promise<string | null> {
  const doc = docOfRow(rt, row)
  if (!doc) return null
  const head = await textOf(rt, doc.d)
  const c = row - doc.r0
  const chunks = await partOf(rt, doc.d, head.part ? Math.floor(c / head.part) : 0)
  return chunks.find(ch => ch.r === row)?.text ?? null
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
  st.aligned = al
  st.data = data
  st.picked = picked
  panel.innerHTML = panelHtml(bake, data, al, picked, labels, glosses)
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
      if (!st.aligned || !st.data) return
      const target = event.target as HTMLElement
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
      const f = (event.target as HTMLElement).closest<HTMLElement>('.lat-gm-f')
      if (!f) return
      void elsewhere(rt, st, Number(f.dataset.l), Number(f.dataset.f))
    },
    { signal: rt.signal },
  )
  rt.signal.addEventListener('abort', clearHighlights)
}
