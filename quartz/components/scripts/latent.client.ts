import { toHtml } from 'hast-util-to-html'
import type { LinkItem } from '../../util/latent/markup'
import type {
  LatentChunk,
  LatentDoc,
  LatentKind,
  LatentPreview,
  LatentRowLinks,
  LatentText,
  LatentTextPart,
  LatentTrackDoc,
} from '../../util/latent/types'
import type { SemanticEmbedRequest, SemanticWorkerResponse } from '../../util/semantic-protocol'
import type { LatentWorkerRequest, LatentWorkerResponse } from '../../workers/latent.worker'
import type { ToastHandle } from './toast'
import {
  chunkArticles,
  docHead,
  linksPane,
  moreSentinel,
  railSvg,
  readerDocument,
} from '../../util/latent/markup'
import {
  LABEL_X,
  RAIL_HEIGHT,
  RAIL_WIDTH,
  TRACK_X,
  layoutRail,
  truncate,
  type RailLayout,
} from '../../util/latent/rail'
import {
  blockName,
  blockOf,
  decodeKnnBlock,
  decodeTrack,
  inScope,
  partName,
  yearOfDay,
} from '../../util/latent/types'
import { buildSelect } from '../controls/select'
import { loadPageStyles } from './page-styles'
import { rootNavSignal } from './root-lifecycle'
import { setStatusText } from './status-flap'

/**
 * Hydration for `layout: latent`. The build rendered the first screen; this makes the reader a
 * spine: scrolling moves a reading line through the open document, the chunk under it is the
 * focus, and the rail, links and probe follow. Past a document's end the next one in time is
 * mounted and the rail collapses the old one. Each focus change fetches one kNN block and the
 * preview blocks it names; a document shift fetches that document's text.
 */

const READING_LINE = 0.38

interface Runtime {
  root: HTMLElement
  base: string
  k: number
  block: number
  axes: number
  kinds: LatentKind[]
  scope: string[]
  signal: AbortSignal
  docs: LatentDoc[]
  track: LatentTrackDoc[]
  knn: Map<number, Promise<LatentRowLinks[]>>
  prev: Map<number, Promise<LatentPreview[]>>
  text: Map<number, Promise<LatentText>>
  parts: Map<string, Promise<LatentChunk[]>>
  mounting: Map<number, Promise<HTMLElement | null>>
  regions: Map<string, Promise<Record<string, { bbox: number[]; text: string }[]> | null>>
  docByRow: LatentDoc[]
  focus: number
  open: number
  rail: HTMLElement
  reader: HTMLElement
  /** the notes' KaTeX macros, from the build */
  macros: Record<string, string>
  links: HTMLElement
  probe: HTMLElement
  bench: Bench | null
  /** the vectors worker, shared by the bench and the query */
  vec: { worker: Worker; seq: number } | null
  semantic: Worker | null
  model: string
  /** a typed query pins the links pane to its hits until cleared */
  query: { text: string; items: LinkItem[] } | null
  railLayout: RailLayout | null
  /** rail viewBox height: the pane's aspect ratio in rail units, so the track spans the pane */
  railHeight: number
  railItems: LinkItem[]
  /** the bottom-right toast that carries the search phases; one per page, text changes in place */
  toast: ToastHandle | null
  /** the text the latest query was started for; older queries drop their results */
  wanted: string
}

interface Bench {
  worker: Worker
  seq: number
  level: 'docs' | 'rows'
  coords: Float32Array | null
  extent: number
  yaw: number
  pitch: number
  zoom: number
  kinds: Uint8Array | null
  axes: number[]
  meta: {
    scale: number
    mean: number[]
    axes: { id: string; name: string; dir: number[] }[]
    rows: number
    docs: number
  } | null
  frame: number
  hover: number
  // canvas size in css px; the backing store is w·dpr × h·dpr so points and labels stay sharp
  w: number
  h: number
  dpr: number
}

async function fetchJson<T>(rt: Pick<Runtime, 'signal'>, url: string): Promise<T> {
  const response = await fetch(url, { signal: rt.signal })
  if (!response.ok) throw new Error(`${response.status} ${url}`)
  return (await response.json()) as T
}

async function fetchBytes(rt: Pick<Runtime, 'signal'>, url: string): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal: rt.signal })
  if (!response.ok) throw new Error(`${response.status} ${url}`)
  return response.arrayBuffer()
}

function knnOf(rt: Runtime, row: number): Promise<LatentRowLinks> {
  const b = blockOf(row, rt.block)
  let block = rt.knn.get(b)
  if (!block) {
    block = fetchBytes(rt, `${rt.base}/knn/${blockName(b)}.bin`).then(bytes =>
      decodeKnnBlock(bytes, b, { k: rt.k, axes: rt.axes, block: rt.block }),
    )
    rt.knn.set(b, block)
  }
  return block.then(rows => rows[row - b * rt.block])
}

function previewOf(rt: Runtime, row: number): Promise<LatentPreview> {
  const b = blockOf(row, rt.block)
  let block = rt.prev.get(b)
  if (!block) {
    block = fetchJson<LatentPreview[]>(rt, `${rt.base}/prev/${blockName(b)}.json`)
    rt.prev.set(b, block)
  }
  return block.then(rows => rows[row - b * rt.block])
}

/** The head file: document meta plus part 0. */
function textOf(rt: Runtime, doc: number): Promise<LatentText> {
  let text = rt.text.get(doc)
  if (!text) {
    text = fetchJson<LatentText>(rt, `${rt.base}/text/${partName(doc, 0)}`)
    rt.text.set(doc, text)
  }
  return text
}

/** Chunks of one part of a document. */
function partOf(rt: Runtime, doc: number, part: number): Promise<LatentChunk[]> {
  if (part === 0) return textOf(rt, doc).then(t => t.chunks)
  const key = `${doc}:${part}`
  let chunks = rt.parts.get(key)
  if (!chunks) {
    chunks = fetchJson<LatentTextPart>(rt, `${rt.base}/text/${partName(doc, part)}`).then(
      p => p.chunks,
    )
    rt.parts.set(key, chunks)
  }
  return chunks
}

function stubText(doc: LatentDoc): LatentText {
  return {
    d: doc.d,
    id: doc.id,
    store: doc.s,
    title: doc.t,
    date: doc.date,
    prec: doc.prec,
    total: doc.n,
    part: 0,
    chunks: [],
  }
}

/** HTML of a document rendered around one part. */
async function documentHtml(rt: Runtime, d: number, part: number, focus: number): Promise<string> {
  const head = await textOf(rt, d)
  const chunks = part === 0 ? head.chunks : await partOf(rt, d, part)
  const offset = part * head.part
  return toHtml(readerDocument({ ...head, chunks }, rt.docs[d], focus, { offset }) as never)
}

// The chunk under the reading line and its offset in the document. Anything above it that
// grows or shrinks (a document mounted or dropped, a part filled, math typeset, its stylesheet
// or a font arriving) moves that offset; hold() scrolls by the same amount before the frame is
// painted, so the passage under the line never moves. Scrolling leaves the offset alone, so a
// scroll between the anchor and the check is not undone. The browser's own scroll anchoring is
// off for this layout (latent.scss), so there is one correction, not two.
const steady: { el: HTMLElement | null; y: number } = { el: null, y: 0 }

function anchorTo(el: HTMLElement | null): void {
  steady.el = el
  steady.y = el ? el.getBoundingClientRect().top + window.scrollY : 0
}

function hold(): void {
  const el = steady.el
  if (!el || !el.isConnected) return
  const y = el.getBoundingClientRect().top + window.scrollY
  const moved = y - steady.y
  steady.y = y
  if (Math.abs(moved) > 0.5) window.scrollBy({ top: moved, behavior: 'instant' })
}

/** Keep the text under the reading line still when something above it grows or shrinks. */
function keepLine(anchor: HTMLElement | null, mutate: () => void): void {
  if (anchor?.isConnected && anchor !== steady.el) anchorTo(anchor)
  const el = steady.el
  // read the anchor's place on screen before the change: dropping documents above can shorten
  // the page so much that the browser clamps the scroll position while laying the change out,
  // and a correction by the offset alone would then carry the clamp (tens of pixels) as a jump
  const screen = el?.isConnected ? el.getBoundingClientRect().top : null
  mutate()
  if (!el || screen === null || !el.isConnected) return hold()
  const top = el.getBoundingClientRect().top
  if (Math.abs(top - screen) > 0.5)
    window.scrollTo({ top: window.scrollY + top - screen, behavior: 'instant' })
  steady.y = el.getBoundingClientRect().top + window.scrollY
}

/** Fill a part sentinel with its part, moving the sentinel on to the next one. */
async function fillMore(rt: Runtime, more: HTMLElement): Promise<void> {
  if (more.dataset.loading !== undefined) return
  more.dataset.loading = ''
  const d = Number(more.dataset.doc)
  const part = Number(more.dataset.part)
  const side = more.dataset.side as 'before' | 'after'
  const [head, chunks] = await Promise.all([textOf(rt, d), partOf(rt, d, part)])
  if (rt.signal.aborted || !more.isConnected) return
  const docEl = more.closest<HTMLElement>('.lat-doc')
  if (!docEl) return
  const html = chunkArticles(head, rt.docs[d], chunks, rt.focus)
    .map(el => toHtml(el as never))
    .join('')
  const anchor = chunkUnderLine(rt)
  keepLine(anchor, () => {
    more.insertAdjacentHTML(side === 'before' ? 'afterend' : 'beforebegin', html)
    const lo = side === 'before' ? part : Number(docEl.dataset.lo)
    const hi = side === 'after' ? part : Number(docEl.dataset.hi)
    docEl.dataset.lo = String(lo)
    docEl.dataset.hi = String(hi)
    const remaining =
      side === 'before' ? lo * head.part : head.total - Math.min(head.total, (hi + 1) * head.part)
    if (remaining <= 0) more.remove()
    else {
      more.outerHTML = toHtml(
        moreSentinel(head, side, remaining, side === 'before' ? lo - 1 : hi + 1) as never,
      )
    }
  })
}

function docOfRow(rt: Runtime, row: number): LatentDoc {
  return rt.docByRow[row]
}

function readingLineY(): number {
  return window.innerHeight * READING_LINE
}

/** Which chunk sits under the reading line right now. */
function chunkUnderLine(rt: Runtime): HTMLElement | null {
  const y = readingLineY()
  let best: HTMLElement | null = null
  let bestDist = Infinity
  for (const chunk of rt.reader.querySelectorAll<HTMLElement>('.lat-chunk[data-row]')) {
    const rect = chunk.getBoundingClientRect()
    if (rect.bottom < y - 1 || rect.top > y + 1) {
      const dist = rect.top > y ? rect.top - y : y - rect.bottom
      if (dist < bestDist && rect.bottom > 0 && rect.top < window.innerHeight) {
        bestDist = dist
        best = chunk
      }
      continue
    }
    return chunk
  }
  return best
}

async function renderLinks(rt: Runtime, row: number): Promise<void> {
  const links = await knnOf(rt, row)
  const focusPreview = await previewOf(rt, row)
  const previews = await Promise.all(links.neighbours.map(n => previewOf(rt, n.r)))
  if (rt.signal.aborted || rt.focus !== row) return
  const items: LinkItem[] = previews.map((preview, i) => {
    const doc = rt.docs[preview.d]
    return { preview, score: links.neighbours[i].score, doc, inScope: inScope(doc.id, rt.scope) }
  })
  if (!rt.query) {
    const pane = linksPane(focusPreview, items, rt.kinds)
    linksBody(rt).innerHTML = toHtml(pane.children as never, { allowDangerousHtml: false })
    rt.links.scrollTop = 0
    rt.links.dataset.focus = String(row)
  }
  // probe ticks
  const ticks = rt.probe.querySelectorAll<SVGLineElement>('.lat-axis .lat-focus-tick')
  rt.probe.querySelectorAll<SVGSVGElement>('.lat-axis').forEach((card, i) => {
    const value = links.axes[i]
    if (value === undefined) return
    const x = (64 + (value / 127) * 64).toFixed(1)
    let tick = card.querySelector<SVGLineElement>('.lat-focus-tick')
    if (!tick) {
      tick = document.createElementNS('http://www.w3.org/2000/svg', 'line')
      tick.setAttribute('class', 'lat-focus-tick')
      tick.setAttribute('y1', '0')
      tick.setAttribute('y2', '18')
      card.querySelector('.lat-hist')?.append(tick)
    }
    tick.setAttribute('x1', x)
    tick.setAttribute('x2', x)
  })
  void ticks
  renderRail(rt, rt.query ? rt.query.items : items)
  if (rt.bench) drawBench(rt)
}

function linksBody(rt: Runtime): HTMLElement {
  return rt.links.querySelector<HTMLElement>('.lat-links-body') ?? rt.links
}

// ---- typed query ----------------------------------------------------------------------------

function vecWorker(rt: Runtime): { worker: Worker; seq: number } {
  if (!rt.vec) {
    const worker = new Worker(new URL('latent.worker.js', import.meta.url), { type: 'module' })
    rt.vec = { worker, seq: 0 }
    rt.signal.addEventListener('abort', () => worker.terminate())
  }
  return rt.vec
}

type VecMessage = LatentWorkerRequest extends infer R
  ? R extends LatentWorkerRequest
    ? Omit<R, 'seq'>
    : never
  : never

function requestVec(
  rt: Runtime,
  message: VecMessage,
  transfer: Transferable[] = [],
): Promise<LatentWorkerResponse> {
  const vec = vecWorker(rt)
  const seq = ++vec.seq
  return new Promise((resolve, reject) => {
    const onMessage = (event: MessageEvent<LatentWorkerResponse>) => {
      if (event.data.seq !== seq) return
      vec.worker.removeEventListener('message', onMessage)
      if (event.data.type === 'error') reject(new Error(event.data.message))
      else resolve(event.data)
    }
    vec.worker.addEventListener('message', onMessage)
    vec.worker.postMessage({ ...message, seq } as LatentWorkerRequest, transfer)
  })
}

/** The site's query encoder, in its own worker; loads on the first query. */
function embedQuery(
  rt: Runtime,
  text: string,
  status: (text: string) => void,
): Promise<Float32Array> {
  if (!rt.semantic) {
    rt.semantic = new Worker(new URL('semantic.worker.js', import.meta.url), { type: 'module' })
    rt.signal.addEventListener('abort', () => rt.semantic?.terminate())
  }
  const worker = rt.semantic
  const seq = Date.now()
  return new Promise((resolve, reject) => {
    const onMessage = (event: MessageEvent<SemanticWorkerResponse>) => {
      const data = event.data
      if (data.type === 'model-progress') {
        // `phase · loaded/total unit` so the status flap swaps digits in place
        status(
          data.phase === 'loading' && data.totalBytes
            ? `encoder · ${megabytes(data.loadedBytes)}/${megabytes(data.totalBytes)} MB`
            : 'starting the encoder…',
        )
        return
      }
      if (data.type === 'embedding' && data.seq === seq) {
        worker.removeEventListener('message', onMessage)
        resolve(data.vector)
      } else if (data.type === 'error' && data.seq === seq) {
        worker.removeEventListener('message', onMessage)
        reject(new Error(data.message))
      }
    }
    worker.addEventListener('message', onMessage)
    const message: SemanticEmbedRequest = {
      type: 'embed',
      seq,
      text,
      model: rt.model,
      dims: 256,
      queryDtype: 'q4',
    }
    worker.postMessage(message)
  })
}

/** Probe axes whose name or pole terms contain the query: "search for features". */
function matchAxes(rt: Runtime, text: string): number {
  const needle = text.trim().toLowerCase()
  let first: HTMLElement | null = null
  let count = 0
  for (const card of rt.probe.querySelectorAll<HTMLElement>('.lat-axis')) {
    const hit = needle.length > 1 && (card.textContent ?? '').toLowerCase().includes(needle)
    if (hit) {
      card.dataset.match = ''
      count++
      first ??= card
    } else delete card.dataset.match
  }
  if (first) {
    for (const tab of rt.probe.querySelectorAll<HTMLButtonElement>('.lat-tab'))
      tab.setAttribute('aria-selected', String(tab.dataset.tab === 'axes'))
    for (const panel of rt.probe.querySelectorAll<HTMLElement>('.lat-panel'))
      panel.hidden = panel.dataset.panel !== 'axes'
    first.scrollIntoView({ block: 'nearest' })
  }
  return count
}

const MB = new Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })
const megabytes = (bytes: number) => MB.format(bytes / 1e6)

/**
 * Search phases go to the site's bottom-right toast: one element whose text changes in place
 * (status-flap motion), held open until the query settles.
 */
function searchStatus(rt: Runtime): {
  set: (text: string) => void
  done: (text: string) => void
  fail: (text: string) => void
} {
  const toast = (): ToastHandle | null => {
    if (rt.toast?.element.isConnected) return rt.toast
    rt.toast =
      window.quartzToast?.show('', {
        durationMs: 0,
        // the same type as the ledger's micro text, so it reads like the site's other status lines
        styles: {
          fontFamily: 'var(--lat-mono, var(--codeFont))',
          fontSize: 'var(--lat-micro, 0.68rem)',
          fontWeight: '400',
          lineHeight: '1.4',
          color: 'var(--gray)',
          padding: '0.3rem 0.6rem',
        },
      }) ?? null
    return rt.toast
  }
  const set = (text: string) => {
    const handle = toast()
    if (handle) setStatusText(handle.element, text)
  }
  const settle = (text: string, holdMs: number) => {
    set(text)
    rt.toast?.dismiss(holdMs)
    rt.toast = null
  }
  return { set, done: text => settle(text, 1800), fail: text => settle(text, 3200) }
}

async function runQuery(rt: Runtime, text: string): Promise<void> {
  const status = searchStatus(rt)
  const axes = matchAxes(rt, text)
  if (rt.wanted === text && rt.query?.text === text) return
  rt.wanted = text
  const stale = () => rt.signal.aborted || rt.wanted !== text
  status.set('embedding the query…')
  try {
    const vector = await embedQuery(rt, text, status.set)
    if (stale()) return
    const url = `${rt.base}/vec/rows-256-int8.bin`
    const meta = await metaOf(rt)
    status.set(`scoring · ${meta.rows.toLocaleString('en-US')} chunks`)
    await requestVec(rt, { type: 'load', url, rows: meta.rows, dims: 256 })
    const t0 = performance.now()
    const result = await requestVec(
      rt,
      { type: 'query', url, vector, scale: meta.scale, k: rt.k },
      [vector.buffer],
    )
    if (result.type !== 'queried' || stale()) return
    const previews = await Promise.all([...result.hits].map(r => previewOf(rt, r)))
    if (stale()) return
    const items: LinkItem[] = previews.map((preview, i) => {
      const doc = rt.docs[preview.d]
      return {
        preview,
        score: Math.round(Math.max(0, result.scores[i]) * 255),
        doc,
        inScope: inScope(doc.id, rt.scope),
      }
    })
    rt.query = { text, items }
    const pane = linksPane(
      { r: -1, d: -1, s: 'note', t: text, c: 0, p: null, l: '', x: '' },
      items,
      rt.kinds,
      'query',
    )
    linksBody(rt).innerHTML = toHtml(pane.children as never, { allowDangerousHtml: false })
    rt.links.scrollTop = 0
    rt.links.dataset.query = text
    status.done(
      `${items.length} chunks · ${(performance.now() - t0).toFixed(0)} ms${axes ? ` · ${axes} ${axes === 1 ? 'axis' : 'axes'} match` : ''}`,
    )
    renderRail(rt, items)
  } catch (error) {
    if (!stale()) status.fail(error instanceof Error ? error.message : String(error))
  }
}

function clearQuery(rt: Runtime): void {
  for (const card of rt.probe.querySelectorAll<HTMLElement>('.lat-axis[data-match]'))
    delete card.dataset.match
  rt.toast?.dismiss()
  rt.toast = null
  rt.wanted = ''
  if (!rt.query && !rt.links.dataset.query) return
  rt.query = null
  delete rt.links.dataset.query
  void renderLinks(rt, rt.focus)
}

let metaCache: WeakMap<Runtime, Promise<NonNullable<Bench['meta']>>> = new WeakMap()
function metaOf(rt: Runtime): Promise<NonNullable<Bench['meta']>> {
  let meta = metaCache.get(rt)
  if (!meta) {
    meta = fetchJson<NonNullable<Bench['meta']>>(rt, `${rt.base}/vec/meta.json`)
    metaCache.set(rt, meta)
  }
  return meta
}

const SEARCH_DEBOUNCE_MS = 450

/** The box searches as you type: axes match at once, the embedding query runs after a pause. */
function bindSearch(rt: Runtime): void {
  const form = rt.links.querySelector<HTMLFormElement>('form.lat-search')
  const input = form?.querySelector<HTMLInputElement>('input')
  if (!form || !input) return
  let timer: number | null = null
  const cancel = () => {
    if (timer !== null) window.clearTimeout(timer)
    timer = null
  }
  const run = () => {
    cancel()
    const text = input.value.trim()
    if (!text) clearQuery(rt)
    else void runQuery(rt, text)
  }
  form.addEventListener(
    'submit',
    event => {
      event.preventDefault()
      run()
    },
    { signal: rt.signal },
  )
  input.addEventListener(
    'keydown',
    event => {
      if (event.key === 'Escape') {
        cancel()
        input.value = ''
        clearQuery(rt)
        input.blur()
      }
    },
    { signal: rt.signal },
  )
  input.addEventListener(
    'input',
    () => {
      cancel()
      const text = input.value.trim()
      if (!text) {
        clearQuery(rt)
        return
      }
      matchAxes(rt, text)
      timer = window.setTimeout(run, SEARCH_DEBOUNCE_MS)
    },
    { signal: rt.signal },
  )
}

// ---- the rail as a control -----------------------------------------------------------------

function railLayoutOf(rt: Runtime): RailLayout {
  rt.railLayout ??= layoutRail(rt.track, rt.railHeight)
  return rt.railLayout
}

/** Size the rail's viewBox to its pane so the track runs the full height; true when it changed. */
function measureRail(rt: Runtime): boolean {
  const svg = rt.rail.querySelector<SVGSVGElement>('svg.lat-track')
  if (!svg) return false
  const box = svg.getBoundingClientRect()
  if (box.width < 40 || box.height < 120) return false
  const height = Math.max(400, Math.round((RAIL_WIDTH * box.height) / box.width))
  if (Math.abs(height - rt.railHeight) < 2) return false
  rt.railHeight = height
  rt.railLayout = null
  return true
}

/** The document whose slot is nearest a y in rail viewBox units. */
function docNearY(rt: Runtime, y: number): number {
  const layout = railLayoutOf(rt)
  let best = 0
  let bestDist = Infinity
  for (let i = 0; i < layout.y.length; i++) {
    const dist = Math.abs(layout.y[i] - y)
    if (dist < bestDist) {
      bestDist = dist
      best = i
    }
  }
  return best
}

function bindRail(rt: Runtime): void {
  let stepping = false
  let wheelBudget = 0
  const observer = new ResizeObserver(() => {
    if (measureRail(rt)) renderRail(rt, rt.railItems)
  })
  observer.observe(rt.rail)
  rt.signal.addEventListener('abort', () => observer.disconnect())
  // a click on the track opens the document nearest that point in time
  rt.rail.addEventListener(
    'click',
    event => {
      const target = event.target as Element | null
      if (!target || target.closest('text.lat-label')) return
      const svg = target.closest<SVGSVGElement>('svg.lat-track')
      if (!svg) return
      const ctm = svg.getScreenCTM()
      if (!ctm) return
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse())
      if (point.y < 0 || point.y > rt.railHeight) return
      event.preventDefault()
      const d = docNearY(rt, point.y)
      if (d !== rt.open) void goTo(rt, rt.docs[d].r0)
    },
    { signal: rt.signal },
  )
  bindRailHover(rt, () => stepping)
  // the wheel over the rail steps through documents in time order instead of scrolling the page
  rt.rail.addEventListener(
    'wheel',
    event => {
      event.preventDefault()
      if (stepping) return
      wheelBudget += event.deltaY
      if (Math.abs(wheelBudget) < 40) return
      const direction = Math.sign(wheelBudget)
      wheelBudget = 0
      const next = rt.open + direction
      if (next < 0 || next >= rt.docs.length) return
      stepping = true
      void goTo(rt, rt.docs[next].r0).finally(() => {
        stepping = false
      })
    },
    { passive: false, signal: rt.signal },
  )
}

const HOVER_DWELL_MS = 260

/**
 * Hovering the dated track scrubs it: a probe line names the document under the pointer, and
 * resting on one for a beat opens it. The undated band is one row per kind, so it has no probe.
 */
function bindRailHover(rt: Runtime, busy: () => boolean): void {
  let hovered = -1
  let dwell: number | null = null
  let frame = 0
  let pending: { x: number; y: number } | null = null
  const svgOf = () => rt.rail.querySelector<SVGSVGElement>('svg.lat-track')
  const clear = () => {
    hovered = -1
    pending = null
    if (dwell !== null) window.clearTimeout(dwell)
    dwell = null
    svgOf()?.querySelector('.lat-probe')?.remove()
  }
  const probe = (d: number) => {
    const svg = svgOf()
    if (!svg) return
    const layout = railLayoutOf(rt)
    const y = layout.y[d]
    let group = svg.querySelector<SVGGElement>('g.lat-probe')
    if (!group) {
      group = document.createElementNS('http://www.w3.org/2000/svg', 'g')
      group.setAttribute('class', 'lat-probe')
      group.innerHTML = '<line></line><text></text>'
      svg.append(group)
    }
    const line = group.querySelector('line')!
    line.setAttribute('x1', String(TRACK_X - 6))
    line.setAttribute('x2', String(RAIL_WIDTH))
    line.setAttribute('y1', y.toFixed(1))
    line.setAttribute('y2', y.toFixed(1))
    const text = group.querySelector('text')!
    const year = yearOfDay(rt.track[d].day)
    text.setAttribute('x', String(LABEL_X))
    // the probe reads above the line near the bottom of the track, below it elsewhere
    text.setAttribute('y', (y > rt.railHeight * 0.8 ? y - 4 : y + 10).toFixed(1))
    text.textContent = `${truncate(rt.docs[d]?.t ?? '', 24)}${year ? ` · ${year}` : ''}`
  }
  const settle = () => {
    frame = 0
    const at = pending
    pending = null
    const svg = svgOf()
    if (!at || !svg) return
    const ctm = svg.getScreenCTM()
    if (!ctm) return
    const point = new DOMPoint(at.x, at.y).matrixTransform(ctm.inverse())
    const layout = railLayoutOf(rt)
    const floor = layout.datedCount < rt.track.length ? layout.undatedTop - 8 : rt.railHeight
    if (point.y < 0 || point.y > floor) {
      clear()
      return
    }
    const d = docNearY(rt, point.y)
    if (rt.track[d].day === 0) {
      clear()
      return
    }
    // a rail re-render (the dwell opening a document) drops the probe; draw it again in place
    if (d === hovered && svg.querySelector('.lat-probe')) return
    const same = d === hovered
    hovered = d
    probe(d)
    if (same) return
    if (dwell !== null) window.clearTimeout(dwell)
    dwell = window.setTimeout(() => {
      dwell = null
      if (hovered !== d || d === rt.open || busy()) return
      void goTo(rt, rt.docs[d].r0)
    }, HOVER_DWELL_MS)
  }
  rt.rail.addEventListener(
    'pointermove',
    event => {
      if (event.pointerType === 'touch') return
      const target = event.target as Element | null
      if (!target?.closest('svg.lat-track') || target.closest('text.lat-label')) {
        clear()
        return
      }
      pending = { x: event.clientX, y: event.clientY }
      if (!frame) frame = requestAnimationFrame(settle)
    },
    { signal: rt.signal },
  )
  rt.rail.addEventListener('pointerleave', clear, { signal: rt.signal })
  rt.signal.addEventListener('abort', clear)
}

function renderRail(rt: Runtime, items: LinkItem[]): void {
  rt.railItems = items
  measureRail(rt)
  const open = rt.open
  const neighbours = new Map<number, number>()
  const weights = new Map<number, number>()
  for (const item of items) {
    const best = neighbours.get(item.doc.d) ?? 0
    if (item.score > best) neighbours.set(item.doc.d, item.score)
    weights.set(item.doc.d, (weights.get(item.doc.d) ?? 0) + 1)
  }
  const chunks = [
    ...rt.reader.querySelectorAll<HTMLElement>(`.lat-doc[data-doc="${open}"] .lat-chunk[data-row]`),
  ]
  const focusChunkIndex = Math.max(
    0,
    chunks.findIndex(c => Number(c.dataset.row) === rt.focus),
  )
  const flags = new Uint8Array(rt.docs.length)
  for (const doc of rt.docs) flags[doc.d] = inScope(doc.id, rt.scope) ? 1 : 0
  const svg = railSvg({
    docs: rt.track,
    titles: d => rt.docs[d]?.t ?? '',
    open,
    focus: rt.focus,
    neighbours,
    weights,
    focusChunkIndex,
    scopeFlags: flags,
    height: rt.railHeight,
  })
  const current = rt.rail.querySelector('svg.lat-track')
  const html = toHtml(svg as never)
  if (current) current.outerHTML = html
  else rt.rail.insertAdjacentHTML('beforeend', html)
}

function setFocus(rt: Runtime, row: number, options: { push?: boolean } = {}): void {
  if (rt.focus === row) return
  rt.focus = row
  rt.open = docOfRow(rt, row).d
  rt.root.dataset.latentFocus = String(row)
  if (rt.root.dataset.latentDoc !== String(rt.open)) setReaderHead(rt)
  rt.root.dataset.latentDoc = String(rt.open)
  for (const chunk of rt.reader.querySelectorAll<HTMLElement>('.lat-chunk[data-focus]'))
    delete chunk.dataset.focus
  const chunk = rt.reader.querySelector<HTMLElement>(`.lat-chunk[data-row="${row}"]`)
  if (chunk) chunk.dataset.focus = ''
  if (options.push !== false) {
    const url = new URL(window.location.href)
    url.searchParams.set('focus', String(row))
    history.replaceState(history.state, '', url)
  }
  void renderLinks(rt, row).catch(error => {
    if (!rt.signal.aborted) console.warn('[latent]', error)
  })
}

/** The reader's frozen head (see `docHead`), or null on a page built before it. */
function readerHead(rt: Runtime): HTMLElement | null {
  return rt.reader.querySelector<HTMLElement>(':scope > .lat-reader-head')
}

/** Retitle the frozen head for the open document; it never moves, only its text changes. */
function setReaderHead(rt: Runtime): void {
  const doc = rt.docs[rt.open]
  if (!doc) return
  const html = toHtml(docHead(doc, true) as never)
  const head = readerHead(rt)
  if (head) head.outerHTML = html
  else rt.reader.insertAdjacentHTML('afterbegin', html)
}

/** Put a document before the current reader contents (after the frozen head). */
function prependDocument(rt: Runtime, html: string): void {
  const head = readerHead(rt)
  if (head) head.insertAdjacentHTML('afterend', html)
  else rt.reader.insertAdjacentHTML('afterbegin', html)
}

/** Mount a document into the reader (replacing its stub), keeping one stub either side. */
function mountDocument(
  rt: Runtime,
  d: number,
  part: number | 'last' = 0,
): Promise<HTMLElement | null> {
  const existing = rt.reader.querySelector<HTMLElement>(
    `.lat-doc[data-doc="${d}"]:not(.lat-doc-stub)`,
  )
  if (existing) return Promise.resolve(existing)
  const pending = rt.mounting.get(d)
  if (pending) return pending

  const mounting = (async (): Promise<HTMLElement | null> => {
    const head = await textOf(rt, d)
    const index =
      part === 'last' ? Math.max(0, Math.ceil(head.total / Math.max(1, head.part)) - 1) : part
    const html = await documentHtml(rt, d, index, rt.focus)
    if (rt.signal.aborted) return null
    // Pruning can remove this stub while its data is in flight. Only replace the original slot;
    // appending a late result at the reader's end creates duplicate or out-of-order documents.
    const stub = rt.reader.querySelector<HTMLElement>(`.lat-doc-stub[data-doc="${d}"]`)
    if (!stub) return null
    const anchor = chunkUnderLine(rt)
    // the previous document mounted above must not move the text under the reading line
    keepLine(anchor, () => {
      stub.outerHTML = html
      // a document mounted above gets the stub above it in the same step: a short document
      // replacing a tall stub would otherwise shrink the page above the reading line by more
      // than the scroll offset near the top, the correction would clamp at 0, and the reading
      // line would slide onto the document before, which then mounts in turn
      if (d > 0 && d < rt.open && !rt.reader.querySelector(`.lat-doc[data-doc="${d - 1}"]`))
        prependDocument(
          rt,
          toHtml(
            readerDocument(stubText(rt.docs[d - 1]), rt.docs[d - 1], -1, { stub: true }) as never,
          ),
        )
    })
    return rt.reader.querySelector<HTMLElement>(`.lat-doc[data-doc="${d}"]:not(.lat-doc-stub)`)
  })().finally(() => {
    rt.mounting.delete(d)
  })
  rt.mounting.set(d, mounting)
  return mounting
}

function ensureStubs(rt: Runtime): void {
  const mounted = [...rt.reader.querySelectorAll<HTMLElement>('.lat-doc:not(.lat-doc-stub)')].map(
    el => Number(el.dataset.doc),
  )
  if (mounted.length === 0) return
  const lo = Math.min(...mounted)
  const hi = Math.max(...mounted)
  const stubHtml = (d: number) =>
    toHtml(readerDocument(stubText(rt.docs[d]), rt.docs[d], -1, { stub: true }) as never)
  // a stub added or a document dropped above the reading line must not move the text under it
  keepLine(chunkUnderLine(rt), () => {
    if (lo > 0 && !rt.reader.querySelector(`.lat-doc[data-doc="${lo - 1}"]`))
      prependDocument(rt, stubHtml(lo - 1))
    if (hi + 1 < rt.docs.length && !rt.reader.querySelector(`.lat-doc[data-doc="${hi + 1}"]`))
      rt.reader.insertAdjacentHTML('beforeend', stubHtml(hi + 1))
    // unmount documents two away from the open one once they are well off screen: short
    // documents (an image, a two-row base) fit several to a viewport, and pruning by count
    // alone left the reader without page above it at the top of the scroll range
    for (const el of rt.reader.querySelectorAll<HTMLElement>('.lat-doc')) {
      const d = Number(el.dataset.doc)
      if (Math.abs(d - rt.open) <= 2) continue
      const rect = el.getBoundingClientRect()
      if (rect.bottom < -window.innerHeight || rect.top > 2 * window.innerHeight) el.remove()
    }
  })
}

/** Open a row's document in the reader and put that row under the reading line. */
async function goTo(rt: Runtime, row: number): Promise<void> {
  const doc = docOfRow(rt, row)
  let chunk = rt.reader.querySelector<HTMLElement>(`.lat-chunk[data-row="${row}"]`)
  if (!chunk) {
    const head = await textOf(rt, doc.d)
    if (rt.signal.aborted) return
    const part = Math.floor((row - doc.r0) / Math.max(1, head.part))
    const html = await documentHtml(rt, doc.d, part, row)
    if (rt.signal.aborted) return
    const prev = doc.d > 0 ? rt.docs[doc.d - 1] : null
    const next = doc.d + 1 < rt.docs.length ? rt.docs[doc.d + 1] : null
    const stub = (d: LatentDoc) =>
      toHtml(readerDocument(stubText(d), d, -1, { stub: true }) as never)
    const frozen = readerHead(rt)
    rt.reader.innerHTML = [prev ? stub(prev) : '', html, next ? stub(next) : ''].join('')
    if (frozen) rt.reader.prepend(frozen)
    chunk = rt.reader.querySelector<HTMLElement>(`.lat-chunk[data-row="${row}"]`)
  }
  if (!chunk) return
  const top = chunk.getBoundingClientRect().top + window.scrollY - readingLineY() + 8
  window.scrollTo({ top, behavior: 'instant' })
  anchorTo(chunk)
  setFocus(rt, row)
  // a scroll clamped at 0 fires no scroll event; let the reader look for stubs to mount anyway
  window.dispatchEvent(new Event('scroll'))
}

function bindReader(rt: Runtime): void {
  // a page built before the frozen head existed gets one now
  if (!readerHead(rt)) setReaderHead(rt)
  let frame = 0
  const update = () => {
    frame = 0
    const chunk = chunkUnderLine(rt)
    if (!chunk) return
    if (chunk !== steady.el) anchorTo(chunk)
    const row = Number(chunk.dataset.row)
    if (row !== rt.focus) {
      const open = rt.open
      setFocus(rt, row)
      // crossing into another document moves the one-stub-either-side window with it; without
      // this the reader runs out of page above a document mounted from its stub
      if (rt.open !== open) ensureStubs(rt)
    }
    const near = (el: HTMLElement) => {
      const rect = el.getBoundingClientRect()
      return rect.top < window.innerHeight * 1.5 && rect.bottom > -window.innerHeight * 0.5
    }
    // approach the rest of the open document: fill the part sentinel within one viewport
    for (const more of rt.reader.querySelectorAll<HTMLElement>('.lat-more')) {
      if (near(more)) void fillMore(rt, more)
    }
    // approach the neighbours: mount the stub that is within one viewport
    for (const stub of rt.reader.querySelectorAll<HTMLElement>('.lat-doc-stub')) {
      if (near(stub)) {
        const d = Number(stub.dataset.doc)
        void mountDocument(rt, d, d < rt.open ? 'last' : 0).then(() => ensureStubs(rt))
      }
    }
  }
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(update)
  }
  window.addEventListener('scroll', schedule, { passive: true, signal: rt.signal })
  // at the top of the page a scroll up fires no scroll event, so the stub above the opening
  // document would never mount; look once on arrival
  schedule()
  // every change of the reader's height, whatever made it, is held against the anchor
  const sizes = new ResizeObserver(() => hold())
  sizes.observe(rt.reader)
  rt.signal.addEventListener('abort', () => {
    sizes.disconnect()
    anchorTo(null)
  })
  window.addEventListener('resize', schedule, { passive: true, signal: rt.signal })
  rt.signal.addEventListener('abort', () => cancelAnimationFrame(frame))
}

function bindClicks(rt: Runtime): void {
  rt.root.addEventListener(
    'click',
    event => {
      const target = event.target as HTMLElement | null
      if (!target) return
      const focusLink = target.closest<HTMLElement>('[data-lat-focus]')
      if (focusLink) {
        event.preventDefault()
        void goTo(rt, Number(focusLink.dataset.latFocus))
        return
      }
      const label = target.closest<SVGTextElement>('text.lat-label[data-doc]')
      if (label) {
        event.preventDefault()
        const d = Number(label.dataset.doc)
        // the neighbour row in that document is in the links list
        const item = rt.links.querySelector<HTMLElement>(`.lat-link[data-doc="${d}"]`)
        void goTo(rt, item ? Number(item.dataset.row) : rt.docs[d].r0)
        return
      }
      const tab = target.closest<HTMLButtonElement>('.lat-tab[data-tab]')
      if (tab) {
        for (const other of rt.probe.querySelectorAll<HTMLButtonElement>('.lat-tab'))
          other.setAttribute('aria-selected', String(other === tab))
        for (const panel of rt.probe.querySelectorAll<HTMLElement>('.lat-panel'))
          panel.hidden = panel.dataset.panel !== tab.dataset.tab
        if (tab.dataset.tab === 'bench') void openBench(rt)
      }
    },
    { signal: rt.signal },
  )
}

function tokens(text: string): Set<string> {
  return new Set((text.toLowerCase().match(/[a-z0-9]{3,}/g) ?? []).filter(w => w.length > 2))
}

/**
 * A note as its page map, drawn like a pdf page's region boxes: one box per chunk in reading
 * order, its height the chunk's share of the text, the linked chunk marked.
 */
async function noteSilhouette(rt: Runtime, item: HTMLElement): Promise<void> {
  const slot = item.querySelector<HTMLElement>('.lat-silhouette[data-note]')
  if (!slot) return
  const d = Number(item.dataset.doc)
  const row = Number(item.dataset.row)
  const head = await textOf(rt, d)
  // a long note's later parts are fetched too, up to a map's worth of text
  const parts = Math.min(8, Math.ceil(head.total / Math.max(1, head.part || head.total)))
  const chunks = (
    await Promise.all(Array.from({ length: parts }, (_, p) => partOf(rt, d, p)))
  ).flat()
  if (rt.signal.aborted || chunks.length === 0) return
  // a line of the card holds about this many characters; a one-line chunk (a heading, a
  // short paragraph) is drawn as wide as its text
  const perLine = 60
  const lines = chunks.map(chunk => Math.max(1, chunk.text.length / perLine))
  const gap = 24
  const total = lines.reduce((sum, n) => sum + n, 0)
  const unit = Math.min(40, (1160 - gap * (chunks.length - 1)) / total)
  let y = 70
  let rects = ''
  chunks.forEach((chunk, i) => {
    const height = Math.max(4, lines[i] * unit)
    const raw = chunk.text.length / perLine
    const width = raw < 1 ? Math.max(0.2, raw) * 840 : 840
    rects += `<rect x="80" y="${y.toFixed(1)}" width="${width.toFixed(0)}" height="${height.toFixed(1)}"${chunk.r === row ? ' data-hit=""' : ''}/>`
    y += height + Math.max(6, Math.min(gap, unit))
  })
  slot.innerHTML = `<svg viewBox="0 0 1000 1300" preserveAspectRatio="xMidYMid meet" aria-hidden="true">${rects}</svg>`
  slot.dataset.ready = ''
}

function bindSilhouettes(rt: Runtime): void {
  rt.links.addEventListener(
    'pointerover',
    event => {
      const note = (event.target as HTMLElement).closest<HTMLElement>('.lat-link[data-row]')
      if (
        note?.querySelector('.lat-silhouette[data-note]') &&
        note.dataset.silhouetteDone === undefined
      ) {
        note.dataset.silhouetteDone = ''
        void noteSilhouette(rt, note).catch(() => {})
        return
      }
      const item = (event.target as HTMLElement).closest<HTMLElement>(
        '.lat-link[data-sha][data-page]',
      )
      if (!item || item.dataset.silhouetteDone !== undefined) return
      item.dataset.silhouetteDone = ''
      const sha = item.dataset.sha!
      const page = item.dataset.page!
      let regions = rt.regions.get(sha)
      if (!regions) {
        regions = fetchJson<{ pages: Record<string, { bbox: number[]; text: string }[]> }>(
          rt,
          `/embeddings/pdf-regions/${sha}.json`,
        )
          .then(data => data.pages)
          .catch(() => null)
        rt.regions.set(sha, regions)
      }
      void regions.then(pages => {
        const boxes = pages?.[page]
        const slot = item.querySelector<HTMLElement>('.lat-silhouette')
        if (!boxes || !slot || rt.signal.aborted) return
        const extract = tokens(item.querySelector('.lat-extract')?.textContent ?? '')
        let best = -1
        let bestScore = 0
        boxes.forEach((box, i) => {
          const have = tokens(box.text)
          if (have.size === 0) return
          let overlap = 0
          for (const w of have) if (extract.has(w)) overlap++
          const score = overlap / have.size
          if (score > bestScore) {
            bestScore = score
            best = i
          }
        })
        const rects = boxes
          .map((box, i) => {
            const [x0, y0, x1, y1] = box.bbox
            return `<rect x="${x0}" y="${y0}" width="${Math.max(2, x1 - x0)}" height="${Math.max(2, y1 - y0)}"${i === best && bestScore >= 0.3 ? ' data-hit=""' : ''}/>`
          })
          .join('')
        slot.innerHTML = `<svg viewBox="0 0 1000 1300" preserveAspectRatio="xMidYMid meet" aria-hidden="true">${rects}</svg>`
        slot.dataset.ready = ''
      })
    },
    { signal: rt.signal, passive: true },
  )
}

// ---- 3-d bench --------------------------------------------------------------------------

const KIND_INDEX: Record<LatentKind, number> = { note: 0, pdf: 1, image: 2, base: 3 }

async function openBench(rt: Runtime): Promise<void> {
  if (rt.bench) return
  const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
  const canvas = rt.probe.querySelector<HTMLCanvasElement>('.lat-bench-canvas')
  const controls = rt.probe.querySelector<HTMLElement>('.lat-bench-controls')
  if (!status || !canvas || !controls) return
  status.textContent = 'loading document vectors…'
  const { worker } = vecWorker(rt)
  const bench: Bench = {
    worker,
    seq: 0,
    level: 'docs',
    coords: null,
    extent: 1,
    yaw: 0.6,
    pitch: 0.5,
    zoom: 1,
    kinds: null,
    axes: [0, 1, 2],
    meta: null,
    frame: 0,
    hover: -1,
    w: canvas.clientWidth || canvas.width,
    h: canvas.clientHeight || canvas.height,
    dpr: 1,
  }
  rt.bench = bench
  fitBenchCanvas(rt, canvas)
  const meta = await metaOf(rt)
  bench.meta = meta
  bench.kinds = new Uint8Array(rt.docs.length)
  for (const doc of rt.docs) bench.kinds[doc.d] = KIND_INDEX[doc.s]
  // x, y, z and the level as rows of a ruled table: the axis, a garden select for its component
  // and the variance it carries. The menus float in the top layer because the probe scrolls
  // and would clip them
  const table = document.createElement('table')
  table.className = 'lat-bench-table'
  const body = document.createElement('tbody')
  table.append(body)
  const row = (label: string, ...cells: HTMLTableCellElement[]) => {
    const tr = document.createElement('tr')
    const th = document.createElement('th')
    th.scope = 'row'
    th.textContent = label
    tr.append(th, ...cells)
    body.append(tr)
  }
  const cell = (className: string, ...children: (Node | string)[]) => {
    const td = document.createElement('td')
    td.className = className
    td.append(...children)
    return td
  }
  // axis names carry their variance ("pc1 · 6.2 %"); the table gives it its own column
  const parts = meta.axes.map(axis => {
    const match = /^(.*?)\s+·\s+([\d.]+\s*%)$/.exec(axis.name)
    return match ? { name: match[1], share: match[2] } : { name: axis.name, share: '' }
  })
  const options = parts.map((part, i) => ({ value: i, label: part.name, detail: part.share }))
  for (const slot of [0, 1, 2]) {
    const name = 'xyz'[slot]
    const share = cell('lat-bench-share', parts[bench.axes[slot]]?.share ?? '')
    const select = buildSelect<number>({
      id: `lat-bench-${name}`,
      label: `${name} axis`,
      groups: [{ options }],
      selected: bench.axes[slot],
      onSelect: value => {
        bench.axes[slot] = value
        share.textContent = parts[value]?.share ?? ''
        void projectBench(rt)
      },
      floating: true,
    })
    const unmount = select.mount()
    rt.signal.addEventListener('abort', unmount, { once: true })
    row(name, cell('lat-bench-pick', select.element), share)
  }
  const level = (value: 'docs' | 'rows', text: string) => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'lat-bench-level'
    button.dataset.level = value
    button.setAttribute('aria-pressed', String(value === 'docs'))
    button.textContent = text
    return button
  }
  const levels = cell('lat-bench-levels', level('docs', 'documents'), level('rows', 'chunks'))
  levels.colSpan = 2
  row('level', levels)
  controls.replaceChildren(table)
  controls.addEventListener(
    'click',
    event => {
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>('.lat-bench-level')
      if (!button || button.getAttribute('aria-pressed') === 'true') return
      for (const other of controls.querySelectorAll<HTMLButtonElement>('.lat-bench-level'))
        other.setAttribute('aria-pressed', String(other === button))
      void loadBench(rt, button.dataset.level === 'rows' ? 'rows' : 'docs')
    },
    { signal: rt.signal },
  )
  bindBenchPointer(rt, canvas)
  await loadBench(rt, 'docs')
}

async function loadBench(rt: Runtime, level: 'docs' | 'rows'): Promise<void> {
  const bench = rt.bench
  const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
  if (!bench || !bench.meta || !status) return
  bench.level = level
  const rows = level === 'docs' ? bench.meta.docs : bench.meta.rows
  const url = `${rt.base}/vec/${level}-256-int8.bin`
  status.textContent = `loading ${level === 'docs' ? 'document' : 'chunk'} vectors (${((rows * 256) / 1e6).toFixed(1)} MB)…`
  const t0 = performance.now()
  await requestVec(rt, { type: 'load', url, rows, dims: 256 })
  if (level === 'rows') {
    bench.kinds = new Uint8Array(rows)
    for (const doc of rt.docs) bench.kinds.fill(KIND_INDEX[doc.s], doc.r0, doc.r0 + doc.n)
  } else {
    bench.kinds = new Uint8Array(rt.docs.length)
    for (const doc of rt.docs) bench.kinds[doc.d] = KIND_INDEX[doc.s]
  }
  status.textContent = `${rows} ${level === 'docs' ? 'documents' : 'chunks'} in ${(performance.now() - t0).toFixed(0)} ms`
  await projectBench(rt)
}

async function projectBench(rt: Runtime): Promise<void> {
  const bench = rt.bench
  if (!bench || !bench.meta) return
  const dirs = new Float32Array(3 * 256)
  bench.axes.forEach((axis, slot) => dirs.set(bench.meta!.axes[axis].dir, slot * 256))
  const mean = new Float32Array(bench.meta.mean)
  const t0 = performance.now()
  const url = `${rt.base}/vec/${bench.level}-256-int8.bin`
  const result = await requestVec(
    rt,
    { type: 'project', url, dirs, mean, scale: bench.meta.scale },
    [dirs.buffer, mean.buffer],
  )
  if (result.type !== 'projected') return
  bench.coords = result.coords
  bench.extent = result.extent || 1
  const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
  if (status)
    status.textContent = `${bench.coords.length / 3} points · projected in ${(performance.now() - t0).toFixed(0)} ms`
  drawBench(rt)
}

// Match the backing store to device pixels. device-pixel-content-box gives the exact snapped
// size (Chromium, Firefox) and re-fires when the window moves to a screen with another density;
// Safari falls back to css size × devicePixelRatio. Capped at 3× so a zoomed-in page does not
// blow the fill cost up.
function fitBenchCanvas(rt: Runtime, canvas: HTMLCanvasElement): void {
  const apply = (cssW: number, cssH: number, devW?: number, devH?: number) => {
    const bench = rt.bench
    if (!bench || cssW <= 0 || cssH <= 0) return
    const dpr = Math.min(3, window.devicePixelRatio || 1)
    let pw = Math.round(cssW * dpr)
    let ph = Math.round(cssH * dpr)
    if (devW && devH && devW / cssW <= 3) {
      pw = devW
      ph = devH
    }
    bench.w = cssW
    bench.h = cssH
    bench.dpr = pw / cssW
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw
      canvas.height = ph
    }
    drawBench(rt)
  }
  const observer = new ResizeObserver(entries => {
    const entry = entries[entries.length - 1]
    const box = entry.contentBoxSize?.[0]
    const dev = entry.devicePixelContentBoxSize?.[0]
    const cssW = box ? box.inlineSize : entry.contentRect.width
    const cssH = box ? box.blockSize : entry.contentRect.height
    apply(cssW, cssH, dev?.inlineSize, dev?.blockSize)
  })
  try {
    observer.observe(canvas, { box: 'device-pixel-content-box' })
  } catch {
    observer.observe(canvas)
  }
  rt.signal.addEventListener('abort', () => observer.disconnect())
}

function bindBenchPointer(rt: Runtime, canvas: HTMLCanvasElement): void {
  let dragging = false
  let lastX = 0
  let lastY = 0
  canvas.addEventListener(
    'pointerdown',
    event => {
      dragging = true
      lastX = event.clientX
      lastY = event.clientY
      canvas.setPointerCapture(event.pointerId)
    },
    { signal: rt.signal },
  )
  canvas.addEventListener(
    'pointermove',
    event => {
      const bench = rt.bench
      if (!bench) return
      if (dragging) {
        bench.yaw += (event.clientX - lastX) * 0.01
        bench.pitch = Math.max(0.05, Math.min(1.2, bench.pitch + (event.clientY - lastY) * 0.005))
        lastX = event.clientX
        lastY = event.clientY
        drawBench(rt)
      } else {
        const rect = canvas.getBoundingClientRect()
        hoverBench(rt, event.clientX - rect.left, event.clientY - rect.top)
      }
    },
    { signal: rt.signal },
  )
  canvas.addEventListener('pointerup', () => (dragging = false), { signal: rt.signal })
  canvas.addEventListener(
    'wheel',
    event => {
      const bench = rt.bench
      if (!bench) return
      event.preventDefault()
      bench.zoom = Math.max(0.4, Math.min(6, bench.zoom * (event.deltaY > 0 ? 0.9 : 1.1)))
      drawBench(rt)
    },
    { signal: rt.signal, passive: false },
  )
  canvas.addEventListener(
    'click',
    () => {
      const bench = rt.bench
      if (!bench || bench.hover < 0) return
      const row = bench.level === 'docs' ? rt.docs[bench.hover].r0 : bench.hover
      void goTo(rt, row)
    },
    { signal: rt.signal },
  )
}

function projected(bench: Bench, i: number, w: number, h: number): [number, number, number] {
  const c = bench.coords!
  return projectPoint(bench, c[i * 3], c[i * 3 + 1], c[i * 3 + 2], w, h)
}

/** A point of the bench's space on the canvas: [x, y, depth in -1..1]. */
function projectPoint(
  bench: Bench,
  x: number,
  y: number,
  z: number,
  w: number,
  h: number,
): [number, number, number] {
  const s = (Math.min(w, h) * 0.42 * bench.zoom) / bench.extent
  const cy = Math.cos(bench.yaw)
  const sy = Math.sin(bench.yaw)
  const cp = Math.cos(bench.pitch)
  const sp = Math.sin(bench.pitch)
  const rx = x * cy - z * sy
  const rz = x * sy + z * cy
  const ry = y * cp - rz * sp
  const depth = y * sp + rz * cp
  return [w / 2 + rx * s, h / 2 - ry * s, depth / bench.extent]
}

function hoverBench(rt: Runtime, px: number, py: number): void {
  const bench = rt.bench
  if (!bench || !bench.coords) return
  const n = bench.coords.length / 3
  let best = -1
  let bestDist = 36
  for (let i = 0; i < n; i++) {
    const [x, y] = projected(bench, i, bench.w, bench.h)
    const d = (x - px) ** 2 + (y - py) ** 2
    if (d < bestDist) {
      bestDist = d
      best = i
    }
  }
  if (best !== bench.hover) {
    bench.hover = best
    drawBench(rt)
    const status = rt.probe.querySelector<HTMLElement>('.lat-bench-status')
    if (status && best >= 0) {
      const doc = bench.level === 'docs' ? rt.docs[best] : docOfRow(rt, best)
      setBenchStatus(status, `${doc.s} · ${doc.t}`)
    }
  }
}

/**
 * The hovered point's name on the status line's one line. A name wider than the line scrolls
 * to its end and back (latent.scss), at a reading pace, so it never wraps or moves the pane.
 */
function setBenchStatus(status: HTMLElement, text: string): void {
  const span = document.createElement('span')
  span.textContent = text
  status.replaceChildren(span)
  delete status.dataset.overflow
  const style = getComputedStyle(status)
  const room = status.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
  const shift = span.getBoundingClientRect().width - room
  if (shift <= 1) return
  status.style.setProperty('--lat-shift', `${-Math.ceil(shift)}px`)
  // about 40px a second, plus the holds at either end (a fifth of the cycle each)
  status.style.setProperty('--lat-marquee', `${(shift / 40 / 0.6).toFixed(2)}s`)
  status.dataset.overflow = ''
}

function drawBench(rt: Runtime): void {
  const bench = rt.bench
  const canvas = rt.probe.querySelector<HTMLCanvasElement>('.lat-bench-canvas')
  if (!bench || !bench.coords || !bench.kinds || !canvas) return
  if (bench.frame) return
  bench.frame = requestAnimationFrame(() => {
    bench.frame = 0
    const ctx = canvas.getContext('2d')
    if (!ctx || !bench.coords || !bench.kinds) return
    const style = getComputedStyle(canvas)
    const ink = style.getPropertyValue('--dark').trim() || '#100f0f'
    const paper = style.getPropertyValue('--light').trim() || '#fffcf0'
    const colours = [
      ink,
      style.getPropertyValue('--lat-pdf').trim() || '#fdb2a2',
      style.getPropertyValue('--lat-image').trim() || '#cdd597',
      style.getPropertyValue('--gray').trim() || '#6f6e69',
    ]
    const w = bench.w
    const h = bench.h
    // draw in css px; the transform maps them onto the device-pixel backing store
    ctx.setTransform(bench.dpr, 0, 0, bench.dpr, 0, 0)
    ctx.fillStyle = paper
    ctx.fillRect(0, 0, w, h)
    const n = bench.coords.length / 3
    const order: number[] = Array.from({ length: n }, (_, i) => i)
    const depth = new Float32Array(n)
    const xs = new Float32Array(n)
    const ys = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      const [x, y, z] = projected(bench, i, w, h)
      xs[i] = x
      ys[i] = y
      depth[i] = z
      order[i] = i
    }
    order.sort((a, b) => depth[a] - depth[b])
    drawBenchAxes(ctx, bench, w, h, colours[3], style)
    const openDoc = rt.open
    const focusRow = rt.focus
    for (const i of order) {
      const t = (depth[i] + 1) / 2
      const r = bench.level === 'docs' ? 1.2 + t * 2.2 : 0.6 + t * 1.4
      ctx.globalAlpha = 0.25 + t * 0.65
      ctx.fillStyle = colours[bench.kinds![i]] ?? ink
      ctx.beginPath()
      ctx.arc(xs[i], ys[i], r, 0, Math.PI * 2)
      ctx.fill()
    }
    const mark = bench.level === 'docs' ? openDoc : focusRow
    if (mark >= 0 && mark < n) {
      ctx.globalAlpha = 1
      ctx.strokeStyle = ink
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(xs[mark], ys[mark], 6, 0, Math.PI * 2)
      ctx.stroke()
    }
    if (bench.hover >= 0 && bench.hover < n) {
      ctx.globalAlpha = 1
      ctx.strokeStyle = colours[1]
      ctx.beginPath()
      ctx.arc(xs[bench.hover], ys[bench.hover], 5, 0, Math.PI * 2)
      ctx.stroke()
    }
    // the three axis names on one line along the floor, in the page's mono face (the canvas
    // inherits it from latent.scss), on a paper halo so the point cloud does not run through
    // the glyphs. The variance is in the table above, so only the names
    ctx.globalAlpha = 1
    ctx.font = `${style.fontSize} ${style.fontFamily}`
    ctx.textBaseline = 'alphabetic'
    ctx.lineJoin = 'round'
    ctx.lineWidth = 3
    ctx.strokeStyle = paper
    ctx.fillStyle = ink
    const names = bench.axes.map(
      (a, slot) => `${'xyz'[slot]} ${axisBase(bench.meta?.axes[a].name ?? '')}`,
    )
    // the names read whole on one line: the type steps down to fit (to 80 %), and only a line
    // still too long cuts its longest name
    const room = w - 12
    const widthAt = (list: string[]) =>
      list.reduce((sum, name) => sum + ctx.measureText(name).width, 0) +
      ctx.measureText('  ').width * 2
    const base = parseFloat(style.fontSize) || 11
    const size = Math.max(base * 0.8, Math.min(base, (base * room) / widthAt(names)))
    ctx.font = `${size}px ${style.fontFamily}`
    const gap = ctx.measureText('  ').width
    const fitted = [...names]
    for (let over = widthAt(fitted) - room; over > 0; over = widthAt(fitted) - room) {
      let longest = 0
      for (let i = 1; i < 3; i++)
        if (ctx.measureText(fitted[i]).width > ctx.measureText(fitted[longest]).width) longest = i
      const next = fitText(ctx, fitted[longest], ctx.measureText(fitted[longest]).width - over)
      if (next === fitted[longest]) break
      fitted[longest] = next
    }
    let x = 6
    for (const text of fitted) {
      ctx.strokeText(text, x, h - 6)
      ctx.fillText(text, x, h - 6)
      x += ctx.measureText(text).width + gap
    }
  })
}

/** "pc1 · 6.2 %" → "pc1"; the variance has its own column in the controls table. */
function axisBase(name: string): string {
  return name.replace(/\s+·\s+[\d.]+\s*%$/, '')
}

function fitText(ctx: CanvasRenderingContext2D, text: string, room: number): string {
  if (ctx.measureText(text).width <= room) return text
  let cut = text.length
  while (cut > 2 && ctx.measureText(`${text.slice(0, cut)}…`).width > room) cut--
  return `${text.slice(0, cut)}…`
}

/**
 * The three axes through the origin, drawn under the points: a hairline from end to end with
 * ticks at the half and full extent, and the axis letter past the positive end.
 */
function drawBenchAxes(
  ctx: CanvasRenderingContext2D,
  bench: Bench,
  w: number,
  h: number,
  colour: string,
  style: CSSStyleDeclaration,
): void {
  const e = bench.extent
  const ends: [number, number, number][] = [
    [e, 0, 0],
    [0, e, 0],
    [0, 0, e],
  ]
  const [ox, oy] = projectPoint(bench, 0, 0, 0, w, h)
  ctx.save()
  ctx.globalAlpha = 0.7
  ctx.strokeStyle = colour
  ctx.fillStyle = colour
  ctx.lineWidth = 1
  ctx.font = `${style.fontSize} ${style.fontFamily}`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ends.forEach(([ex, ey, ez], slot) => {
    const at = (t: number) => projectPoint(bench, ex * t, ey * t, ez * t, w, h)
    const [ax, ay] = at(-1)
    const [bx, by] = at(1)
    ctx.beginPath()
    ctx.moveTo(ax, ay)
    ctx.lineTo(bx, by)
    ctx.stroke()
    // ticks perpendicular to the axis as drawn; a foreshortened axis keeps short ticks
    const len = Math.hypot(bx - ax, by - ay)
    if (len > 8) {
      const nx = -(by - ay) / len
      const ny = (bx - ax) / len
      ctx.beginPath()
      for (const t of [-1, -0.5, 0.5, 1]) {
        const [tx, ty] = at(t)
        ctx.moveTo(tx - nx * 3, ty - ny * 3)
        ctx.lineTo(tx + nx * 3, ty + ny * 3)
      }
      ctx.stroke()
    }
    // the letter just past the positive end, pushed out from the origin
    const dx = bx - ox
    const dy = by - oy
    const d = Math.hypot(dx, dy) || 1
    ctx.fillText('xyz'[slot], bx + (dx / d) * 9, by + (dy / d) * 9)
  })
  ctx.restore()
}

// ---- math ------------------------------------------------------------------------------

// The build rendered the first screen's math. Parts, documents and link extracts mounted later
// carry their TeX in `.math-inline` / `.math-display` spans; KaTeX comes from the CDN the first
// time one appears, so the runtime stays small, and its stylesheet is the page-scoped sheet the
// Latex plugin registers (loadPageStyles links it once `.katex` markup exists).
const KATEX = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist'

interface Katex {
  render(tex: string, element: HTMLElement, options: Record<string, unknown>): void
}

let katexLoad: Promise<Katex> | null = null

function loadKatex(): Promise<Katex> {
  if (!katexLoad) {
    const url = `${KATEX}/katex.mjs`
    katexLoad = import(url).then((module: { default: Katex }) => module.default)
    katexLoad.catch(() => {
      katexLoad = null
    })
  }
  return katexLoad
}

const MATH = '.math-inline, .math-display'

async function renderMath(rt: Runtime): Promise<void> {
  if (!rt.root.querySelector(MATH)) return
  const katex = await loadKatex().catch(() => null)
  if (!katex || rt.signal.aborted) return
  keepLine(chunkUnderLine(rt), () => {
    for (const el of rt.root.querySelectorAll<HTMLElement>(MATH)) {
      const displayMode = el.classList.contains('math-display')
      el.classList.remove('math-inline', 'math-display')
      try {
        katex.render(el.textContent ?? '', el, {
          displayMode,
          output: 'html',
          throwOnError: false,
          strict: 'ignore',
          macros: { ...rt.macros },
        })
      } catch {
        // malformed TeX stays as its source
      }
    }
  })
  void loadPageStyles(rt.root)
}

/** Render math whenever the reader or links gain markup: part fills, documents, a new focus. */
function watchMath(rt: Runtime): void {
  let queued = false
  const observer = new MutationObserver(() => {
    if (queued) return
    queued = true
    queueMicrotask(() => {
      queued = false
      void renderMath(rt)
    })
  })
  observer.observe(rt.reader, { childList: true, subtree: true })
  observer.observe(rt.links, { childList: true, subtree: true })
  rt.signal.addEventListener('abort', () => observer.disconnect())
}

// ---- mount -----------------------------------------------------------------------------

function parseMacros(raw: string | undefined): Record<string, string> {
  try {
    return raw ? (JSON.parse(raw) as Record<string, string>) : {}
  } catch {
    return {}
  }
}

export function mountLatent(root: HTMLElement): void {
  if (root.dataset.latentReady !== undefined) return
  root.dataset.latentReady = ''
  const signal = rootNavSignal(root)
  const rail = root.querySelector<HTMLElement>('.lat-rail')
  const reader = root.querySelector<HTMLElement>('.lat-reader')
  const links = root.querySelector<HTMLElement>('.lat-links')
  const probe = root.querySelector<HTMLElement>('.lat-probe')
  if (!rail || !reader || !links || !probe) return
  const rt: Runtime = {
    root,
    base: root.dataset.latentBase!,
    k: Number(root.dataset.latentK),
    block: Number(root.dataset.latentBlock),
    axes: Number(root.dataset.latentAxes),
    kinds: (root.dataset.latentKinds ?? '').split(',').filter(Boolean) as LatentKind[],
    scope: (root.dataset.latentScope ?? '').split(',').filter(Boolean),
    signal,
    docs: [],
    track: [],
    knn: new Map(),
    prev: new Map(),
    text: new Map(),
    parts: new Map(),
    mounting: new Map(),
    regions: new Map(),
    docByRow: [],
    focus: Number(root.dataset.latentFocus),
    open: Number(root.dataset.latentDoc),
    rail,
    reader,
    macros: parseMacros(root.dataset.latentMacros),
    links,
    probe,
    bench: null,
    vec: null,
    semantic: null,
    model: root.dataset.latentModel ?? '',
    query: null,
    railLayout: null,
    railHeight: RAIL_HEIGHT,
    railItems: [],
    toast: null,
    wanted: '',
  }
  bindSilhouettes(rt)
  watchMath(rt)
  void Promise.all([
    fetchJson<LatentDoc[]>(rt, `${rt.base}/docs.json`),
    fetchBytes(rt, `${rt.base}/track.bin`),
  ])
    .then(([docs, trackBytes]) => {
      if (signal.aborted) return
      rt.docs = docs
      rt.track = decodeTrack(trackBytes, rt.kinds)
      const byRow: LatentDoc[] = Array.from({ length: Number(root.dataset.latentRows) })
      for (const doc of docs) for (let r = doc.r0; r < doc.r0 + doc.n; r++) byRow[r] = doc
      rt.docByRow = byRow
      bindReader(rt)
      bindClicks(rt)
      bindSearch(rt)
      bindRail(rt)
      root.dataset.latentLive = ''
      // the probe opens on the 3-d bench: its vectors wait for an idle moment after the reader
      if (rt.probe.querySelector('.lat-tab[data-tab="bench"][aria-selected="true"]')) {
        const open = () => {
          if (!signal.aborted) void openBench(rt)
        }
        if ('requestIdleCallback' in window) requestIdleCallback(open, { timeout: 2000 })
        else setTimeout(open, 300)
      }
      const param = new URL(window.location.href).searchParams.get('focus')
      const wanted = param === null ? NaN : Number(param)
      if (Number.isInteger(wanted) && wanted >= 0 && wanted !== rt.focus && wanted < byRow.length)
        void goTo(rt, wanted)
    })
    .catch(error => {
      if (!signal.aborted) console.warn('[latent] hydration failed', error)
    })
}
