import { toHtml } from 'hast-util-to-html'
import type { SemanticEmbedRequest, SemanticWorkerResponse } from '../../../util/semantic-protocol'
import type { ToastHandle } from '../toast'
import {
  filterLabel,
  keepsAll,
  parseFilter,
  rowMask,
  withFilter,
} from '../../../util/latent/filter'
import {
  completeGoto,
  gotoIndex,
  gotoPane,
  matchDocs,
  parseGoto,
  type GotoIndex,
} from '../../../util/latent/goto'
import { railSvg, type LinkItem } from '../../../util/latent/markup'
import { RAIL_WIDTH } from '../../../util/latent/rail'
import { KIND_ORDER, inScope, type LatentDoc, type LatentKind } from '../../../util/latent/types'
import { setStatusText } from '../status-flap'
import { showAxis } from './axes-carousel'
import { drawBench } from './bench-canvas'
import { bindChunkPreview } from './chunk-preview'
import { linkItems, renderDocumentLinks, stepDocumentLinks } from './document-list'
import { goTo, reframeReader } from './reader'
import { knnOf, metaOf, requestVec, type Runtime } from './runtime'
import { bindSearchCompletion } from './search-completion'
import { saveView } from './view'

export async function renderLinks(rt: Runtime, row: number): Promise<void> {
  const links = await knnOf(rt, row)
  if (rt.signal.aborted || rt.focus !== row) return
  let items = linkItems(
    rt,
    links.neighbours.map(n => [n.r, n.score]),
  )
  // A typed query or open goto list holds the pane until it is cleared.
  const owns = !rt.query && rt.links.dataset.goto === undefined
  const filter = rt.filter
  if (filter) {
    const kept = items.filter(item => filter.kinds.includes(item.doc.s))
    // the shown kinds drop the rare ones, so the precomputed links mostly survive them. A typed
    // `filter:`, or a focus among hidden kinds (its links are mostly its own kind), scores the
    // focus against the whole store instead
    if (owns && (filter.typed || kept.length < Math.ceil(rt.k / 2))) {
      const rows = await filteredNeighbours(rt, row).catch(error => {
        console.warn('[latent] filter', error)
        return null
      })
      if (rt.signal.aborted || rt.focus !== row || rt.filter !== filter) return
      items = rows ? linkItems(rt, rows) : kept
    } else items = kept
  }
  // the rail measures its pane, so it goes before the list's writes: one restyle, not two
  renderRail(rt, rt.query ? rt.query.items : items)
  if (owns) {
    const focusPreview = linkItems(rt, [[row, 255]])[0]?.preview
    if (focusPreview)
      renderDocumentLinks(
        rt,
        focusPreview,
        items,
        filter?.typed ? `links · ${filterLabel(filter.kinds)}` : undefined,
      )
    rt.links.dataset.focus = String(row)
  }
  // probe ticks
  const ticks = rt.probe.querySelectorAll<SVGLineElement>('.lat-axis .lat-focus-tick')
  rt.probe.querySelectorAll<HTMLElement>('.lat-axis').forEach(card => {
    // the carousel puts the pcs first; the index is the axis's place in probe.json
    const value = links.axes[Number(card.dataset.axisIndex)]
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
  if (rt.bench) drawBench(rt)
  rt.root.dispatchEvent(new CustomEvent('latent:focus', { detail: row }))
}

const ROWS_URL = (rt: Runtime) => `${rt.base}/vec/rows-256-int8.bin`

/**
 * Keep `kinds` for the box's `filter:`, or the shown kinds (the toggles under the sources) when
 * the box names none; true when that changed what the pane may show.
 */
function setFilter(rt: Runtime, kinds: LatentKind[] | null): boolean {
  const want = kinds ?? rt.view ?? null
  const next = want && !keepsAll(want, rt.kinds) ? want : null
  const typed = kinds !== null && next !== null
  syncKinds(rt, next ?? rt.kinds, kinds !== null)
  if (
    (next?.join(',') ?? '') === (rt.filter?.kinds.join(',') ?? '') &&
    typed === !!rt.filter?.typed
  )
    return false
  rt.filter = next ? { kinds: next, mask: null, typed } : null
  return true
}

/**
 * The toggles show what the links keep; `typed` marks kinds the box's `filter:` named. Written
 * only on change: every attribute write restyles the page (the site's :has() rules).
 */
function syncKinds(rt: Runtime, shown: LatentKind[], typed: boolean): void {
  const bar = rt.probe.querySelector<HTMLElement>('.lat-kinds')
  if (!bar) return
  if ((bar.dataset.typed !== undefined) !== typed) bar.toggleAttribute('data-typed', typed)
  for (const button of bar.querySelectorAll<HTMLButtonElement>('.lat-kind')) {
    const pressed = String(shown.includes(button.dataset.kind as LatentKind))
    if (button.getAttribute('aria-pressed') !== pressed)
      button.setAttribute('aria-pressed', pressed)
  }
}

/**
 * The toggles under the probe's sources: which kinds the explorer shows. With no `filter:`
 * in the box they set the shown kinds for the links, the reader and the rail, kept for the next
 * visit; with one, they rewrite it, so the box and the toggles never disagree.
 */
function bindKinds(rt: Runtime, input: HTMLInputElement): void {
  const bar = rt.probe.querySelector<HTMLElement>('.lat-kinds')
  if (!bar) return
  bar.addEventListener(
    'click',
    event => {
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>('.lat-kind')
      const kind = button?.dataset.kind as LatentKind | undefined
      if (!kind || !rt.kinds.includes(kind)) return
      const shown = rt.filter?.kinds ?? rt.kinds
      const next = shown.includes(kind)
        ? shown.filter(k => k !== kind)
        : KIND_ORDER.filter(k => k === kind || shown.includes(k))
      // the last kind stays on: a pane that shows nothing has nothing to come back from
      if (next.length === 0) return
      if (parseFilter(input.value).kinds) input.value = withFilter(input.value, next)
      else {
        rt.view = next
        saveView(next)
        reframeReader(rt)
        renderRail(
          rt,
          rt.railItems.filter(item => next.includes(item.doc.s)),
        )
      }
      // the box reads itself again: the links, a query or a goto list follow the new kinds
      rt.wanted = ''
      input.dispatchEvent(new Event('input'))
    },
    { signal: rt.signal },
  )
}

/** The worker's row mask for the box's `filter:`, or undefined when it keeps every row. */
export async function filterMask(rt: Runtime): Promise<Uint8Array | undefined> {
  const filter = rt.filter
  if (!filter) return undefined
  if (!filter.mask) {
    const meta = await metaOf(rt)
    filter.mask = rowMask(rt.docByRow, filter.kinds, meta.rows)
  }
  return filter.mask
}

/** The precomputed links keep at most this many chunks of the focus's own document. */
const SAME_DOC = 4
/** Chunks a filtered list scores, so its documents fill the pane; and chunks kept per document. */
const FILL_K = 384
const FILL_PER_DOC = 8

/** Documents the links pane shows without scrolling, plus a few to scroll into. */
function paneDocs(rt: Runtime): number {
  const row =
    rt.links.querySelector<HTMLElement>('.lat-link-document')?.getBoundingClientRect().height || 48
  return Math.max(rt.k, Math.ceil(rt.links.clientHeight / row) + 4)
}

/**
 * Hits in score order until `docs` documents are in: at most FILL_PER_DOC chunks each, and
 * SAME_DOC of the focus's own document (`own`).
 */
function fillDocs(
  rt: Runtime,
  hits: Uint32Array,
  scores: Float32Array,
  docs: number,
  own = -1,
): [number, number][] {
  const per = new Map<number, number>()
  const out: [number, number][] = []
  for (let i = 0; i < hits.length; i++) {
    const d = rt.docByRow[hits[i]]?.d
    if (d === undefined) continue
    const n = per.get(d) ?? 0
    if (n === 0 && per.size >= docs) continue
    if (n >= (d === own ? SAME_DOC : FILL_PER_DOC)) continue
    per.set(d, n + 1)
    out.push([hits[i], Math.round(Math.max(0, scores[i]) * 255)])
  }
  return out
}

/**
 * The focus row's nearest rows of the filtered kinds, scored over the whole matrix: the
 * precomputed links (k=16) hold too few of a rare kind to filter after the fact.
 */
async function filteredNeighbours(rt: Runtime, row: number): Promise<[number, number][]> {
  const meta = await metaOf(rt)
  const url = ROWS_URL(rt)
  await requestVec(rt, { type: 'load', url, rows: meta.rows, dims: 256 })
  const got = await requestVec(rt, { type: 'row', url, row, scale: meta.scale })
  const mask = await filterMask(rt)
  if (got.type !== 'row' || !mask) return []
  const only = mask.slice()
  only[row] = 0
  const result = await requestVec(
    rt,
    { type: 'query', url, vector: got.vector, scale: meta.scale, k: FILL_K, only },
    [got.vector.buffer, only.buffer],
  )
  if (result.type !== 'queried') return []
  return fillDocs(rt, result.hits, result.scores, paneDocs(rt), rt.docByRow[row]?.d)
}

function linksBody(rt: Runtime): HTMLElement {
  return rt.links.querySelector<HTMLElement>('.lat-links-body') ?? rt.links
}

interface Encoder {
  seq: number
  /** a request is with the worker; it answers requests one at a time, in order */
  busy: boolean
  /** the newest text typed while the worker was busy; any older one waiting was dropped */
  next: { text: string; resolve: (v: Float32Array) => void; reject: (e: Error) => void } | null
  status: (text: string) => void
  /** phrase → vector, newest last; backspacing to an earlier phrase skips the encoder */
  cache: Map<string, Float32Array>
}

const encoders = new WeakMap<Runtime, Encoder>()
const ENCODER_CACHE = 48

class Superseded extends Error {}

/**
 * The site's query encoder, in its own worker; loads on the first query. The worker queues every
 * request and runs them in order, so text typed while it works waits here instead: only the
 * newest waits, older ones are dropped, and a pause in typing costs one encoding, not a backlog.
 */
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
  let enc = encoders.get(rt)
  if (!enc) {
    enc = { seq: 0, busy: false, next: null, status, cache: new Map() }
    encoders.set(rt, enc)
  }
  const state = enc
  state.status = status
  const hit = state.cache.get(text)
  if (hit) {
    state.cache.delete(text)
    state.cache.set(text, hit)
    // the caller hands its copy's buffer to the vectors worker
    return Promise.resolve(hit.slice())
  }
  const send = (
    text: string,
    resolve: (v: Float32Array) => void,
    reject: (e: Error) => void,
  ): void => {
    state.busy = true
    const seq = ++state.seq
    const settle = () => {
      worker.removeEventListener('message', onMessage)
      state.busy = false
      const queued = state.next
      state.next = null
      if (queued && !rt.signal.aborted) send(queued.text, queued.resolve, queued.reject)
    }
    const onMessage = (event: MessageEvent<SemanticWorkerResponse>) => {
      const data = event.data
      if (data.type === 'model-progress') {
        // `phase · loaded/total unit` so the status flap swaps digits in place
        state.status(
          data.phase === 'loading' && data.totalBytes
            ? `encoder · ${megabytes(data.loadedBytes)}/${megabytes(data.totalBytes)} MB`
            : 'starting the encoder…',
        )
        return
      }
      if (data.type === 'embedding' && data.seq === seq) {
        state.cache.set(text, data.vector.slice())
        if (state.cache.size > ENCODER_CACHE)
          state.cache.delete(state.cache.keys().next().value as string)
        settle()
        resolve(data.vector)
      } else if (data.type === 'error' && data.seq === seq) {
        settle()
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
  }
  return new Promise((resolve, reject) => {
    if (!state.busy) {
      send(text, resolve, reject)
      return
    }
    state.next?.reject(new Superseded('superseded'))
    state.next = { text, resolve, reject }
  })
}

const firstAxis = new WeakMap<Runtime, HTMLElement | null>()

/** Probe axes whose name or pole terms contain the query: "search for features". */
function matchAxes(rt: Runtime, text: string): number {
  const needle = text.trim().toLowerCase()
  let first: HTMLElement | null = null
  let count = 0
  for (const card of rt.probe.querySelectorAll<HTMLElement>('.lat-axis')) {
    const hit = needle.length > 1 && (card.textContent ?? '').toLowerCase().includes(needle)
    if (hit !== (card.dataset.match !== undefined)) card.toggleAttribute('data-match', hit)
    if (hit) {
      count++
      first ??= card
    }
  }
  // the carousel slides when the first match changes, not on every key that keeps it
  const moved = first !== firstAxis.get(rt)
  firstAxis.set(rt, first)
  // an open features tab keeps the probe: it lists the features whose terms match. Otherwise
  // the 3-d tab opens and its axis carousel slides to the first match
  const features = rt.probe.querySelector<HTMLElement>('.lat-panel[data-panel="features"]')
  if (first && moved && (!features || features.hidden)) {
    const bench = rt.probe.querySelector<HTMLButtonElement>('.lat-tab[data-tab="bench"]')
    if (bench && bench.getAttribute('aria-selected') !== 'true') bench.click()
    showAxis(first)
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

export async function runQuery(rt: Runtime, text: string): Promise<void> {
  // `filter:` tokens narrow the hits; the rest of the box is what gets embedded
  const filter = parseFilter(text)
  const filtered = setFilter(rt, filter.kinds)
  const phrase = filter.rest
  if (!phrase) {
    clearQuery(rt, filtered)
    return
  }
  const status = searchStatus(rt)
  const axes = matchAxes(rt, phrase)
  if (rt.wanted === text && rt.query?.text === text) return
  rt.wanted = text
  const stale = () => rt.signal.aborted || rt.wanted !== text
  status.set('embedding the query…')
  try {
    const vector = await embedQuery(rt, phrase, status.set)
    if (stale()) return
    const url = ROWS_URL(rt)
    const meta = await metaOf(rt)
    if (stale()) return
    status.set(`scoring · ${meta.rows.toLocaleString('en-US')} chunks`)
    // the worker takes the buffer; the features tab encodes its own copy
    const kept = vector.slice()
    await requestVec(rt, { type: 'load', url, rows: meta.rows, dims: 256 })
    if (stale()) return
    const t0 = performance.now()
    const only = await filterMask(rt)
    if (stale()) return
    // a typed filter fills the pane with documents; otherwise the k nearest chunks of the shown
    // kinds
    const fill = !!only && !!rt.filter?.typed
    const result = await requestVec(
      rt,
      { type: 'query', url, vector, scale: meta.scale, k: fill ? FILL_K : rt.k, only },
      [vector.buffer],
    )
    if (result.type !== 'queried' || stale()) return
    const items = linkItems(
      rt,
      fill
        ? fillDocs(rt, result.hits, result.scores, paneDocs(rt))
        : [...result.hits].map((row, i) => [row, Math.round(Math.max(0, result.scores[i]) * 255)]),
    )
    rt.query = { text, items }
    // the rail measures its pane, so it goes before the list's writes: one restyle, not two
    renderRail(rt, items)
    renderDocumentLinks(
      rt,
      { r: -1, d: -1, s: 'note', t: text, c: 0, p: null, l: '', x: '' },
      items,
      rt.filter?.typed ? `query · ${filterLabel(rt.filter.kinds)}` : 'query',
    )
    rt.links.dataset.query = text
    status.done(
      `${items.length} chunks · ${(performance.now() - t0).toFixed(0)} ms${axes ? ` · ${axes} ${axes === 1 ? 'axis' : 'axes'} match` : ''}${unknownTypes(filter.unknown)}`,
    )
    rt.root.dispatchEvent(new CustomEvent('latent:query', { detail: { text, vector: kept } }))
  } catch (error) {
    if (!stale() && !(error instanceof Superseded))
      status.fail(error instanceof Error ? error.message : String(error))
  }
}

const unknownTypes = (names: string[]) =>
  names.length ? ` · no type ${names.map(n => `“${n}”`).join(', ')} (note, pdf, image, base)` : ''

/** Back to the focus's links; `force` re-renders them even with nothing pinned (a new filter). */
function clearQuery(rt: Runtime, force = false): void {
  for (const card of rt.probe.querySelectorAll<HTMLElement>('.lat-axis[data-match]'))
    delete card.dataset.match
  firstAxis.set(rt, null)
  rt.toast?.dismiss()
  rt.toast = null
  // the features tab re-renders (and steering re-fuses) on this, so an empty box that had no
  // query, or a key inside `filter:`, says nothing
  if (rt.wanted || rt.query || rt.links.dataset.query)
    rt.root.dispatchEvent(new CustomEvent('latent:query', { detail: null }))
  rt.wanted = ''
  if (!force && !rt.query && !rt.links.dataset.query && rt.links.dataset.goto === undefined) return
  rt.query = null
  delete rt.links.dataset.query
  delete rt.links.dataset.goto
  void renderLinks(rt, rt.focus)
}

/**
 * `goto:` lists documents by title, slug or path as you type (no embedding); ↑↓ pick, Enter or a
 * click opens the pick at its first row, Tab fills its full path into the box.
 */
function gotoMode(rt: Runtime, input: HTMLInputElement) {
  let index: GotoIndex | null = null
  let hits: LatentDoc[] = []
  let active = 0
  // what the pane shows now: a key that lands on the same list writes nothing
  let shown = ''
  let frame = 0
  const items = () => [...rt.links.querySelectorAll<HTMLElement>('.lat-goto-item')]
  const show = (needle: string) => {
    if (frame) cancelAnimationFrame(frame)
    frame = 0
    index ??= gotoIndex(rt.docs)
    const kinds = rt.filter?.kinds
    const keep = kinds ? (doc: LatentDoc) => kinds.includes(doc.s) : undefined
    const next = matchDocs(index, needle, 60, keep)
    const key = `${needle}\u0000${kinds?.join(',') ?? ''}\u0000${next.map(doc => doc.d).join(',')}`
    if (key === shown && linksBody(rt).querySelector('.lat-goto, .lat-goto-empty')) return
    shown = key
    hits = next
    active = 0
    // a query still embedding drops its result, and its status line goes with it
    rt.wanted = ''
    rt.toast?.dismiss()
    rt.toast = null
    for (const card of rt.probe.querySelectorAll<HTMLElement>('.lat-axis[data-match]'))
      delete card.dataset.match
    rt.links.dataset.goto = needle
    const pane = gotoPane(
      hits,
      needle,
      keep ? rt.docs.filter(keep).length : rt.docs.length,
      active,
      doc => !inScope(doc.id, rt.scope),
      kinds && rt.filter?.typed ? `goto · ${filterLabel(kinds)}` : 'goto',
    )
    // scroll first: writing scrollTop after the new list would force a restyle of the page
    if (rt.links.scrollTop) rt.links.scrollTop = 0
    linksBody(rt).innerHTML = toHtml(pane as never, { allowDangerousHtml: false })
  }
  // Typing renders the list once a frame: each render rewrites the pane, and the site's :has()
  // rules restyle the whole page for it. The box is in goto mode at once, so keys route here.
  const showSoon = (needle: string) => {
    rt.links.dataset.goto = needle
    if (frame) return
    frame = requestAnimationFrame(() => {
      frame = 0
      const now = rt.links.dataset.goto
      if (now !== undefined && !rt.signal.aborted) show(now)
    })
  }
  const leave = () => {
    if (frame) cancelAnimationFrame(frame)
    frame = 0
    shown = ''
  }
  const pick = (next: number) => {
    if (frame && rt.links.dataset.goto !== undefined) show(rt.links.dataset.goto)
    const all = items()
    if (all.length === 0) return
    active = (next + all.length) % all.length
    all.forEach((el, i) => el.toggleAttribute('data-active', i === active))
    all[active].scrollIntoView({ block: 'nearest' })
  }
  const open = async (i = active) => {
    // Enter right after a key: render that key's list first, so the pick matches the box
    if (frame && rt.links.dataset.goto !== undefined) show(rt.links.dataset.goto)
    const doc = hits[i]
    if (!doc) return
    leave()
    input.value = ''
    input.blur()
    delete rt.links.dataset.goto
    setFilter(rt, null)
    if (rt.query || rt.links.dataset.query) {
      rt.query = null
      delete rt.links.dataset.query
      rt.root.dispatchEvent(new CustomEvent('latent:query', { detail: null }))
    }
    // goTo re-renders the links through setFocus, which skips a row that already has the focus
    const same = rt.focus === doc.r0
    await goTo(rt, doc.r0)
    if (same && !rt.signal.aborted) void renderLinks(rt, rt.focus)
  }
  const fill = (): boolean => {
    if (frame && rt.links.dataset.goto !== undefined) show(rt.links.dataset.goto)
    const doc = hits[active]
    if (!doc) return false
    // Tab keeps the box's `filter:` tokens in front of the filled path
    const value = [parseFilter(input.value).tokens, completeGoto(doc)].filter(Boolean).join(' ')
    if (input.value.trim() === value) return false
    input.value = value
    show(parseGoto(parseFilter(value).rest) ?? '')
    return true
  }
  rt.links.addEventListener(
    'click',
    event => {
      const item = (event.target as HTMLElement).closest<HTMLElement>('.lat-goto-item')
      if (!item) return
      event.preventDefault()
      event.stopPropagation()
      const d = Number(item.dataset.goto)
      void open(hits.findIndex(doc => doc.d === d))
    },
    { signal: rt.signal },
  )
  return {
    active: () => rt.links.dataset.goto !== undefined,
    show,
    showSoon,
    leave,
    pick: (step: number) => pick(active + step),
    open: () => void open(),
    fill,
  }
}

const SEARCH_DEBOUNCE_MS = 450

/** The box searches as you type: axes match at once, the embedding query runs after a pause. */
export function bindSearch(rt: Runtime): void {
  const form = rt.links.querySelector<HTMLFormElement>('form.lat-search')
  const input = form?.querySelector<HTMLInputElement>('input')
  if (!form || !input) return
  let timer: number | null = null
  const cancel = () => {
    if (timer !== null) window.clearTimeout(timer)
    timer = null
  }
  bindChunkPreview(rt)
  // the shown kinds filter the links from the start; the first screen was built with notes and
  // pdfs, so a reader who picked others gets the reader's neighbours rebuilt
  setFilter(rt, null)
  if (rt.view && rt.root.dataset.latentView !== rt.view.join(',')) reframeReader(rt)
  void renderLinks(rt, rt.focus)
  bindKinds(rt, input)
  const goto = gotoMode(rt, input)
  bindSearchCompletion(rt, input, form)
  document.addEventListener(
    'keydown',
    event => {
      if (
        event.ctrlKey &&
        !event.altKey &&
        !event.metaKey &&
        !event.shiftKey &&
        !event.isComposing &&
        event.code === 'KeyK'
      ) {
        event.preventDefault()
        event.stopImmediatePropagation()
        input.focus({ preventScroll: true })
      }
    },
    { capture: true, signal: rt.signal },
  )
  const run = () => {
    cancel()
    const text = input.value.trim()
    const filter = parseFilter(text)
    const filtered = setFilter(rt, filter.kinds)
    if (parseGoto(filter.rest) !== null) goto.open()
    else if (!filter.rest) {
      clearQuery(rt, filtered)
      if (filter.unknown.length) searchStatus(rt).fail(`filter${unknownTypes(filter.unknown)}`)
    }
    // `#…` searches features (features.ts); Enter selects the best match now
    else if (text.startsWith('#'))
      rt.root.dispatchEvent(new CustomEvent('latent:feature-search', { detail: text }))
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
      if (event.defaultPrevented || event.isComposing) return
      const step =
        event.ctrlKey && !event.altKey && !event.metaKey && !event.shiftKey
          ? event.key.toLowerCase() === 'n'
            ? 1
            : event.key.toLowerCase() === 'p'
              ? -1
              : 0
          : 0
      if (step) {
        event.preventDefault()
        event.stopPropagation()
        if (goto.active()) goto.pick(step)
        else {
          stepDocumentLinks(rt, step)
        }
      } else if (
        event.key === 'Enter' &&
        !event.altKey &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey
      ) {
        event.preventDefault()
        run()
      } else if (event.key === 'Escape') {
        cancel()
        goto.leave()
        input.value = ''
        clearQuery(rt, setFilter(rt, null))
        input.blur()
      } else if (goto.active() && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
        event.preventDefault()
        goto.pick(event.key === 'ArrowDown' ? 1 : -1)
      } else if (goto.active() && event.key === 'Tab' && !event.shiftKey) {
        if (goto.fill()) event.preventDefault()
      }
    },
    { signal: rt.signal },
  )
  input.addEventListener(
    'input',
    () => {
      cancel()
      const text = input.value.trim()
      const filter = parseFilter(text)
      const filtered = setFilter(rt, filter.kinds)
      const needle = parseGoto(filter.rest)
      if (needle !== null) {
        goto.showSoon(needle)
        return
      }
      goto.leave()
      // `filter:` alone narrows the focus's links at once; with words it waits for the query
      if (!filter.rest) {
        clearQuery(rt, filtered)
        return
      }
      // leaving goto: the focus's links come back until the new query settles
      if (goto.active()) {
        delete rt.links.dataset.goto
        void renderLinks(rt, rt.focus)
      }
      if (text.startsWith('#')) return
      matchAxes(rt, filter.rest)
      timer = window.setTimeout(run, SEARCH_DEBOUNCE_MS)
    },
    { signal: rt.signal },
  )
}

/** Size the rail's viewBox to its pane so the track runs the full height; true when it changed. */
export function measureRail(rt: Runtime): boolean {
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

export function renderRail(rt: Runtime, items: LinkItem[]): void {
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
    hide: rt.view ? rt.kinds.filter(kind => !rt.view!.includes(kind)) : undefined,
  })
  const current = rt.rail.querySelector('svg.lat-track')
  const html = toHtml(svg as never)
  if (current) current.outerHTML = html
  else rt.rail.insertAdjacentHTML('beforeend', html)
}
