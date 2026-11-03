import { toHtml } from 'hast-util-to-html'
import type { SemanticEmbedRequest, SemanticWorkerResponse } from '../../../util/semantic-protocol'
import type { ToastHandle } from '../toast'
import { linksPane, railSvg, type LinkItem } from '../../../util/latent/markup'
import { RAIL_WIDTH } from '../../../util/latent/rail'
import { inScope } from '../../../util/latent/types'
import { setStatusText } from '../status-flap'
import { drawBench } from './bench-canvas'
import { knnOf, metaOf, previewOf, requestVec, type Runtime } from './runtime'

export async function renderLinks(rt: Runtime, row: number): Promise<void> {
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
  rt.root.dispatchEvent(new CustomEvent('latent:focus', { detail: row }))
}

function linksBody(rt: Runtime): HTMLElement {
  return rt.links.querySelector<HTMLElement>('.lat-links-body') ?? rt.links
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
    // the worker takes the buffer; the features tab encodes its own copy
    const kept = vector.slice()
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
    rt.root.dispatchEvent(new CustomEvent('latent:query', { detail: { text, vector: kept } }))
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
  rt.root.dispatchEvent(new CustomEvent('latent:query', { detail: null }))
  if (!rt.query && !rt.links.dataset.query) return
  rt.query = null
  delete rt.links.dataset.query
  void renderLinks(rt, rt.focus)
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
  })
  const current = rt.rail.querySelector('svg.lat-track')
  const html = toHtml(svg as never)
  if (current) current.outerHTML = html
  else rt.rail.insertAdjacentHTML('beforeend', html)
}
