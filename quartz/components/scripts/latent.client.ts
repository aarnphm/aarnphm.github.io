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
import type { LatentWorkerRequest, LatentWorkerResponse } from '../../workers/latent.worker'
import {
  chunkArticles,
  linksPane,
  moreSentinel,
  railSvg,
  readerDocument,
} from '../../util/latent/markup'
import {
  blockName,
  blockOf,
  decodeKnnBlock,
  decodeTrack,
  inScope,
  partName,
} from '../../util/latent/types'
import { loadPageStyles } from './page-styles'
import { rootNavSignal } from './root-lifecycle'

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

/** Keep the text under the reading line still when something above it grows or shrinks. */
function keepLine(anchor: HTMLElement | null, mutate: () => void): void {
  const before = anchor?.getBoundingClientRect().top ?? 0
  mutate()
  if (!anchor || !anchor.isConnected) return
  const moved = anchor.getBoundingClientRect().top - before
  if (Math.abs(moved) > 0.5) window.scrollBy(0, moved)
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
  const pane = linksPane(focusPreview, items, rt.kinds)
  rt.links.innerHTML = toHtml(pane.children as never, { allowDangerousHtml: false })
  rt.links.dataset.focus = String(row)
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
  renderRail(rt, items)
  if (rt.bench) drawBench(rt)
}

function renderRail(rt: Runtime, items: LinkItem[]): void {
  const open = rt.open
  const neighbours = new Map<number, number>()
  for (const item of items) {
    const best = neighbours.get(item.doc.d) ?? 0
    if (item.score > best) neighbours.set(item.doc.d, item.score)
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
    focusChunkIndex,
    scopeFlags: flags,
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

/** Mount a document into the reader (replacing its stub), keeping one stub either side. */
async function mountDocument(
  rt: Runtime,
  d: number,
  part: number | 'last' = 0,
): Promise<HTMLElement | null> {
  const existing = rt.reader.querySelector<HTMLElement>(
    `.lat-doc[data-doc="${d}"]:not(.lat-doc-stub)`,
  )
  if (existing) return existing
  const head = await textOf(rt, d)
  const index =
    part === 'last' ? Math.max(0, Math.ceil(head.total / Math.max(1, head.part)) - 1) : part
  const html = await documentHtml(rt, d, index, rt.focus)
  if (rt.signal.aborted) return null
  const stub = rt.reader.querySelector<HTMLElement>(`.lat-doc[data-doc="${d}"]`)
  const anchor = chunkUnderLine(rt)
  // the previous document mounted above must not move the text under the reading line
  keepLine(anchor, () => {
    if (stub) stub.outerHTML = html
    else rt.reader.insertAdjacentHTML('beforeend', html)
  })
  return rt.reader.querySelector<HTMLElement>(`.lat-doc[data-doc="${d}"]`)
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
  if (lo > 0 && !rt.reader.querySelector(`.lat-doc[data-doc="${lo - 1}"]`))
    rt.reader.insertAdjacentHTML('afterbegin', stubHtml(lo - 1))
  if (hi + 1 < rt.docs.length && !rt.reader.querySelector(`.lat-doc[data-doc="${hi + 1}"]`))
    rt.reader.insertAdjacentHTML('beforeend', stubHtml(hi + 1))
  // unmount documents two away from the open one
  for (const el of rt.reader.querySelectorAll<HTMLElement>('.lat-doc')) {
    const d = Number(el.dataset.doc)
    if (Math.abs(d - rt.open) > 2) el.remove()
  }
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
    rt.reader.innerHTML = [prev ? stub(prev) : '', html, next ? stub(next) : ''].join('')
    chunk = rt.reader.querySelector<HTMLElement>(`.lat-chunk[data-row="${row}"]`)
  }
  if (!chunk) return
  const top = chunk.getBoundingClientRect().top + window.scrollY - readingLineY() + 8
  window.scrollTo({ top, behavior: 'auto' })
  setFocus(rt, row)
}

function bindReader(rt: Runtime): void {
  let frame = 0
  const update = () => {
    frame = 0
    const chunk = chunkUnderLine(rt)
    if (!chunk) return
    const row = Number(chunk.dataset.row)
    if (row !== rt.focus) setFocus(rt, row)
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

function bindSilhouettes(rt: Runtime): void {
  rt.links.addEventListener(
    'pointerover',
    event => {
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
  const worker = new Worker(new URL('latent.worker.js', import.meta.url), { type: 'module' })
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
  }
  rt.bench = bench
  rt.signal.addEventListener('abort', () => worker.terminate())
  const meta = await fetchJson<NonNullable<Bench['meta']>>(rt, `${rt.base}/vec/meta.json`)
  bench.meta = meta
  bench.kinds = new Uint8Array(rt.docs.length)
  for (const doc of rt.docs) bench.kinds[doc.d] = KIND_INDEX[doc.s]
  controls.innerHTML =
    [0, 1, 2]
      .map(
        slot =>
          `<label><span>${'xyz'[slot]}</span><select data-slot="${slot}">${meta.axes
            .map(
              (axis, i) =>
                `<option value="${i}"${i === bench.axes[slot] ? ' selected' : ''}>${axis.name}</option>`,
            )
            .join('')}</select></label>`,
      )
      .join('') +
    '<button type="button" class="lat-bench-level" data-level="rows">chunk level</button>'
  controls.addEventListener(
    'change',
    event => {
      const select = (event.target as HTMLElement).closest<HTMLSelectElement>('select[data-slot]')
      if (!select) return
      bench.axes[Number(select.dataset.slot)] = Number(select.value)
      void projectBench(rt)
    },
    { signal: rt.signal },
  )
  controls.addEventListener(
    'click',
    event => {
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>('.lat-bench-level')
      if (!button) return
      const level = button.dataset.level === 'rows' ? 'rows' : 'docs'
      button.dataset.level = level === 'rows' ? 'docs' : 'rows'
      button.textContent = level === 'rows' ? 'document level' : 'chunk level'
      void loadBench(rt, level)
    },
    { signal: rt.signal },
  )
  bindBenchPointer(rt, canvas)
  await loadBench(rt, 'docs')
}

function request(
  bench: Bench,
  message: LatentWorkerRequest,
  transfer: Transferable[] = [],
): Promise<LatentWorkerResponse> {
  return new Promise((resolve, reject) => {
    const seq = message.seq
    const onMessage = (event: MessageEvent<LatentWorkerResponse>) => {
      if (event.data.seq !== seq) return
      bench.worker.removeEventListener('message', onMessage)
      if (event.data.type === 'error') reject(new Error(event.data.message))
      else resolve(event.data)
    }
    bench.worker.addEventListener('message', onMessage)
    bench.worker.postMessage(message, transfer)
  })
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
  await request(bench, { type: 'load', seq: ++bench.seq, url, rows, dims: 256 })
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
  const result = await request(
    bench,
    { type: 'project', seq: ++bench.seq, dirs, mean, scale: bench.meta.scale },
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
        hoverBench(
          rt,
          (event.clientX - rect.left) * (canvas.width / rect.width),
          (event.clientY - rect.top) * (canvas.height / rect.height),
        )
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
  const s = (Math.min(w, h) * 0.42 * bench.zoom) / bench.extent
  const x = c[i * 3]
  const y = c[i * 3 + 1]
  const z = c[i * 3 + 2]
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
  const canvas = rt.probe.querySelector<HTMLCanvasElement>('.lat-bench-canvas')
  if (!bench || !bench.coords || !canvas) return
  const n = bench.coords.length / 3
  let best = -1
  let bestDist = 36
  for (let i = 0; i < n; i++) {
    const [x, y] = projected(bench, i, canvas.width, canvas.height)
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
      status.textContent = `${doc.s} · ${doc.t}`
    }
  }
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
    const w = canvas.width
    const h = canvas.height
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
    // axis names at the frame edges
    ctx.globalAlpha = 0.8
    ctx.fillStyle = ink
    ctx.font = '10px ui-monospace, monospace'
    const names = bench.axes.map(a => bench.meta?.axes[a].name ?? '')
    ctx.fillText(`x ${names[0]}`, 6, h - 18)
    ctx.fillText(`y ${names[1]}`, 6, h - 6)
    ctx.fillText(`z ${names[2]}`, w / 2, h - 6)
    ctx.globalAlpha = 1
  })
}

// ---- math ------------------------------------------------------------------------------

// The build rendered the first screen's math. Parts, documents and link extracts mounted later
// carry their TeX in `.math-inline` / `.math-display` spans; KaTeX comes from the CDN the first
// time one appears, so the runtime stays small, and its stylesheet is the page-scoped sheet the
// Latex plugin registers (loadPageStyles links it once `.katex` markup exists).
const KATEX = 'https://cdn.jsdelivr.net/npm/katex@0.17.0/dist'

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
      root.dataset.latentLive = ''
      const param = new URL(window.location.href).searchParams.get('focus')
      const wanted = param === null ? NaN : Number(param)
      if (Number.isInteger(wanted) && wanted >= 0 && wanted !== rt.focus && wanted < byRow.length)
        void goTo(rt, wanted)
    })
    .catch(error => {
      if (!signal.aborted) console.warn('[latent] hydration failed', error)
    })
}
