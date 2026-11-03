import { RAIL_HEIGHT } from '../../util/latent/rail'
import { decodeTrack, type LatentDoc, type LatentKind } from '../../util/latent/types'
import { openBench } from './latent-client/bench'
import { bindFeatures } from './latent-client/features'
import { bindRail } from './latent-client/rail'
import { bindReader, chunkUnderLine, goTo, keepLine } from './latent-client/reader'
import { fetchBytes, fetchJson, partOf, textOf, type Runtime } from './latent-client/runtime'
import { bindSearch } from './latent-client/search'
import { loadPageStyles } from './page-styles'
import { rootNavSignal } from './root-lifecycle'

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
      bindFeatures(rt)
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
