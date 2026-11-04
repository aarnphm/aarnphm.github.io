import { RAIL_HEIGHT } from '../../util/latent/rail'
import { decodeTrack, type LatentDoc, type LatentKind } from '../../util/latent/types'
import { openBench } from './latent-client/bench'
import { bindFeatures } from './latent-client/features'
import { bindCallouts, claimTiles, renderFigures } from './latent-client/figures'
import { bindFolios } from './latent-client/folio'
import { bindMarquee } from './latent-client/marquee'
import { bindProbeShortcuts } from './latent-client/probe-shortcuts'
import { bindRail } from './latent-client/rail'
import { bindReader, chunkUnderLine, goTo, keepLine } from './latent-client/reader'
import { fetchBytes, fetchJson, type Runtime } from './latent-client/runtime'
import { bindSearch } from './latent-client/search'
import { bindTips } from './latent-client/tips'
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
        // Virtualized results can be outside the mounted list.
        const item = rt.railItems.find(item => item.doc.d === d)
        const row = item?.preview.r ?? rt.docs[d]?.r0
        if (row !== undefined) void goTo(rt, row)
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

// ---- code -------------------------------------------------------------------------------------

// The build highlighted the first screen's code. Blocks mounted later name their language in
// `data-lang`; the site's shared highlighter chunk (the one comments and previews use) loads the
// first time one appears and writes the note pages' token markup, so syntax.scss colours it.
const CODE = 'pre.lat-code[data-lang] > code:not([data-theme])'

async function renderCode(rt: Runtime): Promise<void> {
  const blocks = rt.root.querySelectorAll<HTMLElement>(CODE)
  if (blocks.length === 0) return
  const highlighter = await import('../../util/highlight-source').catch(() => null)
  if (!highlighter || rt.signal.aborted) return
  for (const code of blocks) {
    const pre = code.parentElement
    if (!pre?.isConnected || code.hasAttribute('data-theme')) continue
    const out = highlighter.highlightCode(code.textContent ?? '', pre.dataset.lang ?? '')
    if (out.language === 'plain') {
      pre.removeAttribute('data-lang')
      continue
    }
    for (const [name, value] of Object.entries(highlighter.codeAttributes(out)))
      code.setAttribute(name, value)
    // a passage is an excerpt: numbering from 1 would misstate where it sits in the file
    code.dataset.disableLineNumber = ''
    code.innerHTML = out.html
  }
}

/**
 * Math, code and figures whenever the reader or links gain markup: part fills, documents, a new
 * focus.
 */
function watchMath(rt: Runtime): void {
  let queued = false
  const observer = new MutationObserver(() => {
    if (queued) return
    queued = true
    queueMicrotask(() => {
      queued = false
      void renderMath(rt)
      void renderCode(rt)
      renderFigures(rt)
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
  bindCallouts(rt)
  bindFolios(rt)
  claimTiles(rt)
  watchMath(rt)
  renderFigures(rt)
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
      bindProbeShortcuts(rt)
      bindMarquee(rt, rt.probe)
      bindTips(rt)
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
