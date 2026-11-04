import type { Runtime } from './runtime'
import { chunkUnderLine, keepLine } from './reader'

/**
 * The garden components a note passage can hold besides math and code: callouts fold, tikz
 * figures take the picture their note page rendered, procedural drop caps grow. Telescopic text
 * needs nothing here: telescopic.inline.ts listens on the document.
 */

// ---- callouts ---------------------------------------------------------------------------------

// callout.inline.ts binds the callouts on the page at navigation, which covers the first screen's
// passages but not the ones mounted later. A click on a title is checked on the way down and on
// the way up: when the page's handler has not folded it in between, this one does.
export function bindCallouts(rt: Runtime): void {
  const TITLE = '.callout.is-collapsible > .callout-title'
  let pending: { callout: HTMLElement; collapsed: boolean } | null = null
  rt.reader.addEventListener(
    'click',
    event => {
      const title = (event.target as Element | null)?.closest?.(TITLE)
      const callout = title?.parentElement
      pending = callout ? { callout, collapsed: callout.classList.contains('is-collapsed') } : null
    },
    { capture: true, signal: rt.signal },
  )
  rt.reader.addEventListener(
    'click',
    () => {
      const seen = pending
      pending = null
      if (!seen || seen.callout.classList.contains('is-collapsed') !== seen.collapsed) return
      const collapsed = seen.callout.classList.toggle('is-collapsed')
      const content = seen.callout.querySelector<HTMLElement>(':scope > .callout-content')
      if (content) content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
    },
    { signal: rt.signal },
  )
}

// ---- tikz -------------------------------------------------------------------------------------

// A passage holds a tikz figure's source; the note page has the build's picture of it, keyed by
// that source (the figure's MathML annotation). A page is fetched once, when its first figure
// nears the screen, and a figure the page does not have shows its source.
const TIKZ = 'figure.lat-tikz[data-page]:not([data-state])'
const pages = new Map<string, Promise<Map<string, HTMLImageElement>>>()
const squash = (source: string) => source.replace(/\s+/g, ' ').trim()

function pictures(href: string): Promise<Map<string, HTMLImageElement>> {
  let found = pages.get(href)
  if (!found) {
    found = fetch(href)
      .then(response => (response.ok ? response.text() : ''))
      .then(html => {
        const out = new Map<string, HTMLImageElement>()
        const page = new DOMParser().parseFromString(html, 'text/html')
        for (const figure of page.querySelectorAll('figure.tikz')) {
          const tex = figure.querySelector('annotation')?.textContent
          const img = figure.querySelector<HTMLImageElement>(':scope > img')
          if (!tex || !img) continue
          try {
            out.set(squash(JSON.parse(tex) as string), img)
          } catch {
            // an annotation that is not the source's JSON names no figure
          }
        }
        return out
      })
      .catch(() => new Map<string, HTMLImageElement>())
    pages.set(href, found)
  }
  return found
}

async function showTikz(rt: Runtime, figure: HTMLElement): Promise<void> {
  figure.dataset.state = 'loading'
  const source = figure.querySelector('.lat-tikz-source')?.textContent ?? ''
  const img = (await pictures(figure.dataset.page!)).get(squash(source))
  if (rt.signal.aborted || !figure.isConnected) return
  keepLine(chunkUnderLine(rt), () => {
    if (!img) {
      figure.dataset.state = 'source'
      return
    }
    const picture = document.createElement('img')
    picture.src = img.getAttribute('src') ?? ''
    picture.alt = img.alt || 'tikz diagram'
    picture.decoding = 'async'
    const style = img.closest('figure')?.getAttribute('style')
    if (style) figure.setAttribute('style', style)
    figure.replaceChildren(picture)
    figure.dataset.state = 'ready'
  })
}

let tikzSeen: IntersectionObserver | null = null

function renderTikz(rt: Runtime): void {
  const figures = rt.root.querySelectorAll<HTMLElement>(TIKZ)
  if (figures.length === 0) return
  if (!tikzSeen) {
    tikzSeen = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue
          tikzSeen?.unobserve(entry.target)
          void showTikz(rt, entry.target as HTMLElement)
        }
      },
      { rootMargin: '800px 0px' },
    )
    rt.signal.addEventListener('abort', () => {
      tikzSeen?.disconnect()
      tikzSeen = null
    })
  }
  for (const figure of figures) tikzSeen.observe(figure)
}

// ---- procedural drop caps ---------------------------------------------------------------------

// procedural-initial.inline.ts grows the tiles on the page at navigation; tiles in passages
// mounted later grow here, from the same Initialis module, when they come into view.
const INITIALIS = 'https://initialis.azlen.me/lib/initialis.js'
const TILE = 'canvas.gv-initial[data-settings]:not([data-lat-tile])'

type Paint = (
  canvas: HTMLCanvasElement,
  settings: unknown,
  options: Record<string, unknown>,
) => Promise<unknown>

let initialis: Promise<{ grow: Paint; draw: Paint }> | null = null
let tileSeen: IntersectionObserver | null = null

async function paintTile(canvas: HTMLCanvasElement): Promise<void> {
  if (!initialis) {
    initialis = import(INITIALIS) as Promise<{ grow: Paint; draw: Paint }>
    initialis.catch(() => {
      initialis = null
    })
  }
  const lib = await initialis.catch(() => null)
  const raw = canvas.dataset.settings
  if (!lib || !raw || !canvas.isConnected) return
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const settings = JSON.parse(raw) as unknown
  await (
    reduced
      ? lib.draw(canvas, settings, { paper: true })
      : lib.grow(canvas, settings, { paper: true, seconds: 2.4 })
  ).catch(() => null)
  canvas.closest('.mag-initial')?.classList.add('is-drawn')
}

/** Tiles on the page when the reader mounts are the page script's; the rest are painted here. */
export function claimTiles(rt: Runtime): void {
  for (const canvas of rt.root.querySelectorAll<HTMLElement>(TILE)) canvas.dataset.latTile = 'page'
}

function renderTiles(rt: Runtime): void {
  const tiles = rt.root.querySelectorAll<HTMLCanvasElement>(TILE)
  if (tiles.length === 0) return
  if (!tileSeen) {
    tileSeen = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue
        tileSeen?.unobserve(entry.target)
        void paintTile(entry.target as HTMLCanvasElement)
      }
    })
    rt.signal.addEventListener('abort', () => {
      tileSeen?.disconnect()
      tileSeen = null
    })
  }
  for (const canvas of tiles) {
    canvas.dataset.latTile = ''
    tileSeen.observe(canvas)
  }
}

/** Figures in markup the reader or links just gained. */
export function renderFigures(rt: Runtime): void {
  renderTikz(rt)
  renderTiles(rt)
}
