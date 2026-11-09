import type { Runtime } from './runtime'
import { chunkUnderLine, keepLine } from './reader'

/**
 * The garden components a note passage can hold besides math and code: callouts fold, tikz
 * figures take the picture their note page rendered, latest-activities markers take the page's
 * card, signatures take the page's drawing, embeds take the page's transclusion, procedural drop
 * caps grow. Telescopic text needs nothing here: telescopic.inline.ts
 * listens on the document.
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

// ---- note pages -------------------------------------------------------------------------------

// Tikz pictures and activity cards are built into the note page; a page is fetched once, when the
// first passage component that needs it nears the screen, and shared by all of them.
const pages = new Map<string, Promise<Document | null>>()

function notePage(href: string): Promise<Document | null> {
  let found = pages.get(href)
  if (!found) {
    found = fetch(href)
      .then(response => (response.ok ? response.text() : null))
      .then(html => (html ? new DOMParser().parseFromString(html, 'text/html') : null))
      .catch(() => null)
    pages.set(href, found)
  }
  return found
}

/** Elements matching `selector` get `show` once each, when they come within 800px of the screen. */
function whenNear(
  rt: Runtime,
  selector: string,
  show: (rt: Runtime, element: HTMLElement) => Promise<void>,
): () => void {
  let seen: IntersectionObserver | null = null
  return () => {
    const elements = rt.root.querySelectorAll<HTMLElement>(selector)
    if (elements.length === 0) return
    if (!seen) {
      const observer = new IntersectionObserver(
        entries => {
          for (const entry of entries) {
            if (!entry.isIntersecting) continue
            observer.unobserve(entry.target)
            void show(rt, entry.target as HTMLElement)
          }
        },
        { rootMargin: '800px 0px' },
      )
      seen = observer
      rt.signal.addEventListener('abort', () => {
        observer.disconnect()
        seen = null
      })
    }
    for (const element of elements) seen.observe(element)
  }
}

// ---- tikz -------------------------------------------------------------------------------------

// A passage holds a tikz figure's source; the note page has the build's picture of it, keyed by
// that source (the figure's MathML annotation). A figure the page does not have shows its source.
const TIKZ = 'figure.lat-tikz[data-page]:not([data-state])'
const tikzPages = new Map<string, Promise<Map<string, HTMLImageElement>>>()
const squash = (source: string) => source.replace(/\s+/g, ' ').trim()

function pictures(href: string): Promise<Map<string, HTMLImageElement>> {
  let found = tikzPages.get(href)
  if (!found) {
    found = notePage(href).then(page => {
      const out = new Map<string, HTMLImageElement>()
      if (page)
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
    tikzPages.set(href, found)
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

// ---- latest activities ------------------------------------------------------------------------

// A `<div data-triathlon-latest>` marker in a note (dating, friends, craft) is filled at build
// from the Strava store (quartz/components/strava-activities.ts). The reader holds a link to the
// training log and swaps in the page's card: the one with as many rows as the marker asked for,
// else the page's first. Its links are relative to the note page, so they resolve against it.
const TRIATHLON = 'div.lat-triathlon[data-page]:not([data-state])'

function latestCard(page: Document, count: string | undefined): Element | null {
  const cards = [...page.querySelectorAll('.strava-activities')]
  const rows = (card: Element) => card.querySelectorAll('.strava-activities-list > li').length
  return cards.find(card => count !== undefined && rows(card) === Number(count)) ?? cards[0] ?? null
}

async function showTriathlon(rt: Runtime, holder: HTMLElement): Promise<void> {
  holder.dataset.state = 'loading'
  const href = holder.dataset.page!
  const page = await notePage(href)
  const card = page ? latestCard(page, holder.dataset.count) : null
  if (rt.signal.aborted || !holder.isConnected) return
  if (!card) {
    // no card on the page (no synced data, or the fetch failed): the link stays
    holder.dataset.state = 'link'
    return
  }
  const copy = document.importNode(card, true)
  const base = new URL(href, location.href)
  for (const link of copy.querySelectorAll<HTMLAnchorElement>('a[href]')) {
    const url = new URL(link.getAttribute('href')!, base)
    link.setAttribute(
      'href',
      url.origin === location.origin ? url.pathname + url.search + url.hash : url.href,
    )
  }
  keepLine(chunkUnderLine(rt), () => {
    holder.classList.add('strava-activities')
    holder.replaceChildren(...copy.childNodes)
    holder.dataset.state = 'ready'
  })
}

// ---- signatures -------------------------------------------------------------------------------

// `[^sign]` in a note is drawn at build from the Aarnphm transformer's letter banks, in the text
// the page's frontmatter names. The reader holds the signature's empty line and copies the page's
// drawing in; its strokes draw themselves again on insert (custom.scss drawPath).
const SIGNATURE = 'p.lat-signature[data-page]:not([data-state])'

async function showSignature(rt: Runtime, holder: HTMLElement): Promise<void> {
  holder.dataset.state = 'loading'
  const page = await notePage(holder.dataset.page!)
  const drawn = page?.querySelector('p.signature')
  if (rt.signal.aborted || !holder.isConnected) return
  if (!drawn) {
    // a page without one (or a failed fetch) leaves no line
    holder.dataset.state = 'none'
    return
  }
  const copy = document.importNode(drawn, true)
  keepLine(chunkUnderLine(rt), () => {
    holder.replaceChildren(...copy.childNodes)
    holder.dataset.state = 'ready'
  })
}

// ---- transclusions ----------------------------------------------------------------------------

// `![[watch]]` in a note shows the watch note on its page (`blockquote.transclude`, named by its
// origin slug and its source link's anchor). The reader holds a link to the target and swaps in
// the page's transclusion, its links and pictures resolved against the note page.
const EMBED = 'div.lat-embed[data-page][data-embed]:not([data-state])'

function transclusion(page: Document, href: string): Element | null {
  const url = new URL(href, location.origin)
  const origin = decodeURIComponent(url.pathname.replace(/^\/+/, ''))
  const blocks = [...page.querySelectorAll('blockquote.transclude')].filter(
    block => block.getAttribute('data-transclude-origin') === origin,
  )
  const anchor = (block: Element) => {
    const source = block.querySelector(':scope > a.transclude-src')?.getAttribute('href') ?? ''
    const hash = source.indexOf('#')
    return hash < 0 ? '' : source.slice(hash)
  }
  return blocks.find(block => anchor(block) === url.hash) ?? blocks[0] ?? null
}

/** Relative links and pictures in a copy of `href`'s markup, made absolute to this page. */
function rebase(copy: Element, href: string): void {
  const base = new URL(href, location.href)
  const local = (value: string) => {
    const url = new URL(value, base)
    return url.origin === location.origin ? url.pathname + url.search + url.hash : url.href
  }
  for (const node of copy.querySelectorAll<HTMLElement>('[href], [src], [srcset]')) {
    for (const name of ['href', 'src'] as const) {
      const value = node.getAttribute(name)
      if (value && !value.startsWith('#') && !/^(?:data|blob|javascript):/i.test(value))
        node.setAttribute(name, local(value))
    }
    const srcset = node.getAttribute('srcset')
    if (srcset)
      node.setAttribute(
        'srcset',
        srcset
          .split(',')
          .map(part => {
            const [url, ...rest] = part.trim().split(/\s+/)
            return [local(url), ...rest].join(' ')
          })
          .join(', '),
      )
  }
}

async function showEmbed(rt: Runtime, holder: HTMLElement): Promise<void> {
  holder.dataset.state = 'loading'
  const href = holder.dataset.page!
  const page = await notePage(href)
  const block = page ? transclusion(page, holder.dataset.embed!) : null
  if (rt.signal.aborted || !holder.isConnected) return
  if (!block) {
    // the page has no transclusion of it (or the fetch failed): the link stays
    holder.dataset.state = 'link'
    return
  }
  const copy = document.importNode(block, true)
  rebase(copy, href)
  keepLine(chunkUnderLine(rt), () => {
    holder.replaceChildren(copy)
    holder.dataset.state = 'ready'
  })
}

let renderTikz: (() => void) | null = null
let renderTriathlon: (() => void) | null = null
let renderSignature: (() => void) | null = null
let renderEmbed: (() => void) | null = null

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
  if (!renderTikz || !renderTriathlon || !renderSignature || !renderEmbed) {
    renderTikz = whenNear(rt, TIKZ, showTikz)
    renderTriathlon = whenNear(rt, TRIATHLON, showTriathlon)
    renderSignature = whenNear(rt, SIGNATURE, showSignature)
    renderEmbed = whenNear(rt, EMBED, showEmbed)
    rt.signal.addEventListener('abort', () => {
      renderTikz = null
      renderTriathlon = null
      renderSignature = null
      renderEmbed = null
    })
  }
  renderTikz()
  renderTriathlon()
  renderSignature()
  renderEmbed()
  renderTiles(rt)
}
