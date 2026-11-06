// The latent instrument is server-rendered; this loads the runtime only on a latent page, as a
// lazy script under static/scripts/, so other pages never pay for it. A latent embed in a note
// (figure.latent-embed) loads its own smaller runtime, latent-embed.js, the same way. Transformer resources are
// bundled as iife, where import.meta.url is empty, and LazyScripts leaves the name unhashed, so
// the path is spelled out.
type LatentModule = typeof import('./latent.client')
let modulePromise: Promise<LatentModule> | undefined
let failedLoads = 0

function loadLatent(): Promise<LatentModule> {
  if (!modulePromise) {
    const source = new URL('/static/scripts/latent.js', document.baseURI)
    if (failedLoads) source.searchParams.set('retry', String(failedLoads))
    modulePromise = import(source.href).catch(cause => {
      modulePromise = undefined
      failedLoads++
      throw cause
    })
  }
  return modulePromise
}

/** Put the server-rendered focus chunk under the reading line before the runtime arrives. */
function alignFocus(root: HTMLElement): void {
  if (new URL(window.location.href).searchParams.has('focus')) return
  const chunk = root.querySelector<HTMLElement>('.lat-chunk[data-focus]')
  if (!chunk) return
  const top = chunk.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.38 + 8
  if (top > 0) window.scrollTo({ top, behavior: 'auto' })
}

document.addEventListener('nav', () => {
  if (document.body.dataset.layout !== 'latent') return
  const root = document.querySelector<HTMLElement>('.latent[data-latent-base]')
  if (!root) return
  alignFocus(root)
  void loadLatent().then(runtime => {
    if (!root.isConnected) return
    runtime.mountLatent(root)
  })
})

// ---- latent embeds in notes (docs/latent-embed.md §5) ---------------------------------------
// tier 1: a figure within 600px of the viewport preloads the runtime and its coordinates when
// the page is idle; tier 2: intent (a 120 ms hover, focus, a touch) mounts the canvas and the
// live column. A figure in a popover stays the build's poster.
type EmbedModule = typeof import('./latent-embed.client')
let embedPromise: Promise<EmbedModule> | undefined
let embedFailures = 0
const EMBED_SCRIPT = '/static/scripts/latent-embed.js'

function loadEmbed(): Promise<EmbedModule> {
  if (!embedPromise) {
    const source = new URL(EMBED_SCRIPT, document.baseURI)
    if (embedFailures) source.searchParams.set('retry', String(embedFailures))
    embedPromise = import(source.href).catch(cause => {
      embedPromise = undefined
      embedFailures++
      throw cause
    })
  }
  return embedPromise
}

function hint(rel: string, href: string, as?: string): void {
  const url = new URL(href, document.baseURI).href
  for (const link of document.head.querySelectorAll<HTMLLinkElement>(`link[rel="${rel}"]`))
    if (link.href === url) return
  const link = document.createElement('link')
  link.rel = rel
  link.href = url
  if (as) link.as = as
  document.head.append(link)
}

function mountEmbed(figure: HTMLElement): void {
  void loadEmbed()
    .then(runtime => {
      if (figure.isConnected) runtime.mountLatentEmbed(figure)
    })
    .catch(error => console.warn('[latent] embed runtime', error))
}

const embedOf = (target: EventTarget | null): HTMLElement | null => {
  const figure =
    target instanceof Element
      ? target.closest<HTMLElement>('figure.latent-embed[data-latent-coords]')
      : null
  return figure && figure.dataset.latentReady === undefined && !figure.closest('.popover')
    ? figure
    : null
}

// Intent is delegated from the document, once: a figure the page adds after `nav` (the stream's
// feed appends entries as it scrolls) mounts the same way as one the page shipped with.
let dwellOn: HTMLElement | null = null
let dwell = 0
document.addEventListener(
  'pointerover',
  event => {
    const figure = embedOf(event.target)
    if (!figure || figure === dwellOn) return
    window.clearTimeout(dwell)
    dwellOn = figure
    dwell = window.setTimeout(() => {
      dwellOn = null
      mountEmbed(figure)
    }, 120)
  },
  { passive: true },
)
document.addEventListener(
  'pointerout',
  event => {
    if (!dwellOn || dwellOn.contains(event.relatedTarget as Node | null)) return
    window.clearTimeout(dwell)
    dwellOn = null
  },
  { passive: true },
)
for (const type of ['focusin', 'touchstart'] as const)
  document.addEventListener(
    type,
    event => {
      const figure = embedOf(event.target)
      if (figure) mountEmbed(figure)
    },
    { passive: true },
  )

// tier 1 for the figures a page ships with: preload the runtime and the coordinates when idle
document.addEventListener('nav', () => {
  const figures = [
    ...document.querySelectorAll<HTMLElement>('figure.latent-embed[data-latent-coords]'),
  ].filter(figure => !figure.closest('.popover'))
  if (figures.length === 0) return
  const controller = new AbortController()
  const signal = controller.signal
  window.addCleanup?.(() => controller.abort())
  const near = new IntersectionObserver(
    entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue
        near.unobserve(entry.target)
        const coords = (entry.target as HTMLElement).dataset.latentCoords!
        const idle = () => {
          if (signal.aborted) return
          hint('modulepreload', EMBED_SCRIPT)
          hint('prefetch', coords, 'fetch')
        }
        if ('requestIdleCallback' in window) requestIdleCallback(idle, { timeout: 2000 })
        else setTimeout(idle, 300)
      }
    },
    { rootMargin: '600px 0px' },
  )
  signal.addEventListener('abort', () => near.disconnect(), { once: true })
  for (const figure of figures) near.observe(figure)
})
