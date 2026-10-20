import { rootNavSignal } from './root-lifecycle'

const PELE_CLIENT_SRC = '/static/scripts/pele.js'

type PeleClientModule = typeof import('./pele.client')
let peleClientLoad: Promise<PeleClientModule> | undefined

function loadPeleClient(): Promise<PeleClientModule> {
  if (!peleClientLoad) {
    const src = new URL(PELE_CLIENT_SRC, document.baseURI).href
    peleClientLoad = (import(src) as Promise<PeleClientModule>).catch(error => {
      peleClientLoad = undefined
      throw error
    })
  }
  return peleClientLoad
}

function diagramFigures(root: Document | HTMLElement): HTMLElement[] {
  if (root instanceof HTMLElement && root.matches('figure.pele-figure')) return [root]
  if (root === document && document.documentElement.dataset.slides === 'true') {
    return Array.from(document.querySelectorAll('.slide.active figure.pele-figure'))
  }
  return Array.from(root.querySelectorAll('figure.pele-figure'))
}

document.addEventListener('nav', () => {
  const diagrams = new WeakSet<HTMLElement>()
  const navSignal = rootNavSignal(document)

  async function setupDiagram(figure: HTMLElement) {
    if (diagrams.has(figure)) return
    const client = await loadPeleClient()
    if (navSignal.aborted || !figure.isConnected) return
    await client.setupPeleDiagram(figure, diagrams)
  }

  function setup(root: Document | HTMLElement = document) {
    if (
      root === document &&
      document.getElementById('stacked-notes-container')?.classList.contains('active')
    )
      return
    for (const figure of diagramFigures(root)) {
      // Canvas cards announce their own content and wait until the diagram is visible.
      if (root === document && figure.closest('.node-file-content') && !diagrams.has(figure))
        continue
      void setupDiagram(figure)
    }
  }

  document.addEventListener(
    'contentdecrypted',
    event => {
      const root = event.detail.content
      const figures = diagramFigures(root)
      if (figures.length === 0) return
      const card = root.closest<HTMLElement>('.node-file-content')
      if (!card) {
        setup(root)
        return
      }
      const signal = rootNavSignal(root)
      const observer = new IntersectionObserver(
        entries => {
          for (const entry of entries) {
            if (!entry.isIntersecting || !(entry.target instanceof HTMLElement)) continue
            observer.unobserve(entry.target)
            if (!signal.aborted && entry.target.isConnected) void setupDiagram(entry.target)
          }
        },
        { root: card, rootMargin: '160px' },
      )
      for (const figure of figures) observer.observe(figure)
      signal.addEventListener('abort', () => observer.disconnect(), { once: true })
    },
    { signal: navSignal },
  )
  document.addEventListener('slidechange', () => setup(), { signal: navSignal })
  setup()
})
