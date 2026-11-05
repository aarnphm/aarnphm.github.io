import { getFullSlug, simplifySlug } from '../../util/path'
import { renderGraph } from './graph/render-graph'
import { addToVisited } from './graph/visited'

let activeGraphCleanup: (() => void) | null = null
let activeGraphCleanupRegistered = false

function cleanupActiveGraphView() {
  activeGraphCleanup?.()
  activeGraphCleanup = null
}

function registerActiveGraphCleanup() {
  if (activeGraphCleanupRegistered) return
  activeGraphCleanupRegistered = true
  window.addCleanup(() => {
    activeGraphCleanupRegistered = false
    cleanupActiveGraphView()
  })
}

document.addEventListener('nav', async (e: CustomEventMap['nav']) => {
  const notes = document.getElementById('stacked-notes-container')
  if (notes?.classList.contains('active')) return

  const slug = e.detail.url

  cleanupActiveGraphView()
  registerActiveGraphCleanup()

  addToVisited(simplifySlug(slug))

  const localGraphContainers = Array.from(
    document.getElementsByClassName('graph-container'),
  ).filter((container): container is HTMLElement => container instanceof HTMLElement)
  let localGraphCleanups: (() => void)[] = []
  let localGraphRenderVersion = 0
  let localGraphAnimationFrame: number | null = null
  const localGraphResizeObserver = new ResizeObserver(scheduleLocalGraphRender)

  function cleanupLocalGraphs() {
    for (const cleanup of localGraphCleanups) {
      cleanup()
    }
    localGraphCleanups = []
  }

  async function renderLocalGraphs() {
    const renderVersion = ++localGraphRenderVersion
    cleanupLocalGraphs()

    for (const graphContainer of localGraphContainers) {
      if (graphContainer.offsetWidth === 0 || graphContainer.offsetHeight === 0) continue
      const cleanup = await renderGraph(graphContainer, slug)
      if (renderVersion === localGraphRenderVersion) {
        localGraphCleanups.push(cleanup)
      } else {
        cleanup()
      }
    }
  }

  function scheduleLocalGraphRender() {
    if (localGraphAnimationFrame !== null) return
    localGraphAnimationFrame = requestAnimationFrame(() => {
      localGraphAnimationFrame = null
      void renderLocalGraphs()
    })
  }

  for (const graphContainer of localGraphContainers) {
    localGraphResizeObserver.observe(graphContainer)
  }
  scheduleLocalGraphRender()

  const containers = Array.from(document.getElementsByClassName('global-graph-outer')).filter(
    (container): container is HTMLElement => container instanceof HTMLElement,
  )
  let globalGraphCleanups: (() => void)[] = []
  let globalGraphRenderVersion = 0

  function cleanupGlobalGraphs() {
    for (const cleanup of globalGraphCleanups) {
      cleanup()
    }
    globalGraphCleanups = []
  }

  async function renderGlobalGraph() {
    const renderVersion = ++globalGraphRenderVersion
    cleanupGlobalGraphs()
    const currentSlug = getFullSlug(window)
    for (const container of containers) {
      container.classList.add('active')
      const graphContainer = container.querySelector<HTMLElement>('.global-graph-container')
      if (!graphContainer) continue
      const cleanup = await renderGraph(graphContainer, currentSlug)
      if (renderVersion === globalGraphRenderVersion) {
        globalGraphCleanups.push(cleanup)
      } else {
        cleanup()
      }
    }
  }

  function hideGlobalGraph() {
    globalGraphRenderVersion++
    cleanupGlobalGraphs()
    for (const container of containers) {
      container.classList.remove('active')
    }
  }

  function globalGraphBackdropHandler(this: HTMLElement, e: HTMLElementEventMap['click']) {
    if (e.target !== this) return
    e.preventDefault()
    e.stopPropagation()
    hideGlobalGraph()
  }

  function shortcutHandler(e: HTMLElementEventMap['keydown']) {
    if (
      e.key.startsWith('Esc') &&
      containers.some(container => container.classList.contains('active'))
    ) {
      e.preventDefault()
      hideGlobalGraph()
      return
    }

    if ((e.key === ';' || e.key === 'g') && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      const anyGlobalGraphOpen = containers.some(container =>
        container.classList.contains('active'),
      )
      if (anyGlobalGraphOpen) {
        hideGlobalGraph()
      } else {
        void renderGlobalGraph()
      }
    }
  }

  const containerIcons = document.getElementsByClassName('global-graph-icon')
  for (const icon of Array.from(containerIcons)) {
    icon.addEventListener('click', renderGlobalGraph)
  }

  for (const container of containers) {
    container.addEventListener('click', globalGraphBackdropHandler)
  }

  document.addEventListener('keydown', shortcutHandler)

  activeGraphCleanup = () => {
    document.removeEventListener('keydown', shortcutHandler)
    for (const icon of Array.from(containerIcons)) {
      icon.removeEventListener('click', renderGlobalGraph)
    }
    for (const container of containers) {
      container.removeEventListener('click', globalGraphBackdropHandler)
      container.classList.remove('active')
    }
    if (localGraphAnimationFrame !== null) {
      cancelAnimationFrame(localGraphAnimationFrame)
      localGraphAnimationFrame = null
    }
    localGraphResizeObserver.disconnect()
    localGraphRenderVersion++
    cleanupLocalGraphs()
    globalGraphRenderVersion++
    cleanupGlobalGraphs()
  }
})
