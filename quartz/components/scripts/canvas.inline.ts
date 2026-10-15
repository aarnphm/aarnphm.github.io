import type { ZoomBehavior } from 'd3-zoom'
import { select, zoom as d3Zoom, zoomIdentity, dragEnable, drag as d3Drag } from 'd3'
import { Marked } from 'marked'
import { canvasEdgeGeometry, canvasEdgeMidpoint, canvasEdgePath } from '../../util/canvas-edge'
import { fetchCanonical } from '../../util/fetch-canonical'
import { normalizeRelativeURLs } from '../../util/path'
import { mountHtmlCanvasPreview, type HtmlCanvasPreview } from './html-canvas-preview'
import { loadPageStyles } from './page-styles'
import { cleanupHydratedRoot } from './root-lifecycle'

// page chunks share one `marked` module, so canvas options stay off its global instance
const marked = new Marked({ breaks: true, gfm: true })

const EDGE_LABEL_LINE_HEIGHT = 1.2
const GROUP_LABEL_MIN_SCREEN_PX = 18

function renderTextNodeContent(node: NodeData): string {
  const html = marked.parse(node.resolvedText ?? node.text ?? '', { async: false })

  const doc = htmlParser.parseFromString(html, 'text/html')
  const anchors = Array.from(doc.body.querySelectorAll('a'))
  const remaining = [...(node.wikilinks ?? [])]
  const isLabel = doc.body.children.length === 1 && doc.body.firstElementChild?.tagName === 'P'

  anchors.forEach(anchor => {
    const href = anchor.getAttribute('href') ?? ''
    const idx = remaining.findIndex(link => link.resolvedHref === href)
    const match = idx >= 0 ? remaining.splice(idx, 1)[0] : undefined

    if (!match) {
      return
    }

    if (match.resolvedHref) {
      anchor.classList.add('internal')
      if (match.resolvedSlug) {
        const slug = match.resolvedSlug.startsWith('/')
          ? match.resolvedSlug
          : `/${match.resolvedSlug}`
        anchor.dataset.slug = slug
      }
    }

    if (match.missing) {
      anchor.classList.add('is-missing')
    }
  })

  normalizeRelativeURLs(doc, window.location.href)

  return `<div class="node-text${isLabel ? ' node-text-label' : ''}">${doc.body.innerHTML}</div>`
}

interface CanvasResolvedWikilink {
  raw: string
  target?: string
  anchor?: string
  alias?: string
  embed?: boolean
  resolvedSlug?: string
  resolvedHref?: string
  missing?: boolean
}

type CanvasMeta = {
  slug?: string
  href?: string
  displayName?: string
  description?: string
  content?: string
  resolvedText?: string
  wikilinks?: CanvasResolvedWikilink[]
}

interface CanvasNode {
  id: string
  type: 'text' | 'file' | 'link' | 'group'
  x: number
  y: number
  width: number
  height: number
  color?: string
  text?: string
  file?: string
  url?: string
  label?: string
  displayName?: string
  resolvedSlug?: string
  resolvedHref?: string
  description?: string
  content?: string
  resolvedText?: string
  wikilinks?: CanvasResolvedWikilink[]
}

interface CanvasEdge {
  id: string
  fromNode: string
  toNode: string
  fromSide?: string
  toSide?: string
  label?: string
  color?: string
}

interface CanvasData {
  nodes: CanvasNode[]
  edges: CanvasEdge[]
}

interface CanvasConfig {
  zoom: boolean
}

type CanvasBounds = { minX: number; minY: number; maxX: number; maxY: number }

type NodeData = CanvasNode

type LinkData = CanvasEdge & { source: NodeData; target: NodeData }

interface CanvasLifecycle {
  controller: AbortController
  cleanups: Set<() => void>
}

const jsonFetchCache = new Map<string, Promise<unknown>>()
const filePreviewCache = new Map<string, Promise<string | null>>()
const htmlParser = new DOMParser()
const canvasLifecycles = new WeakMap<HTMLElement, CanvasLifecycle>()
const activeCanvasContainers = new Set<HTMLElement>()

function createCanvasLifecycle(): CanvasLifecycle {
  return { controller: new AbortController(), cleanups: new Set() }
}

function addCanvasCleanup(lifecycle: CanvasLifecycle, cleanup: () => void): void {
  if (lifecycle.controller.signal.aborted) {
    cleanup()
    return
  }
  lifecycle.cleanups.add(cleanup)
}

function disposeCanvas(container: HTMLElement, lifecycle: CanvasLifecycle): void {
  if (canvasLifecycles.get(container) !== lifecycle) return

  lifecycle.controller.abort()
  lifecycle.cleanups.forEach(cleanup => cleanup())
  lifecycle.cleanups.clear()
  canvasLifecycles.delete(container)
  activeCanvasContainers.delete(container)
}

function isCanvasActive(container: HTMLElement, lifecycle: CanvasLifecycle): boolean {
  return (
    container.isConnected &&
    !lifecycle.controller.signal.aborted &&
    canvasLifecycles.get(container) === lifecycle
  )
}

function cleanupCanvases(root?: ParentNode): void {
  for (const container of activeCanvasContainers) {
    if (root instanceof Node && !root.contains(container)) continue
    const lifecycle = canvasLifecycles.get(container)
    if (lifecycle) disposeCanvas(container, lifecycle)
  }
}

window.quartzCanvas = { cleanup: cleanupCanvases }

function resolveToAbsoluteUrl(source: string): string {
  return new URL(source.trim(), window.location.href).toString()
}

async function fetchJson<T>(source: string): Promise<T> {
  const absoluteUrl = resolveToAbsoluteUrl(source)
  let request = jsonFetchCache.get(absoluteUrl)
  if (!request) {
    request = fetch(absoluteUrl, { credentials: 'same-origin' }).then(response => {
      if (!response.ok) {
        throw new Error(`Failed to load canvas data (${response.status})`)
      }
      return response.json()
    })
    jsonFetchCache.set(absoluteUrl, request)
  }

  try {
    return (await request) as T
  } catch (error) {
    if (jsonFetchCache.get(absoluteUrl) === request) jsonFetchCache.delete(absoluteUrl)
    throw error
  }
}

async function renderCanvas(container: HTMLElement, lifecycle: CanvasLifecycle) {
  const canvasUrl = container.getAttribute('data-canvas')?.trim()
  if (!canvasUrl) return

  const { signal } = lifecycle.controller

  const cfgAttr = container.getAttribute('data-cfg')
  const metaUrl = container.getAttribute('data-meta')?.trim()
  const isEmbed = container.getAttribute('data-embed') === 'true'

  try {
    const [canvasData, metaMap] = await Promise.all([
      fetchJson<CanvasData>(canvasUrl),
      metaUrl
        ? fetchJson<Record<string, CanvasMeta>>(metaUrl)
        : Promise.resolve<Record<string, CanvasMeta>>({}),
    ])
    if (!isCanvasActive(container, lifecycle)) return

    const cfg: CanvasConfig = cfgAttr ? JSON.parse(cfgAttr) : {}
    if (isEmbed) {
      cfg.zoom = false
    }

    if (!canvasData.nodes || canvasData.nodes.length === 0) {
      container.textContent = 'Empty canvas'
      return
    }

    container.replaceChildren()

    let width = container.clientWidth || 800
    let height = container.clientHeight || 600

    let toolbar: HTMLElement | null = null
    let helpModal: HTMLDialogElement | null = null

    if (!isEmbed) {
      toolbar = document.createElement('div')
      toolbar.className = 'canvas-controls'
      toolbar.innerHTML = `
        <div class="canvas-control-group">
          <button type="button" class="canvas-control-item" data-action="zoom-in" aria-label="Zoom in" title="Zoom in">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"></path><path d="M12 5v14"></path></svg>
          </button>
          <button type="button" class="canvas-control-item" data-action="zoom-reset" aria-label="Reset zoom to 100%" title="Reset zoom to 100%">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"></path><path d="M21 3v5h-5"></path></svg>
          </button>
          <button type="button" class="canvas-control-item" data-action="zoom-fit" aria-label="Zoom to fit" title="Zoom to fit">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3"></path><path d="M21 8V5a2 2 0 0 0-2-2h-3"></path><path d="M3 16v3a2 2 0 0 0 2 2h3"></path><path d="M16 21h3a2 2 0 0 0 2-2v-3"></path></svg>
          </button>
          <button type="button" class="canvas-control-item" data-action="zoom-out" aria-label="Zoom out" title="Zoom out">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"></path></svg>
          </button>
        </div>
        <div class="canvas-control-group">
          <button type="button" class="canvas-control-item" data-action="help" aria-label="Canvas help" title="Canvas help">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><path d="M12 17h.01"></path></svg>
          </button>
        </div>
      `
      container.appendChild(toolbar)

      helpModal = document.createElement('dialog')
      helpModal.className = 'canvas-help-modal'
      helpModal.setAttribute('aria-label', 'Canvas help')
      helpModal.innerHTML = `
        <div class="canvas-help-backdrop"></div>
        <div class="canvas-help-content">
          <div class="canvas-help-header">
            <h2>Canvas help</h2>
            <button type="button" class="canvas-help-close" aria-label="Close">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>
            </button>
          </div>
          <div class="canvas-help-body">
            <p>Drag cards to arrange this view. Reload restores the saved layout.</p>
            <div class="canvas-help-section">
              <h3>pan</h3>
              <div class="canvas-help-row">
                <span>Pan in either direction</span>
                <span class="canvas-help-keys"><kbd>Scroll</kbd> or drag background</span>
              </div>
              <div class="canvas-help-row">
                <span>Move a group</span>
                <span class="canvas-help-keys">Drag group title</span>
              </div>
              <div class="canvas-help-row">
                <span>Pan over a card</span>
                <span class="canvas-help-keys"><kbd>Space</kbd> + drag, or middle drag</span>
              </div>
              <div class="canvas-help-row">
                <span>Pan horizontally</span>
                <span class="canvas-help-keys"><kbd>Shift</kbd> <kbd>Scroll</kbd></span>
              </div>
            </div>
            <div class="canvas-help-section">
              <h3>zoom</h3>
              <div class="canvas-help-row">
                <span>Zoom</span>
                <span class="canvas-help-keys"><kbd>⌘/Ctrl</kbd> <kbd>Scroll</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Zoom to fit</span>
                <span class="canvas-help-keys"><kbd>Shift</kbd> <kbd>1</kbd> or <kbd>0</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Reset zoom to 100%</span>
                <span class="canvas-help-keys"><kbd>Shift</kbd> <kbd>0</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Zoom to group</span>
                <span class="canvas-help-keys"><kbd>Double-click</kbd> group</span>
              </div>
            </div>
            <div class="canvas-help-section">
              <h3>navigation</h3>
              <div class="canvas-help-row">
                <span>Read card</span>
                <span class="canvas-help-keys"><kbd>Click</kbd> or <kbd>Enter</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Open note</span>
                <span class="canvas-help-keys"><kbd>Click</kbd> card title</span>
              </div>
              <div class="canvas-help-row">
                <span>Open from content</span>
                <span class="canvas-help-keys"><kbd>⌘/Ctrl</kbd> <kbd>Click</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Open in side panel</span>
                <span class="canvas-help-keys"><kbd>Alt</kbd> <kbd>Click</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Scroll node content</span>
                <span class="canvas-help-keys">Focus node, then <kbd>Scroll</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Defocus node</span>
                <span class="canvas-help-keys"><kbd>Esc</kbd></span>
              </div>
            </div>
          </div>
        </div>
      `
      container.appendChild(helpModal)
    }

    const svg = select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .attr('tabindex', isEmbed ? null : 0)
      .attr('role', 'group')
      .attr(
        'aria-label',
        isEmbed
          ? 'Canvas preview'
          : 'Canvas viewer. Arrow keys pan, plus and minus zoom, 0 fits all cards.',
      )

    const defs = svg.append('defs')
    const pattern = defs
      .append('pattern')
      .attr('id', `dots-${Math.random().toString(36).substring(1, 10)}`)
      .attr('width', 20)
      .attr('height', 20)
      .attr('patternUnits', 'userSpaceOnUse')

    pattern
      .append('circle')
      .attr('cx', 1)
      .attr('cy', 1)
      .attr('r', 1)
      .attr('fill', 'var(--gray)')
      .attr('opacity', 0.3)

    svg
      .append('rect')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('fill', `url(#${pattern.attr('id')})`)
      .attr('class', 'canvas-background')

    const g = svg.append('g').attr('class', 'canvas-scene')
    const groupNodeGroup = g.append('g').attr('class', 'group-nodes')
    const edgeGroup = g.append('g').attr('class', 'edges')
    const edgeLabelGroup = g.append('g').attr('class', 'edge-labels')
    const nodeGroup = g.append('g').attr('class', 'nodes')

    addCanvasCleanup(lifecycle, () => {
      svg.interrupt()
      svg.on('.zoom', null)
      svg.on('click', null)
    })

    let focusedNode: SVGGElement | null = null
    let isHelpOpen = false
    let bounds: CanvasBounds | null = null
    let zoomBehavior: ZoomBehavior<SVGSVGElement, unknown> | null = null
    let currentScale = 1
    let currentTransform = zoomIdentity
    let renderFrame = 0
    let spacePressed = false
    let isFitted = true
    let mousePanning = false
    let nodeDragging = false
    let lastCameraInput = 0
    let previewTimer = 0
    let previewIdle = 0
    const previewJobs: { run: (active: boolean) => void; isActive: () => boolean }[] = []
    const movedNodes = new Set<NodeData>()
    let renderedTransform = ''
    let labelledScale = 0
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

    function queueRender() {
      if (renderFrame) return
      renderFrame = requestAnimationFrame(() => {
        renderFrame = 0
        if (!isCanvasActive(container, lifecycle)) return
        if (movedNodes.size > 0) {
          updatePositions(movedNodes)
          movedNodes.clear()
        }
        const transform = `translate(${currentTransform.x}px, ${currentTransform.y}px) scale(${currentScale})`
        if (renderedTransform !== transform) {
          g.style('transform', transform)
          // Grid spacing stays legible while its origin follows the world coordinates.
          const gridStep = 20 * Math.pow(2, Math.ceil(Math.log2(1 / currentScale)))
          const gridSize = gridStep * currentScale
          const offset = (value: number) => ((value % gridSize) + gridSize) % gridSize
          pattern
            .attr('width', gridSize)
            .attr('height', gridSize)
            .attr('x', offset(currentTransform.x))
            .attr('y', offset(currentTransform.y))
          if (labelledScale !== currentScale) {
            updateGroupLabels(currentScale)
            labelledScale = currentScale
          }
          const skeleton = currentScale < 0.5
          if (container.classList.contains('canvas-skeleton-view') !== skeleton) {
            container.classList.toggle('canvas-skeleton-view', skeleton)
            if (skeleton) fitCenterTitles()
            measureEdgeLabels()
          }
          renderedTransform = transform
        }
        updatePreviewVisibility()
      })
    }

    addCanvasCleanup(lifecycle, () => {
      cancelAnimationFrame(renderFrame)
      window.clearTimeout(previewTimer)
      window.cancelIdleCallback(previewIdle)
      for (const job of previewJobs.splice(0)) job.run(false)
      helpModal?.close()
      if (mousePanning || nodeDragging) {
        select(window).on('.zoom', null).on('.drag', null)
        dragEnable(window)
      }
    })

    if (cfg.zoom) {
      const zoom = d3Zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.02, 4])
        .clickDistance(4)
        .extent((): [[number, number], [number, number]] => [
          [0, 0],
          [width, height],
        ])
        .filter(event => {
          if (isHelpOpen || event.type === 'wheel' || event.type === 'dblclick') return false
          if (spacePressed || event.button === 1) return true
          const target = event.target
          if (event.type === 'touchstart') {
            if (event.touches.length > 1) return true
            return !(target instanceof Element && focusedNode?.contains(target))
          }
          return !event.button && !(target instanceof Element && target.closest('.node'))
        })
        .on('start', event => {
          mousePanning = event.sourceEvent?.type === 'mousedown'
        })
        .on('end', () => {
          mousePanning = false
        })
        .on('zoom', event => {
          currentTransform = event.transform
          currentScale = currentTransform.k
          lastCameraInput = performance.now()
          if (event.sourceEvent) isFitted = false
          queueRender()
        })
      zoomBehavior = zoom
      svg.call(zoom).on('dblclick.zoom', null)
    }

    const getBoundsTransform = (b: CanvasBounds) => {
      const contentW = Math.max(1, b.maxX - b.minX)
      const contentH = Math.max(1, b.maxY - b.minY)
      const padding = 64
      const kRaw = Math.min((width - padding) / contentW, (height - padding) / contentH)
      const k = Math.max(0.02, Math.min(4, kRaw))
      const cx = (b.minX + b.maxX) / 2
      const cy = (b.minY + b.maxY) / 2
      const tx = width / 2 - k * cx
      const ty = height / 2 - k * cy
      return { k, tx, ty }
    }

    const applyBoundsTransform = (b: CanvasBounds, animate: boolean) => {
      const { k, tx, ty } = getBoundsTransform(b)
      const transform = zoomIdentity.translate(tx, ty).scale(k)
      if (zoomBehavior) {
        const selection = animate ? svg.transition().duration(reducedMotion.matches ? 0 : 180) : svg
        selection.call(zoomBehavior.transform, transform)
      } else {
        currentTransform = transform
        currentScale = k
        queueRender()
      }
      return transform
    }

    if (!isEmbed) {
      container.addEventListener(
        'wheel',
        event => {
          if (isHelpOpen || !zoomBehavior) return
          const target = event.target
          if (
            !event.metaKey &&
            !event.ctrlKey &&
            focusedNode &&
            target instanceof Element &&
            focusedNode.contains(target) &&
            target.closest('.canvas-card-preview')
          ) {
            // Native HTML-in-Canvas can retarget wheel events to the bitmap surface.
            // Its HTML scroller still owns the card's reading position.
            if (target.closest('.canvas-html-surface') && !target.closest('.node-content')) {
              const content = focusedNode.querySelector<HTMLElement>(
                '.node-file-content, .node-text',
              )
              if (content) {
                const unit =
                  event.deltaMode === WheelEvent.DOM_DELTA_LINE
                    ? 16
                    : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
                      ? content.clientHeight
                      : 1
                event.preventDefault()
                event.stopPropagation()
                content.scrollBy(event.deltaX * unit, event.deltaY * unit)
              }
            }
            return
          }

          event.preventDefault()
          event.stopPropagation()
          svg.interrupt()
          isFitted = false
          const unit =
            event.deltaMode === WheelEvent.DOM_DELTA_LINE
              ? 16
              : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
                ? height
                : 1
          const dx = event.deltaX * unit
          const dy = event.deltaY * unit
          if (event.metaKey || event.ctrlKey) {
            const rect = container.getBoundingClientRect()
            const scale = currentScale * Math.pow(2, -dy * 0.002 * (event.ctrlKey ? 10 : 1))
            svg.call(zoomBehavior.scaleTo, scale, [
              event.clientX - rect.left - container.clientLeft,
              event.clientY - rect.top - container.clientTop,
            ])
          } else {
            const panX = event.shiftKey ? dx || dy : dx
            const panY = event.shiftKey ? 0 : dy
            // D3 translations use world units; wheel deltas use screen pixels.
            svg.call(zoomBehavior.translateBy, -panX / currentScale, -panY / currentScale)
          }
        },
        { passive: false, signal },
      )
    }

    const showHelp = () => {
      if (helpModal) {
        helpModal.classList.add('is-visible')
        helpModal.showModal()
        isHelpOpen = true
      }
    }

    const hideHelp = () => {
      if (helpModal) {
        helpModal.classList.remove('is-visible')
        helpModal.close()
        isHelpOpen = false
      }
    }

    if (helpModal) {
      helpModal.addEventListener(
        'cancel',
        event => {
          event.preventDefault()
          hideHelp()
        },
        { signal },
      )
      const helpBackdrop = helpModal.querySelector<HTMLElement>('.canvas-help-backdrop')
      const helpClose = helpModal.querySelector<HTMLButtonElement>('.canvas-help-close')

      if (helpBackdrop) {
        helpBackdrop.addEventListener('click', hideHelp, { signal })
      }

      if (helpClose) {
        helpClose.addEventListener('click', hideHelp, { signal })
      }
    }

    if (isEmbed) {
      const navigateTo = container.getAttribute('data-navigate-to')
      if (navigateTo) {
        const overlay = document.createElement('a')
        overlay.className = 'canvas-embed-overlay'
        overlay.style.cssText =
          'position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1000; cursor: pointer;'
        overlay.setAttribute('aria-label', `view full canvas: ${navigateTo}`)
        overlay.href = navigateTo.startsWith('/') ? navigateTo : `/${navigateTo}`
        container.appendChild(overlay)
      }
    }

    if (toolbar) {
      toolbar.querySelectorAll<HTMLButtonElement>('[data-action]').forEach(button => {
        button.addEventListener(
          'click',
          event => {
            event.stopPropagation()
            const action = button.dataset.action

            if (action === 'help') {
              showHelp()
              return
            }

            if (!zoomBehavior) return
            svg.interrupt()
            isFitted = action === 'zoom-fit'

            switch (action) {
              case 'zoom-in':
                svg
                  .transition()
                  .duration(reducedMotion.matches ? 0 : 180)
                  .call(zoomBehavior.scaleBy, 1.3)
                break
              case 'zoom-out':
                svg
                  .transition()
                  .duration(reducedMotion.matches ? 0 : 180)
                  .call(zoomBehavior.scaleBy, 0.7)
                break
              case 'zoom-reset':
                svg
                  .transition()
                  .duration(reducedMotion.matches ? 0 : 180)
                  .call(zoomBehavior.scaleTo, 1)
                break
              case 'zoom-fit':
                if (bounds) {
                  applyBoundsTransform(bounds, true)
                }
                break
            }
          },
          { signal },
        )
      })
    }

    const handleKeydown = (event: KeyboardEvent) => {
      const target = event.target
      if (
        target instanceof Element &&
        target.closest('input, textarea, select, [contenteditable="true"]')
      )
        return
      if (event.key === 'Escape') {
        hideHelp()
        if (focusedNode) {
          focusedNode.classList.remove('is-focused')
          focusedNode = null
        }
        return
      }
      if (isEmbed || isHelpOpen || !zoomBehavior) return
      if (target instanceof Element && !container.contains(target) && target !== document.body)
        return
      if (event.key === ' ') {
        if (target instanceof Element && target.closest('button, a')) return
        event.preventDefault()
        spacePressed = true
        return
      }
      if (event.key === '0' || (event.shiftKey && event.code === 'Digit1')) {
        event.preventDefault()
        svg.interrupt()
        isFitted = true
        if (bounds) applyBoundsTransform(bounds, true)
        return
      }
      // On AZERTY, Shift+0 types "0" and fits above; layouts with a distinct Shift+0 reset to 100%.
      if (event.shiftKey && event.code === 'Digit0' && !event.metaKey && !event.ctrlKey) {
        event.preventDefault()
        svg.interrupt()
        isFitted = false
        svg
          .transition()
          .duration(reducedMotion.matches ? 0 : 180)
          .call(zoomBehavior.scaleTo, 1)
        return
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (target instanceof Element && target.closest('.node-content')) return
      const step = event.shiftKey ? 160 : 40
      const pan: Record<string, [number, number]> = {
        ArrowLeft: [step, 0],
        ArrowRight: [-step, 0],
        ArrowUp: [0, step],
        ArrowDown: [0, -step],
      }
      const delta = pan[event.key]
      if (delta) {
        event.preventDefault()
        svg.interrupt()
        isFitted = false
        svg.call(zoomBehavior.translateBy, delta[0] / currentScale, delta[1] / currentScale)
      } else if (event.key === '+' || event.key === '=' || event.key === '-') {
        event.preventDefault()
        svg.interrupt()
        isFitted = false
        svg.call(zoomBehavior.scaleBy, event.key === '-' ? 1 / 1.3 : 1.3)
      }
    }

    document.addEventListener('keydown', handleKeydown, { signal })
    document.addEventListener(
      'keyup',
      event => {
        if (event.key === ' ') spacePressed = false
      },
      { signal },
    )
    window.addEventListener(
      'blur',
      () => {
        spacePressed = false
      },
      { signal },
    )

    const nodes: NodeData[] = canvasData.nodes.map(n => {
      const meta: CanvasMeta = metaMap[n.id] ?? {}
      return {
        ...n,
        displayName: meta.displayName ?? n.displayName,
        resolvedSlug: meta.slug ?? n.resolvedSlug,
        resolvedHref: meta.href ?? n.resolvedHref,
        description: meta.description ?? n.description,
        content: meta.content ?? n.content,
        resolvedText: meta.resolvedText ?? n.resolvedText,
        wikilinks: meta.wikilinks ?? n.wikilinks,
      }
    })

    const nodeMap = new Map<string, NodeData>()
    nodes.forEach(n => nodeMap.set(n.id, n))

    const groupNodes = nodes.filter(n => n.type === 'group')
    const regularNodes = nodes.filter(n => n.type !== 'group')

    const links: LinkData[] = canvasData.edges
      .map(e => {
        const source = nodeMap.get(e.fromNode)
        const target = nodeMap.get(e.toNode)
        if (!source || !target) return null
        return { ...e, source, target }
      })
      .filter((l): l is LinkData => l !== null)

    const groupNode = groupNodeGroup
      .selectAll('g.node')
      .data(groupNodes)
      .join('g')
      .attr('class', 'node node-group')
      .attr('data-node-id', d => d.id)
      .attr('data-color', d => d.color || '')

    groupNode
      .append('rect')
      .attr('class', 'node-bg')
      .attr('width', d => d.width)
      .attr('height', d => d.height)

    groupNode
      .append('rect')
      .attr('class', 'node-border-overlay')
      .attr('width', d => d.width)
      .attr('height', d => d.height)
      .attr('stroke', 'var(--gray)')
      .attr('stroke-width', 1.5)
      .attr('fill', 'none')
      .attr('pointer-events', 'none')

    // Group titles sit flush with the group's left border and keep a readable screen size
    // when zoomed out, so the overview stays navigable by section.
    const groupLabels: {
      element: SVGTextElement
      group: NodeData
      text: string
      metrics: SvgTextMetrics | null
      fontSize: number
    }[] = []

    groupNode
      .append('text')
      .attr('class', 'node-group-label')
      .attr('x', 0)
      .attr('y', -8)
      .attr('font-size', '2rem')
      .each(function (d: NodeData) {
        const text = d.label || 'Group'
        this.textContent = text
        groupLabels.push({ element: this, group: d, text, metrics: null, fontSize: 32 })
      })

    function measureGroupLabels() {
      for (const label of groupLabels) {
        label.metrics = measureSvgText(label.element, label.text)
        label.fontSize = parseFloat(getComputedStyle(label.element).fontSize) || 32
      }
    }

    function updateGroupLabels(scale: number) {
      for (const label of groupLabels) {
        if (!label.metrics) continue
        const labelScale = Math.max(1, GROUP_LABEL_MIN_SCREEN_PX / (label.fontSize * scale))
        label.element.setAttribute('transform', labelScale === 1 ? '' : `scale(${labelScale})`)
        const shown = fitSvgText(label.text, label.metrics, label.group.width / labelScale)
        if (label.element.textContent !== shown) label.element.textContent = shown
      }
    }

    const arrowId = `arrow-${pattern.attr('id')}`
    const edge = edgeGroup
      .selectAll<SVGGElement, LinkData>('g.edge')
      .data(links, d => d.id)
      .join(
        enter => {
          const edgeEnter = enter.append('g').attr('class', 'edge')
          edgeEnter
            .append('path')
            .attr('stroke', d => d.color || 'var(--gray)')
            .attr('stroke-width', 2)
            .attr('fill', 'none')
            .attr('marker-end', `url(#${arrowId})`)
          return edgeEnter
        },
        update => update,
        exit => exit.remove(),
      )

    edge
      .select('path')
      .attr('stroke', d => d.color || 'var(--gray)')
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', `url(#${arrowId})`)

    const labeledLinks = links.filter(
      (d): d is LinkData & { label: string } =>
        typeof d.label === 'string' && d.label.trim().length > 0,
    )

    const edgeLabels = edgeLabelGroup
      .selectAll<SVGGElement, LinkData & { label: string }>('g.edge-label-group')
      .data(labeledLinks, d => d.id)
      .join(
        enter => {
          const labelGroup = enter
            .append('g')
            .attr('class', 'edge-label-group')
            .style('pointer-events', 'none')

          labelGroup
            .append('rect')
            .attr('class', 'edge-label-bg')
            .attr('fill', 'var(--light)')
            .attr('stroke', 'var(--gray)')
            .attr('stroke-width', 1)
            .style('pointer-events', 'none')

          labelGroup
            .append('text')
            .attr('class', 'edge-label')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .style('pointer-events', 'none')

          return labelGroup
        },
        update => update,
        exit => exit.remove(),
      )

    edgeLabels.select('text').each(function (d) {
      const textSelection = select(this as SVGTextElement)
      const normalized = d.label.split(/\r?\n/)
      const totalLines = normalized.length
      const offset = ((totalLines - 1) * EDGE_LABEL_LINE_HEIGHT) / 2
      const offsetStr = `${-offset.toFixed(3)}em`
      const lineStepStr = `${EDGE_LABEL_LINE_HEIGHT.toFixed(3)}em`
      textSelection.text(null)
      textSelection
        .selectAll('tspan')
        .data(normalized)
        .join('tspan')
        .attr('x', 0)
        .attr('dy', (_, i) => {
          if (totalLines === 1) {
            return '0em'
          }
          if (i === 0) {
            return offsetStr
          }
          return lineStepStr
        })
        .text(line => (line.length === 0 ? '\u00A0' : line))
    })

    svg
      .append('defs')
      .append('marker')
      .attr('id', arrowId)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', 'var(--gray)')

    const node = nodeGroup
      .selectAll<SVGGElement, NodeData>('g.node')
      .data(regularNodes)
      .join('g')
      .attr('class', d => `node node-${d.type}`)
      .attr('data-node-id', d => d.id)
      .attr('data-color', d => d.color || '')
      .attr('tabindex', isEmbed ? null : 0)
      .attr('role', 'group')
      .attr('aria-label', d => d.displayName || d.file || d.url || 'Text card')

    node
      .append('rect')
      .attr('class', 'node-bg')
      .attr('width', d => d.width)
      .attr('height', d => d.height)
      .attr('stroke', 'var(--gray)')
      .attr('stroke-width', 1.5)

    const fileNodes = node.filter(d => d.type === 'file')

    fileNodes
      .append('foreignObject')
      .attr('class', 'canvas-card-title')
      .attr('x', 0)
      .attr('y', -28)
      .attr('width', d => d.width)
      .attr('height', 24)
      .append('xhtml:a')
      .attr('class', 'canvas-node-link internal')
      .attr('data-no-popover', 'true')
      .attr('href', d => resolveFileNodeUrl(d)?.toString() ?? null)
      .attr('aria-label', d => `Open ${d.displayName || d.file || 'note'}`)
      .text(d => d.displayName || d.file || '')

    fileNodes
      .append('text')
      .attr('class', 'node-title-text node-title-center')
      .attr('x', d => d.width / 2)
      .attr('y', d => d.height / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', '600')
      .text(d => d.displayName || d.file || '')

    // Centered titles are hidden until the skeleton view, so they can only be measured there.
    let centerTitlesFitted = false
    function fitCenterTitles() {
      if (centerTitlesFitted) return
      centerTitlesFitted = true
      fileNodes.select<SVGTextElement>('.node-title-center').each(function (d) {
        const text = d.displayName || d.file || ''
        this.textContent = fitSvgText(text, measureSvgText(this, text), d.width - 32)
      })
    }

    const textNodes = node.filter(d => d.type === 'text')

    textNodes
      .append('text')
      .attr('class', 'node-title-text node-title-center')
      .attr('x', d => d.width / 2)
      .attr('y', d => d.height / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', '600')

    textNodes.each(function (d: NodeData) {
      const host = select(this)
      const skeleton = host.append('g').attr('class', 'node-skeleton')
      const paddingX = 12
      const paddingY = 16
      const availableWidth = Math.max(12, d.width - paddingX * 2)
      const lineHeight = 8
      const lineGap = 6
      const maxLines = Math.min(4, Math.floor((d.height - paddingY * 2) / (lineHeight + lineGap)))
      const lineCount = Math.max(1, maxLines)

      skeleton.attr('transform', `translate(${paddingX}, ${paddingY})`)

      for (let i = 0; i < lineCount; i++) {
        const lineWidth = availableWidth * (1 - i * 0.12)
        skeleton
          .append('rect')
          .attr('class', 'node-skeleton-line')
          .attr('x', 0)
          .attr('y', i * (lineHeight + lineGap))
          .attr('width', Math.max(24, lineWidth))
          .attr('height', lineHeight)
      }
    })

    node
      .append('foreignObject')
      .attr('class', 'canvas-card-preview')
      .attr('width', d => d.width)
      .attr('height', d => d.height)
      .append('xhtml:div')
      .attr('class', 'node-content')
      .style('width', d => `${d.width}px`)
      .style('height', d => `${d.height}px`)
      .html(d => {
        if (d.type === 'text') {
          return renderTextNodeContent(d)
        } else if (d.type === 'file') {
          const hasDescription =
            typeof d.description === 'string' && d.description.trim().length > 0
          const fallbackSource = hasDescription ? d.description! : 'Loading preview...'
          const fallback = wrapTextHtml(fallbackSource)
          const placeholderAttr = hasDescription ? '' : ' data-placeholder="true"'
          return `<div class="node-file-content" data-loading="true"${placeholderAttr}>${fallback}</div>`
        } else if (d.type === 'link') {
          return `<div class="node-link"><span class="link-icon">🔗</span> ${wrapTextHtml(
            d.url || '',
          )}</div>`
        }
        return ''
      })

    textNodes.selectAll('.node-skeleton').raise()

    node
      .filter(d => d.type === 'file')
      .select('.node-content')
      .each(function (d: NodeData) {
        const container = this as HTMLElement
        const link = document.createElement('a')
        link.className = 'internal canvas-popover-link'
        const href = d.resolvedHref
          ? `/${d.resolvedHref}`
          : d.resolvedSlug
            ? `/${d.resolvedSlug}`
            : (d.file || '').replace(/\.md$/, '')
        link.href = href
        link.dataset.slug = d.resolvedSlug ? `/${d.resolvedSlug}` : href
        link.setAttribute('aria-hidden', 'true')
        link.tabIndex = -1
        link.dataset.noPopover = 'true'
        link.style.position = 'absolute'
        link.style.left = '0'
        link.style.top = '0'
        link.style.width = '100%'
        link.style.height = '100%'
        link.style.opacity = '0'
        link.style.pointerEvents = 'none'
        container.appendChild(link)
      })

    const previews: {
      node: NodeData
      element: SVGForeignObjectElement
      content: HTMLElement | null
      visible: boolean | null
      scrollTop: number
      generation: number
      renderer: HtmlCanvasPreview | null
    }[] = []

    function schedulePreviewWork() {
      if (previewTimer || previewIdle || previewJobs.length === 0 || signal.aborted) return
      previewTimer = window.setTimeout(
        () => {
          previewTimer = 0
          previewIdle = window.requestIdleCallback(() => {
            previewIdle = 0
            if (mousePanning || nodeDragging || performance.now() - lastCameraInput < 120) {
              schedulePreviewWork()
              return
            }
            let job = previewJobs.shift()
            while (job && !job.isActive()) {
              job.run(false)
              job = previewJobs.shift()
            }
            job?.run(true)
            schedulePreviewWork()
          })
        },
        Math.max(0, 120 - (performance.now() - lastCameraInput)),
      )
    }

    function waitForPreviewWork(isActive: () => boolean): Promise<boolean> {
      if (signal.aborted) return Promise.resolve(false)
      return new Promise(resolve => {
        previewJobs.push({ run: resolve, isActive })
        schedulePreviewWork()
      })
    }
    node.each(function (d) {
      const element = this.querySelector<SVGForeignObjectElement>('.canvas-card-preview')
      if (!element) return
      const content = element.querySelector<HTMLElement>('.node-content')
      const renderer = content ? mountHtmlCanvasPreview(content, d.width, d.height, signal) : null
      if (renderer) addCanvasCleanup(lifecycle, renderer.dispose)
      previews.push({
        node: d,
        element,
        content: this.querySelector<HTMLElement>('.node-file-content'),
        visible: null,
        scrollTop: 0,
        generation: 0,
        renderer,
      })
    })

    function updatePreviewVisibility() {
      const transform = currentTransform
      const margin = 160
      for (const preview of previews) {
        const d = preview.node
        const left = transform.x + d.x * transform.k
        const top = transform.y + d.y * transform.k
        const visible =
          transform.k >= 0.5 &&
          left < width + margin &&
          top < height + margin &&
          left + d.width * transform.k > -margin &&
          top + d.height * transform.k > -margin
        preview.renderer?.update(transform.k, visible)
        if (preview.visible === visible) continue
        const wasVisible = preview.visible
        preview.visible = visible
        const generation = ++preview.generation
        const content = preview.content
        if (!content) {
          preview.element.style.visibility = visible ? '' : 'hidden'
          continue
        }
        // SVG already clips offscreen cards. Inherited visibility would restyle every child
        // while panning; retire the HTML during idle work instead.
        if (visible) {
          if (content.children.length > 0 && content.dataset.loading === 'false') continue
          void hydrateFileNodeContent(
            content,
            d,
            () =>
              isCanvasActive(container, lifecycle) &&
              preview.visible === true &&
              preview.generation === generation,
            waitForPreviewWork,
          ).then(() => {
            if (preview.visible && preview.generation === generation)
              content.scrollTop = preview.scrollTop
          })
        } else if (wasVisible) {
          preview.scrollTop = content.scrollTop
          const isActive = () =>
            isCanvasActive(container, lifecycle) &&
            preview.visible === false &&
            preview.generation === generation
          void waitForPreviewWork(isActive).then(active => {
            if (!active || !isActive()) return
            cleanupHydratedRoot(content)
            content.textContent = d.description ?? 'Loading preview...'
            content.dataset.loading = 'true'
          })
        }
      }
    }

    addCanvasCleanup(lifecycle, () => {
      for (const preview of previews) {
        if (preview.content) cleanupHydratedRoot(preview.content)
      }
    })

    node
      .append('rect')
      .attr('class', 'node-border-overlay')
      .attr('width', d => d.width)
      .attr('height', d => d.height)
      .attr('stroke', 'var(--gray)')
      .attr('stroke-width', 1.5)
      .attr('fill', 'none')
      .attr('pointer-events', 'none')

    function focusCard(host: SVGGElement, d: NodeData, zoomToCard: boolean) {
      if (focusedNode && focusedNode !== host) focusedNode.classList.remove('is-focused')
      focusedNode = host
      host.classList.add('is-focused')
      if (zoomToCard && zoomBehavior) {
        svg.interrupt()
        isFitted = false
        const fit = getBoundsTransform({
          minX: d.x,
          minY: d.y,
          maxX: d.x + d.width,
          maxY: d.y + d.height,
        })
        const scale = Math.max(0.65, Math.min(1, fit.k))
        const transform = zoomIdentity
          .translate(
            width / 2 - scale * (d.x + d.width / 2),
            height / 2 - scale * (d.y + d.height / 2),
          )
          .scale(scale)
        svg
          .transition()
          .duration(reducedMotion.matches ? 0 : 180)
          .call(zoomBehavior.transform, transform)
      }
    }

    node.on('focusin', function (event: FocusEvent, d) {
      if (event.target !== this) {
        if (event.target instanceof Element && event.target.closest('.node-content'))
          focusCard(this, d, false)
        return
      }
      const left = currentTransform.x + d.x * currentScale
      const top = currentTransform.y + d.y * currentScale
      if (
        left + d.width * currentScale < 0 ||
        top + d.height * currentScale < 0 ||
        left > width ||
        top > height
      ) {
        focusCard(this, d, true)
      }
    })

    node.on(
      'pointerdown',
      function (event: PointerEvent, d) {
        if (
          event.button === 0 &&
          event.target instanceof Element &&
          event.target.closest('a, button, [role="button"], input, textarea, select, summary') &&
          event.target.closest('.node-content')
        )
          focusCard(this, d, false)
      },
      true,
    )

    node.on('keydown', function (event: KeyboardEvent, d) {
      if (event.target !== this) return
      if (event.altKey && event.key.startsWith('Arrow') && !isEmbed) {
        event.preventDefault()
        event.stopPropagation()
        const step = event.shiftKey ? 16 : 1
        const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0
        const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0
        moveCard(d, dx, dy, membersOf(d))
        refreshBounds()
      } else if (event.key === 'Enter') {
        event.preventDefault()
        focusCard(this, d, true)
      }
    })

    groupNode.on('dblclick', (event: MouseEvent, d) => {
      if (!zoomBehavior) return
      event.preventDefault()
      svg.interrupt()
      isFitted = false
      const label = groupLabels.find(label => label.group === d)
      applyBoundsTransform(
        {
          minX: d.x,
          minY: d.y - (label?.fontSize ?? 32),
          maxX: d.x + d.width,
          maxY: d.y + d.height,
        },
        true,
      )
    })

    node.on('dblclick', function (event: MouseEvent, d) {
      if (event.target instanceof Element && event.target.closest('a')) return
      event.preventDefault()
      focusCard(this, d, true)
    })

    node.on('click', (evt, d) => {
      const clickEvent = evt as MouseEvent
      const target = clickEvent.target
      if (!(target instanceof Element)) return
      const currentNode = evt.currentTarget as SVGGElement

      if (target.closest('.canvas-node-link')) {
        if (
          clickEvent.detail > 0 &&
          !clickEvent.metaKey &&
          !clickEvent.ctrlKey &&
          !clickEvent.altKey &&
          !clickEvent.shiftKey &&
          focusedNode !== currentNode
        ) {
          clickEvent.preventDefault()
          clickEvent.stopPropagation()
          focusCard(currentNode, d, currentScale < 0.5)
        }
        return
      }

      if (target.closest('a:not(.canvas-popover-link)')) return
      clickEvent.stopPropagation()

      if (d.type === 'file') {
        if (clickEvent.altKey) {
          const link = currentNode.querySelector(
            '.node-content a.canvas-popover-link',
          ) as HTMLAnchorElement | null
          if (link) {
            link.dispatchEvent(
              new MouseEvent('click', { altKey: true, bubbles: true, cancelable: true }),
            )
            return
          }
        }

        focusCard(currentNode, d, currentScale < 0.5)
        if (clickEvent.metaKey || clickEvent.ctrlKey) {
          const url = resolveFileNodeUrl(d)
          if (url) window.spaNavigate(url)
        }
      } else if (d.type === 'text') {
        focusCard(currentNode, d, currentScale < 0.5)
      } else if (d.type === 'link' && d.url) {
        window.open(d.url, '_blank', 'noopener,noreferrer')
      }
    })

    svg.on('click', evt => {
      const target = evt.target as SVGElement
      if (
        target.tagName === 'rect' ||
        target.tagName === 'svg' ||
        target.classList.contains('canvas-background')
      ) {
        if (focusedNode) {
          focusedNode.classList.remove('is-focused')
          focusedNode = null
        }
      }
    })

    addCanvasCleanup(lifecycle, () => {
      node.on('click', null)
      node.on('keydown', null)
      node.on('focusin', null)
      node.on('pointerdown', null)
      node.on('dblclick', null)
      groupNode.on('dblclick', null)
    })

    function membersOf(group: NodeData): NodeData[] {
      if (group.type !== 'group') return []
      return nodes.filter(
        d =>
          d !== group &&
          d.x >= group.x &&
          d.y >= group.y &&
          d.x + d.width <= group.x + group.width &&
          d.y + d.height <= group.y + group.height,
      )
    }

    function moveCard(d: NodeData, dx: number, dy: number, members: NodeData[]) {
      isFitted = false
      lastCameraInput = performance.now()
      for (const moved of [d, ...members]) {
        moved.x += dx
        moved.y += dy
        movedNodes.add(moved)
      }
      queueRender()
    }

    function refreshBounds() {
      bounds = {
        minX: Math.min(...nodes.map(d => d.x)),
        minY: Math.min(...nodes.map(d => d.y)),
        maxX: Math.max(...nodes.map(d => d.x + d.width)),
        maxY: Math.max(...nodes.map(d => d.y + d.height)),
      }
    }

    if (!isEmbed) {
      let members: NodeData[] = []
      const scene = g.node()
      if (!scene) return
      const draggable = g.selectAll<SVGGElement, NodeData>(
        ':scope > .nodes > .node, :scope > .group-nodes > .node',
      )
      const drag = d3Drag<SVGGElement, NodeData>()
        .container(() => scene)
        .touchable(() => false)
        .clickDistance(4)
        .filter(event => {
          if (isHelpOpen || spacePressed || event.button) return false
          const target = event.target
          if (!(target instanceof Element)) return false
          if (
            target.closest(
              'a, button, [role="button"], input, textarea, select, summary, [contenteditable="true"]',
            )
          )
            return false
          return !(focusedNode?.contains(target) && target.closest('.canvas-card-preview'))
        })
        .on('start', function (_event, d) {
          nodeDragging = true
          members = membersOf(d)
          this.classList.add('is-dragging')
        })
        .on('drag', (event, d) => moveCard(d, event.x - d.x, event.y - d.y, members))
        .on('end', function () {
          nodeDragging = false
          this.classList.remove('is-dragging')
          members = []
          refreshBounds()
        })
      draggable.call(drag)
      addCanvasCleanup(lifecycle, () => draggable.on('.drag', null))
    }

    function updatePositions(moved?: Set<NodeData>) {
      node.filter(d => !moved || moved.has(d)).attr('transform', d => `translate(${d.x},${d.y})`)

      groupNode
        .filter(d => !moved || moved.has(d))
        .attr('transform', d => `translate(${d.x},${d.y})`)

      edge
        .filter(d => !moved || moved.has(d.source) || moved.has(d.target))
        .select('path')
        .attr('d', d =>
          canvasEdgePath(canvasEdgeGeometry(d.source, d.target, d.fromSide, d.toSide)),
        )

      edgeLabels
        .filter(d => !moved || moved.has(d.source) || moved.has(d.target))
        .attr('transform', d => {
          const { x, y } = canvasEdgeMidpoint(
            canvasEdgeGeometry(d.source, d.target, d.fromSide, d.toSide),
          )
          return `translate(${x},${y})`
        })
    }

    updatePositions()

    function measureEdgeLabels() {
      edgeLabels.each(function () {
        const text = this.querySelector('text')
        const bg = this.querySelector('rect')
        if (text && bg) {
          const bbox = text.getBBox()
          const padding = 4
          bg.setAttribute('x', String(bbox.x - padding))
          bg.setAttribute('y', String(bbox.y - padding))
          bg.setAttribute('width', String(bbox.width + padding * 2))
          bg.setAttribute('height', String(bbox.height + padding * 2))
        }
      })
    }
    measureEdgeLabels()
    measureGroupLabels()
    void document.fonts.ready.then(() => {
      if (!isCanvasActive(container, lifecycle)) return
      measureEdgeLabels()
      measureGroupLabels()
      updateGroupLabels(currentScale)
      labelledScale = currentScale
      if (centerTitlesFitted) {
        centerTitlesFitted = false
        if (container.classList.contains('canvas-skeleton-view')) fitCenterTitles()
      }
    })

    const boundsAttr = container.getAttribute('data-canvas-bounds')
    if (boundsAttr) {
      bounds = JSON.parse(boundsAttr)
    } else if (isEmbed) {
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity

      for (const node of nodes) {
        const x1 = node.x
        const y1 = node.y
        const x2 = node.x + node.width
        const y2 = node.y + node.height

        minX = Math.min(minX, x1)
        minY = Math.min(minY, y1)
        maxX = Math.max(maxX, x2)
        maxY = Math.max(maxY, y2)
      }

      if (isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)) {
        bounds = { minX, minY, maxX, maxY }
      }
    }

    if (bounds) applyBoundsTransform(bounds, false)

    const resizeObserver = new ResizeObserver(() => {
      const nextWidth = container.clientWidth
      const nextHeight = container.clientHeight
      if (!nextWidth || !nextHeight || (nextWidth === width && nextHeight === height)) return
      const previousWidth = width
      const previousHeight = height
      width = nextWidth
      height = nextHeight
      svg.attr('width', width).attr('height', height).attr('viewBox', [0, 0, width, height])
      svg.interrupt()
      if (isFitted && bounds) applyBoundsTransform(bounds, false)
      else if (zoomBehavior) {
        svg.call(
          zoomBehavior.transform,
          zoomIdentity
            .translate(
              currentTransform.x + (width - previousWidth) / 2,
              currentTransform.y + (height - previousHeight) / 2,
            )
            .scale(currentScale),
        )
      } else queueRender()
    })
    resizeObserver.observe(container)
    addCanvasCleanup(lifecycle, () => resizeObserver.disconnect())
  } catch (error) {
    if (isCanvasActive(container, lifecycle)) {
      disposeCanvas(container, lifecycle)
      console.error('Failed to render canvas:', error)
      container.innerHTML =
        '<div class="canvas-error">Failed to load canvas data. Check the console for details.</div>'
    }
  }
}

async function hydrateFileNodeContent(
  container: HTMLElement,
  node: NodeData,
  isActive: () => boolean,
  waitForWork: (isActive: () => boolean) => Promise<boolean>,
): Promise<void> {
  container.dataset.loading = 'true'

  try {
    const preview = await getFileNodeMarkup(node)
    if (!preview) {
      if (container.dataset.placeholder === 'true' && isActive()) {
        container.textContent = 'Preview unavailable'
        container.removeAttribute('data-placeholder')
      }
      return
    }

    if (!(await waitForWork(isActive)) || !isActive()) return
    const doc = htmlParser.parseFromString(preview, 'text/html')
    const article = doc.querySelector('article')
    if (!article) return
    const noteUrl = resolveFileNodeUrl(node)
    normalizeRelativeURLs(article, noteUrl ?? window.location.href)
    if (noteUrl) {
      article.querySelectorAll('a[href^="#"]').forEach(anchor => {
        if (!(anchor instanceof HTMLAnchorElement)) return
        const href = anchor.getAttribute('href')
        if (href) anchor.href = new URL(href, noteUrl).toString()
      })
    }
    await loadPageStyles(article)
    if (!(await waitForWork(isActive)) || !isActive()) return
    cleanupHydratedRoot(container)
    container.replaceChildren(...article.childNodes)
    container.removeAttribute('data-placeholder')
    const event: CustomEventMap['contentdecrypted'] = new CustomEvent('contentdecrypted', {
      detail: { article: container, content: container, slug: node.resolvedSlug },
    })
    document.dispatchEvent(event)
  } finally {
    if (isActive()) {
      container.dataset.loading = 'false'
    }
  }
}

async function getFileNodeMarkup(node: NodeData): Promise<string | null> {
  const targetUrl = resolveFileNodeUrl(node)
  if (!targetUrl) return null
  targetUrl.hash = ''

  const cacheKey = targetUrl.toString()

  if (!filePreviewCache.has(cacheKey)) {
    filePreviewCache.set(
      cacheKey,
      (async () => {
        try {
          const response = await fetchCanonical(targetUrl)
          if (!response.ok) {
            if (response.status === 404) return null
            throw new Error(`Failed to load canvas preview (${response.status})`)
          }

          const contentType = response.headers.get('Content-Type') ?? ''
          if (!contentType.startsWith('text/html')) {
            return null
          }

          return await response.text()
        } catch (error) {
          filePreviewCache.delete(cacheKey)
          console.error('Canvas preview failed:', error)
          return null
        }
      })(),
    )
  }

  return filePreviewCache.get(cacheKey)!
}

function resolveFileNodeUrl(node: NodeData): URL | null {
  const base =
    (typeof node.resolvedHref === 'string' && node.resolvedHref.trim().length > 0
      ? node.resolvedHref
      : null) ??
    (typeof node.resolvedSlug === 'string' && node.resolvedSlug.trim().length > 0
      ? node.resolvedSlug
      : null) ??
    (typeof node.file === 'string' && node.file.trim().length > 0
      ? node.file.replace(/\.md$/, '')
      : null)

  if (!base) return null

  const trimmed = base.trim()
  if (trimmed.length === 0) return null

  const normalizedPath = trimmed.startsWith('/') ? trimmed : `/${trimmed}`
  return new URL(normalizedPath, window.location.origin)
}

type SvgTextMetrics = { prefixWidths: number[]; ellipsisWidth: number }

// Prefix widths are measured once, so refitting at another width needs no text layout.
function measureSvgText(element: SVGTextElement, text: string): SvgTextMetrics {
  element.textContent = '…'
  const ellipsisWidth = element.getComputedTextLength()
  element.textContent = text
  const prefixWidths = Array.from({ length: text.length + 1 }, (_, i) =>
    i === 0 ? 0 : element.getSubStringLength(0, i),
  )
  return { prefixWidths, ellipsisWidth }
}

function fitSvgText(text: string, metrics: SvgTextMetrics, maxWidth: number): string {
  const { prefixWidths, ellipsisWidth } = metrics
  if (prefixWidths[text.length] <= maxWidth) return text
  let end = text.length - 1
  while (end > 0 && prefixWidths[end] + ellipsisWidth > maxWidth) end--
  // keep surrogate pairs whole
  if (end > 0 && /[\uDC00-\uDFFF]/.test(text[end])) end--
  return `${text.slice(0, end).trimEnd()}…`
}

function wrapTextHtml(text: string): string {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

function initializeCanvases(root: ParentNode) {
  root
    .querySelectorAll<HTMLElement>(['.canvas-container', '.canvas-embed-container'].join(','))
    .forEach(container => {
      if (canvasLifecycles.has(container)) return

      const lifecycle = createCanvasLifecycle()
      canvasLifecycles.set(container, lifecycle)
      activeCanvasContainers.add(container)
      void renderCanvas(container, lifecycle)
    })
}

document.addEventListener('nav', () => {
  window.addCleanup(cleanupCanvases)
  activeCanvasContainers.forEach(container => {
    if (container.isConnected) return
    const lifecycle = canvasLifecycles.get(container)
    if (lifecycle) disposeCanvas(container, lifecycle)
  })
  initializeCanvases(document)
  const onContentMounted = (event: CustomEventMap['contentdecrypted']) => {
    initializeCanvases(event.detail.content)
  }
  document.addEventListener('contentdecrypted', onContentMounted)
  window.addCleanup(() => document.removeEventListener('contentdecrypted', onContentMounted))
})
