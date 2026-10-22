import { zoom as d3Zoom, dragEnable, select, zoomIdentity } from 'd3'
import type { CanvasBounds, NodeData } from './data'
import type { CanvasScene } from './scene'
import { addCanvasCleanup, isCanvasActive } from './lifecycle'

export function installCamera(s: CanvasScene) {
  s.queueRender = function queueRender() {
    if (s.renderFrame) return
    s.renderFrame = requestAnimationFrame(() => {
      s.renderFrame = 0
      if (!isCanvasActive(s.container, s.lifecycle)) return
      if (s.movedNodes.size > 0) {
        s.updatePositions(s.movedNodes)
        s.movedNodes.clear()
      }
      const transform = `translate(${s.currentTransform.x}px, ${s.currentTransform.y}px) scale(${s.currentScale})`
      if (s.renderedTransform !== transform) {
        s.g.style('transform', transform)
        // Grid spacing stays legible while its origin follows the world coordinates.
        const gridStep = 20 * Math.pow(2, Math.ceil(Math.log2(1 / s.currentScale)))
        const gridSize = gridStep * s.currentScale
        const offset = (value: number) => ((value % gridSize) + gridSize) % gridSize
        s.pattern
          .attr('width', gridSize)
          .attr('height', gridSize)
          .attr('x', offset(s.currentTransform.x))
          .attr('y', offset(s.currentTransform.y))
        const skeleton = s.currentScale < 0.5
        if (s.container.classList.contains('canvas-skeleton-view') !== skeleton) {
          s.container.classList.toggle('canvas-skeleton-view', skeleton)
          s.measureEdgeLabels()
        }
        if (skeleton) s.fitCenterTitles()
        s.measureGroupLabels()
        s.renderedTransform = transform
      }
      s.updatePreviewVisibility()
    })
  }
}

export function mountCamera(s: CanvasScene) {
  s.svg = select(s.container)
    .append('svg')
    .attr('width', s.width)
    .attr('height', s.height)
    .attr('viewBox', [0, 0, s.width, s.height])
    .attr('tabindex', s.isEmbed || s.isLive ? null : 0)
    .attr('role', 'group')
    .attr(
      'aria-label',
      s.isEmbed
        ? 'Canvas preview'
        : 'Canvas viewer. Arrow keys pan, plus and minus zoom, 0 fits all cards.',
    )

  s.defs = s.svg.append('defs')

  s.pattern = s.defs
    .append('pattern')
    .attr('id', `dots-${Math.random().toString(36).substring(1, 10)}`)
    .attr('width', 20)
    .attr('height', 20)
    .attr('patternUnits', 'userSpaceOnUse')

  s.pattern
    .append('circle')
    .attr('cx', 1)
    .attr('cy', 1)
    .attr('r', 1)
    .attr('fill', 'var(--gray)')
    .attr('opacity', 0.3)

  s.svg
    .append('rect')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('fill', `url(#${s.pattern.attr('id')})`)
    .attr('class', 'canvas-background')

  s.g = s.svg.append('g').attr('class', 'canvas-scene')

  s.groupNodeGroup = s.g.append('g').attr('class', 'group-nodes')

  s.edgeGroup = s.g.append('g').attr('class', 'edges')

  s.edgeLabelGroup = s.g.append('g').attr('class', 'edge-labels')

  s.nodeGroup = s.g.append('g').attr('class', 'nodes')

  addCanvasCleanup(s.lifecycle, () => {
    s.svg.interrupt()
    s.svg.on('.zoom', null)
    s.svg.on('click', null)
  })

  s.focusedNode = null

  s.isHelpOpen = false

  s.bounds = null

  s.zoomBehavior = null

  s.currentScale = 1

  s.currentTransform = zoomIdentity

  s.renderFrame = 0

  s.spacePressed = false

  s.isFitted = true

  s.mousePanning = false

  s.nodeDragging = false

  s.lastCameraInput = 0

  s.previewTimer = 0

  s.previewIdle = 0

  s.previewJobs = []

  s.movedNodes = new Set<NodeData>()

  s.renderedTransform = ''

  s.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

  addCanvasCleanup(s.lifecycle, () => {
    cancelAnimationFrame(s.renderFrame)
    window.clearTimeout(s.previewTimer)
    window.cancelIdleCallback(s.previewIdle)
    for (const job of s.previewJobs.splice(0)) job.run(false)
    s.helpModal?.close()
    if (s.mousePanning || s.nodeDragging) {
      select(window).on('.zoom', null).on('.drag', null)
      dragEnable(window)
    }
  })

  if (s.cfg.zoom) {
    const zoom = d3Zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.02, 4])
      .clickDistance(4)
      .extent((): [[number, number], [number, number]] => [
        [0, 0],
        [s.width, s.height],
      ])
      .filter(event => {
        if (s.isHelpOpen || event.type === 'wheel' || event.type === 'dblclick') return false
        if (s.spacePressed || event.button === 1) return true
        const target = event.target
        if (event.type === 'touchstart') {
          if (event.touches.length > 1) return true
          return !(target instanceof Element && s.focusedNode?.contains(target))
        }
        return !event.button && !(target instanceof Element && target.closest('.node'))
      })
      .on('start', event => {
        s.mousePanning = event.sourceEvent?.type === 'mousedown'
      })
      .on('end', () => {
        s.mousePanning = false
      })
      .on('zoom', event => {
        s.currentTransform = event.transform
        s.currentScale = s.currentTransform.k
        s.lastCameraInput = performance.now()
        if (event.sourceEvent) s.isFitted = false
        s.queueRender()
      })
    s.zoomBehavior = zoom
    s.svg.call(zoom).on('dblclick.zoom', null)
    // one finger scrolls the page past a live canvas; two fingers still pinch it
    if (s.isLive) s.svg.style('touch-action', 'pan-y')
  }

  s.getBoundsTransform = (b: CanvasBounds) => {
    const contentW = Math.max(1, b.maxX - b.minX)
    const contentH = Math.max(1, b.maxY - b.minY)
    const padding = 64
    const kRaw = Math.min((s.width - padding) / contentW, (s.height - padding) / contentH)
    const k = Math.max(0.02, Math.min(4, kRaw))
    const cx = (b.minX + b.maxX) / 2
    const cy = (b.minY + b.maxY) / 2
    const tx = s.width / 2 - k * cx
    const ty = s.height / 2 - k * cy
    return { k, tx, ty }
  }

  s.applyBoundsTransform = (b: CanvasBounds, animate: boolean) => {
    const { k, tx, ty } = s.getBoundsTransform(b)
    const transform = zoomIdentity.translate(tx, ty).scale(k)
    if (s.zoomBehavior) {
      const selection = animate
        ? s.svg.transition().duration(s.reducedMotion.matches ? 0 : 180)
        : s.svg
      selection.call(s.zoomBehavior.transform, transform)
    } else {
      s.currentTransform = transform
      s.currentScale = k
      s.queueRender()
    }
    return transform
  }
}
