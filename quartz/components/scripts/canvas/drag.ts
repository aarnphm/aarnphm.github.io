import { drag as d3Drag, zoomIdentity } from 'd3'
import type { NodeData } from './data'
import type { CanvasScene } from './scene'
import { canvasEdgeGeometry, canvasEdgeMidpoint, canvasEdgePath } from '../../../util/canvas-edge'
import { addCanvasCleanup, isCanvasActive } from './lifecycle'

export function installDrag(s: CanvasScene) {
  s.membersOf = function membersOf(group: NodeData): NodeData[] {
    if (group.type !== 'group') return []
    return s.nodes.filter(
      d =>
        d !== group &&
        d.x >= group.x &&
        d.y >= group.y &&
        d.x + d.width <= group.x + group.width &&
        d.y + d.height <= group.y + group.height,
    )
  }

  s.moveCard = function moveCard(d: NodeData, dx: number, dy: number, members: NodeData[]) {
    s.isFitted = false
    s.lastCameraInput = performance.now()
    for (const moved of [d, ...members]) {
      moved.x += dx
      moved.y += dy
      s.movedNodes.add(moved)
    }
    s.queueRender()
  }

  s.refreshBounds = function refreshBounds() {
    s.bounds = {
      minX: Math.min(...s.nodes.map(d => d.x)),
      minY: Math.min(...s.nodes.map(d => d.y)),
      maxX: Math.max(...s.nodes.map(d => d.x + d.width)),
      maxY: Math.max(...s.nodes.map(d => d.y + d.height)),
    }
  }

  s.updatePositions = function updatePositions(moved?: Set<NodeData>) {
    s.node.filter(d => !moved || moved.has(d)).attr('transform', d => `translate(${d.x},${d.y})`)

    s.groupNode
      .filter(d => !moved || moved.has(d))
      .attr('transform', d => `translate(${d.x},${d.y})`)

    s.edge
      .filter(d => !moved || moved.has(d.source) || moved.has(d.target))
      .select('path')
      .attr('d', d => canvasEdgePath(canvasEdgeGeometry(d.source, d.target, d.fromSide, d.toSide)))

    s.edgeLabels
      .filter(d => !moved || moved.has(d.source) || moved.has(d.target))
      .attr('transform', d => {
        const { x, y } = canvasEdgeMidpoint(
          canvasEdgeGeometry(d.source, d.target, d.fromSide, d.toSide),
        )
        return `translate(${x},${y})`
      })
  }

  s.measureEdgeLabels = function measureEdgeLabels() {
    s.edgeLabels.each(function () {
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
}

export function bindDrag(s: CanvasScene) {
  if (!s.isEmbed) {
    let members: NodeData[] = []
    const scene = s.g.node()
    if (!scene) return
    const draggable = s.g.selectAll<SVGGElement, NodeData>(
      ':scope > .nodes > .node, :scope > .group-nodes > .node',
    )
    const drag = d3Drag<SVGGElement, NodeData>()
      .container(() => scene)
      .touchable(() => false)
      .clickDistance(4)
      .filter(event => {
        if (s.isHelpOpen || s.spacePressed || event.button) return false
        const target = event.target
        if (!(target instanceof Element)) return false
        if (
          target.closest(
            'a, button, [role="button"], input, textarea, select, summary, [contenteditable="true"]',
          )
        )
          return false
        return !(s.focusedNode?.contains(target) && target.closest('.canvas-card-preview'))
      })
      .on('start', function (_event, d) {
        s.nodeDragging = true
        members = s.membersOf(d)
        this.classList.add('is-dragging')
      })
      .on('drag', (event, d) => s.moveCard(d, event.x - d.x, event.y - d.y, members))
      .on('end', function () {
        s.nodeDragging = false
        this.classList.remove('is-dragging')
        members = []
        s.refreshBounds()
      })
    draggable.call(drag)
    addCanvasCleanup(s.lifecycle, () => draggable.on('.drag', null))
  }

  s.updatePositions()

  s.measureEdgeLabels()

  s.measureGroupLabels()

  void document.fonts.ready.then(() => {
    if (!isCanvasActive(s.container, s.lifecycle)) return
    s.measureEdgeLabels()
    s.groupLabelSize = 0
    s.measureGroupLabels()
    if (s.centerTitleMetrics.size > 0) {
      s.centerTitleMetrics.clear()
      s.centerTitleSize = 0
      if (s.container.classList.contains('canvas-skeleton-view')) s.fitCenterTitles()
    }
  })

  s.boundsAttr = s.container.getAttribute('data-canvas-bounds')

  if (s.boundsAttr) {
    s.bounds = JSON.parse(s.boundsAttr)
  } else if (s.isEmbed || s.isLive) {
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity

    for (const node of s.nodes) {
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
      s.bounds = { minX, minY, maxX, maxY }
    }
  }

  if (s.bounds) s.applyBoundsTransform(s.bounds, false)

  s.resizeObserver = new ResizeObserver(() => {
    const nextWidth = s.container.clientWidth
    const nextHeight = s.container.clientHeight
    if (!nextWidth || !nextHeight || (nextWidth === s.width && nextHeight === s.height)) return
    const previousWidth = s.width
    const previousHeight = s.height
    s.width = nextWidth
    s.height = nextHeight
    s.svg.attr('width', s.width).attr('height', s.height).attr('viewBox', [0, 0, s.width, s.height])
    s.svg.interrupt()
    if (s.isFitted && s.bounds) s.applyBoundsTransform(s.bounds, false)
    else if (s.zoomBehavior) {
      s.svg.call(
        s.zoomBehavior.transform,
        zoomIdentity
          .translate(
            s.currentTransform.x + (s.width - previousWidth) / 2,
            s.currentTransform.y + (s.height - previousHeight) / 2,
          )
          .scale(s.currentScale),
      )
    } else s.queueRender()
  })

  s.resizeObserver.observe(s.container)

  addCanvasCleanup(s.lifecycle, () => s.resizeObserver.disconnect())
}
