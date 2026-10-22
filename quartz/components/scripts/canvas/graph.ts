import { select } from 'd3'
import type { CanvasMeta, LinkData, NodeData } from './data'
import type { CanvasScene } from './scene'
import { EDGE_LABEL_LINE_HEIGHT, fitSvgText, measureSvgText } from './labels'

export function installGraph(s: CanvasScene) {
  s.measureGroupLabels = function measureGroupLabels() {
    const size = Math.max(24, 13 / s.currentScale)
    const quantized = 2 ** (Math.round(4 * Math.log2(size)) / 4)
    if (quantized === s.groupLabelSize) return
    s.groupLabelSize = quantized
    for (const label of s.groupLabels) {
      label.element.setAttribute('font-size', `${quantized}px`)
      label.element.setAttribute('y', String(-quantized / 3))
      const metrics = measureSvgText(label.element, label.text)
      label.fontSize = quantized
      const shown = fitSvgText(label.text, metrics, label.group.width)
      if (label.element.textContent !== shown) label.element.textContent = shown
    }
  }
}

export function drawGraph(s: CanvasScene) {
  s.nodes = s.canvasData.nodes.map(n => {
    const meta: CanvasMeta = s.metaMap[n.id] ?? {}
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

  s.nodeMap = new Map<string, NodeData>()

  s.nodes.forEach(n => s.nodeMap.set(n.id, n))

  s.groupNodes = s.nodes.filter(n => n.type === 'group')

  s.regularNodes = s.nodes.filter(n => n.type !== 'group')

  s.links = s.canvasData.edges
    .map(e => {
      const source = s.nodeMap.get(e.fromNode)
      const target = s.nodeMap.get(e.toNode)
      if (!source || !target) return null
      return { ...e, source, target }
    })
    .filter((l): l is LinkData => l !== null)

  s.groupNode = s.groupNodeGroup
    .selectAll('g.node')
    .data(s.groupNodes)
    .join('g')
    .attr('class', 'node node-group')
    .attr('data-node-id', d => d.id)
    .attr('data-color', d => d.color || '')

  s.groupNode
    .append('rect')
    .attr('class', 'node-bg')
    .attr('width', d => d.width)
    .attr('height', d => d.height)

  s.groupNode
    .append('rect')
    .attr('class', 'node-border-overlay')
    .attr('width', d => d.width)
    .attr('height', d => d.height)
    .attr('stroke', 'var(--gray)')
    .attr('stroke-width', 1.5)
    .attr('fill', 'none')
    .attr('pointer-events', 'none')

  // Labels use canvas coordinates so their size and width follow the cards through zoom.

  // Labels use canvas coordinates so their size and width follow the cards through zoom.
  s.groupLabels = []

  s.groupNode
    .append('text')
    .attr('class', 'node-group-label')
    .attr('x', 0)
    .attr('y', -8)
    .attr('font-size', '24px')
    .each(function (d: NodeData) {
      const text = d.label || 'Group'
      this.textContent = text
      s.groupLabels.push({ element: this, group: d, text, fontSize: 24 })
    })

  // Counter-scaled: the title keeps about 13px on screen at any zoom, clamped to its group.

  // Counter-scaled: the title keeps about 13px on screen at any zoom, clamped to its group.
  s.groupLabelSize = 0

  s.arrowId = `arrow-${s.pattern.attr('id')}`

  s.edge = s.edgeGroup
    .selectAll<SVGGElement, LinkData>('g.edge')
    .data(s.links, d => d.id)
    .join(
      enter => {
        const edgeEnter = enter.append('g').attr('class', 'edge')
        edgeEnter
          .append('path')
          .attr('stroke', d => d.color || 'var(--gray)')
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('marker-end', `url(#${s.arrowId})`)
        return edgeEnter
      },
      update => update,
      exit => exit.remove(),
    )

  s.edge
    .select('path')
    .attr('stroke', d => d.color || 'var(--gray)')
    .attr('stroke-width', 2)
    .attr('fill', 'none')
    .attr('marker-end', `url(#${s.arrowId})`)

  s.labeledLinks = s.links.filter(
    (d): d is LinkData & { label: string } =>
      typeof d.label === 'string' && d.label.trim().length > 0,
  )

  s.edgeLabels = s.edgeLabelGroup
    .selectAll<SVGGElement, LinkData & { label: string }>('g.edge-label-group')
    .data(s.labeledLinks, d => d.id)
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

  s.edgeLabels.select('text').each(function (d) {
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

  s.svg
    .append('defs')
    .append('marker')
    .attr('id', s.arrowId)
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 8)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', 'var(--gray)')
}
