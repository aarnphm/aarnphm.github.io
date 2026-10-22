import { select } from 'd3'
import type { NodeData } from './data'
import type { CanvasScene } from './scene'
import { renderTextNodeContent, resolveFileNodeUrl, wrapTextHtml } from './content'
import { cardName, measureSvgText, wrapSvgText, type SvgTextMetrics } from './labels'

export function installCards(s: CanvasScene) {
  s.fitCenterTitles = function fitCenterTitles() {
    const size = 16 * 2 ** Math.max(0, Math.ceil(4 * Math.log2(0.75 / s.currentScale)) / 4)
    if (size === s.centerTitleSize) return
    s.centerTitleSize = size
    s.node.select<SVGTextElement>('.node-title-center').each(function (d) {
      const text = cardName(d)
      let metrics = s.centerTitleMetrics.get(this)
      if (!metrics) {
        this.setAttribute('font-size', '16px')
        metrics = measureSvgText(this, text)
        s.centerTitleMetrics.set(this, metrics)
      }
      // a short card caps the title so it never spills over its neighbours
      const px = Math.min(size, d.height * 0.4)
      this.setAttribute('font-size', `${px}px`)
      const lines = wrapSvgText(text, metrics, ((d.width - 32) * 16) / px)
      if (lines.length === 1) {
        this.textContent = lines[0]
        return
      }
      this.textContent = ''
      lines.forEach((line, i) => {
        const tspan = document.createElementNS('http://www.w3.org/2000/svg', 'tspan')
        tspan.setAttribute('x', String(d.width / 2))
        tspan.setAttribute('dy', i === 0 ? '-0.55em' : '1.1em')
        tspan.textContent = line
        this.append(tspan)
      })
    })
  }
}

export function drawCards(s: CanvasScene) {
  s.node = s.nodeGroup
    .selectAll<SVGGElement, NodeData>('g.node')
    .data(s.regularNodes)
    .join('g')
    .attr('class', d => `node node-${d.type}`)
    .attr('data-node-id', d => d.id)
    .attr('data-color', d => d.color || '')
    .attr('tabindex', s.isEmbed ? null : 0)
    .attr('role', 'group')
    .attr('aria-label', d => cardName(d) || d.url || 'Text card')

  s.node
    .append('rect')
    .attr('class', 'node-bg')
    .attr('width', d => d.width)
    .attr('height', d => d.height)
    .attr('stroke', 'var(--gray)')
    .attr('stroke-width', 1.5)

  s.fileNodes = s.node.filter(d => d.type === 'file')

  s.fileNodes
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
    .attr('aria-label', d => `Open ${cardName(d) || 'note'}`)
    .text(d => cardName(d))

  s.fileNodes
    .append('text')
    .attr('class', 'node-title-text node-title-center')
    .attr('x', d => d.width / 2)
    .attr('y', d => d.height / 2)
    .attr('text-anchor', 'middle')
    .attr('dominant-baseline', 'middle')
    .attr('font-size', '16px')
    .attr('font-weight', '600')
    .text(d => cardName(d))

  // Centered titles are hidden until the skeleton view, so they can only be measured there. They

  // Centered titles are hidden until the skeleton view, so they can only be measured there. They
  // are measured once at 16px, then grow as the view zooms out so they stay about 12px on screen;

  // Centered titles are hidden until the skeleton view, so they can only be measured there. They
  // are measured once at 16px, then grow as the view zooms out so they stay about 12px on screen;
  // the size moves in quarter octaves, so a zoom gesture rewrites them a few times, not every frame.

  // Centered titles are hidden until the skeleton view, so they can only be measured there. They
  // are measured once at 16px, then grow as the view zooms out so they stay about 12px on screen;
  // the size moves in quarter octaves, so a zoom gesture rewrites them a few times, not every frame.
  s.centerTitleMetrics = new Map<SVGTextElement, SvgTextMetrics>()

  s.centerTitleSize = 0

  s.textNodes = s.node.filter(d => d.type === 'text')

  s.textNodes
    .append('text')
    .attr('class', 'node-title-text node-title-center')
    .attr('x', d => d.width / 2)
    .attr('y', d => d.height / 2)
    .attr('text-anchor', 'middle')
    .attr('dominant-baseline', 'middle')
    .attr('font-size', '16px')
    .attr('font-weight', '600')
    .text(d => cardName(d))

  s.textNodes.each(function (d: NodeData) {
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

  s.node
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
        const hasDescription = typeof d.description === 'string' && d.description.trim().length > 0
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

  s.textNodes.selectAll('.node-skeleton').raise()

  s.node
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
}
