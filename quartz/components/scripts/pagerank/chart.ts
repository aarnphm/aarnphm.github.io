import { logTicks, type PageRankModel, type PageRankNode } from '../../../util/pagerank-view'

const SVG = 'http://www.w3.org/2000/svg'
const height = 240
const margin = { top: 12, right: 14, bottom: 34, left: 46 }

const svgNode = <K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number>,
): SVGElementTagNameMap[K] => {
  const node = document.createElementNS(SVG, tag)
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value))
  return node
}

const tickLabel = (value: number, unit = '') =>
  (value >= 1000 ? `${value / 1000}k` : value >= 1 ? String(value) : value.toPrecision(1)) + unit

export interface RankSizeChart {
  /** Moves the selected (filled) and pointer (ring) markers; `null` hides one. */
  mark(selected: PageRankNode | null, pointer: PageRankNode | null): void
  render(width: number): void
}

/** Rank–size plot: log position against log ×mean, one polyline for the whole corpus. */
export function rankSizeChart(host: HTMLElement, model: PageRankModel): RankSizeChart {
  const { byPosition, summary } = model
  const n = byPosition.length
  const yMin = byPosition[n - 1]?.times ?? 1
  const yMax = summary.max || 1
  let width = 0
  let place = (_node: PageRankNode) => [0, 0]
  const selectedMark = svgNode('circle', { r: 4.5, class: 'pagerank-mark-selected' })
  const pointerMark = svgNode('circle', { r: 5.5, class: 'pagerank-mark-pointer' })
  let marks: [PageRankNode | null, PageRankNode | null] = [null, null]

  const move = (mark: SVGCircleElement, node: PageRankNode | null) => {
    mark.toggleAttribute('hidden', !node)
    if (!node) return
    const [x, y] = place(node)
    mark.style.transform = `translate(${x}px, ${y}px)`
  }

  const chart: RankSizeChart = {
    mark(selected, pointer) {
      marks = [selected, pointer && pointer !== selected ? pointer : null]
      move(selectedMark, marks[0])
      move(pointerMark, marks[1])
    },
    render(next) {
      if (next === width || next <= 0 || n === 0) return
      width = next
      const plotW = width - margin.left - margin.right
      const plotH = height - margin.top - margin.bottom
      const lx = Math.log10(Math.max(n, 10))
      const ly0 = Math.log10(yMin)
      const ly1 = Math.log10(yMax)
      const sx = (position: number) => margin.left + (Math.log10(position) / lx) * plotW
      const sy = (times: number) =>
        margin.top + (1 - (Math.log10(times) - ly0) / (ly1 - ly0 || 1)) * plotH
      place = node => [sx(node.position), sy(node.times)]

      const svg = svgNode('svg', {
        viewBox: `0 0 ${width} ${height}`,
        width,
        height,
        role: 'img',
        'aria-label': `rank–size plot of ${n} notes: score from ${yMax.toFixed(2)}× down to ${yMin.toFixed(2)}× the mean, both axes logarithmic`,
      })
      const grid = svgNode('g', { class: 'pagerank-grid' })
      const ticks = svgNode('g', { class: 'pagerank-ticks' })
      for (const value of logTicks(1, n)) {
        const x = sx(value)
        grid.append(svgNode('line', { x1: x, x2: x, y1: margin.top, y2: margin.top + plotH }))
        const label = svgNode('text', {
          x,
          y: height - margin.bottom + 15,
          'text-anchor': 'middle',
        })
        label.textContent = tickLabel(value)
        ticks.append(label)
      }
      for (const value of logTicks(yMin, yMax)) {
        const y = sy(value)
        grid.append(svgNode('line', { x1: margin.left, x2: margin.left + plotW, y1: y, y2: y }))
        const label = svgNode('text', { x: margin.left - 6, y: y + 4, 'text-anchor': 'end' })
        label.textContent = tickLabel(value, '×')
        ticks.append(label)
      }
      const xTitle = svgNode('text', {
        x: margin.left + plotW,
        y: height - 4,
        'text-anchor': 'end',
        class: 'pagerank-axis-title',
      })
      xTitle.textContent = 'rank (log)'
      // The curve falls from the top left, so the top right corner stays clear for the label.
      const yTitle = svgNode('text', {
        x: margin.left + plotW - 4,
        y: margin.top + 14,
        'text-anchor': 'end',
        class: 'pagerank-axis-title',
      })
      yTitle.textContent = '×mean (log)'

      let d = ''
      let lastX = -1
      for (const [index, node] of byPosition.entries()) {
        const x = sx(node.position)
        // Sub-pixel steps in the long tail draw nothing visible.
        if (x - lastX < 0.5 && index !== n - 1) continue
        d += `${d ? 'L' : 'M'}${x.toFixed(1)} ${sy(node.times).toFixed(1)}`
        lastX = x
      }
      const fitAt = (position: number) =>
        sy(10 ** (summary.intercept - summary.alpha * Math.log10(position)))
      const clip = svgNode('clipPath', { id: 'pagerank-plot-clip' })
      clip.append(svgNode('rect', { x: margin.left, y: margin.top, width: plotW, height: plotH }))
      const fit = svgNode('path', {
        d: `M${sx(1)} ${fitAt(1)}L${sx(n)} ${fitAt(n)}`,
        class: 'pagerank-fit',
        'clip-path': 'url(#pagerank-plot-clip)',
      })
      const line = svgNode('path', { d, class: 'pagerank-curve' })
      const frame = svgNode('rect', {
        x: margin.left,
        y: margin.top,
        width: plotW,
        height: plotH,
        class: 'pagerank-plot-frame',
      })
      svg.append(clip, grid, frame, fit, line, ticks, xTitle, yTitle, pointerMark, selectedMark)
      host.replaceChildren(svg)
      chart.mark(...marks)
    },
  }
  return chart
}
