import type {
  ActivityCardChart,
  ActivityCardData,
  ActivityCardStat,
} from '../../util/press-kit-activity'
import type { TrainingCardBlock } from './press-kit-day'
import {
  FORMATS,
  metricShown,
  type CardModel,
  type FormatKey,
  type Palette,
  type Fonts,
} from './press-kit-card'
import { cardRoute, routeSvg } from './press-kit-route'
import { node, createSvgText, graphic, sportIcon } from './press-kit-svg'
import { cardDate, runsToPlain } from './press-kit-text'

const ORANGE = '#fc4c02'

export function paintTrainingChart(
  parent: SVGElement,
  chart: ActivityCardChart,
  x: number,
  y: number,
  width: number,
  height: number,
  ink: string,
  fonts: Fonts,
): void {
  const { text, fitText, fitValue } = createSvgText(fonts)
  const size = Math.min(width / 28, height / (chart.kind === 'bars' ? chart.points.length + 4 : 9))
  fitText(parent, chart.title, x, y + size, width, size, ink, fonts.headerWeight, fonts.header)
  const points = chart.points.filter(point => point.y != null)
  if (!points.length) return
  const max = Math.max(...points.map(point => point.y ?? 0), 1)
  if (chart.key === 'runPace') {
    const appearance =
      document.querySelector('.tri-training-zone-grid--pace, .tri-day-embed, .triathlon') ??
      document.documentElement
    const color = getComputedStyle(appearance).getPropertyValue('--tri-zone-pace').trim() || ink
    const grid = node('g', { 'data-chart': chart.key })
    parent.append(grid)
    const total = points.reduce((sum, point) => sum + (point.y ?? 0), 0)
    const rowHeight = (height - size * 3) / points.length
    const top = y + size * 2.4
    const left = x + width * 0.09
    const trackWidth = width * 0.46
    for (let tick = 0; tick <= 4; tick++)
      grid.append(
        node('path', {
          d: `M${left + (trackWidth * tick) / 4} ${top}V${top + rowHeight * points.length}`,
          stroke: ink,
          opacity: 0.1,
        }),
      )
    points.forEach((point, index) => {
      const baseline = top + rowHeight * (index + 0.75)
      const barTop = top + rowHeight * (index + 0.1)
      const barHeight = rowHeight * 0.62
      const share = total > 0 ? ((point.y ?? 0) / total) * 100 : 0
      const row = node('g', {
        'data-zone': point.label,
        'aria-label': `${point.label}, ${point.range ?? '—'}, ${point.value}, ${share.toFixed(1)}%`,
      })
      grid.append(row)
      fitValue(row, point.label, x, baseline, width * 0.07, size * 0.85, ink)
      row.append(
        node('rect', {
          x: left,
          y: barTop,
          width: trackWidth,
          height: barHeight,
          fill: ink,
          opacity: 0.08,
        }),
        node('rect', {
          x: left,
          y: barTop,
          width: (trackWidth * (point.y ?? 0)) / max,
          height: barHeight,
          fill: color,
          opacity: index === 5 ? 0.91 : 0.42 + index * 0.1,
        }),
      )
      const values = node('g', { 'text-anchor': 'end' })
      row.append(values)
      fitValue(
        values,
        `${share.toFixed(1)}%`,
        x + width * 0.69,
        baseline,
        width * 0.12,
        size * 0.8,
        ink,
      )
      fitValue(values, point.range ?? '—', x + width, baseline, width * 0.28, size * 0.85, ink)
    })
  } else if (chart.kind === 'bars') {
    const rowHeight = (height - size * 3) / points.length
    points.forEach((point, index) => {
      const baseline = y + size * 2.4 + rowHeight * (index + 0.75)
      fitValue(parent, point.label, x, baseline, width * 0.32, size * 0.85, ink)
      parent.append(
        node('rect', {
          x: x + width * 0.34,
          y: baseline - rowHeight * 0.65,
          width: (width * 0.44 * (point.y ?? 0)) / max,
          height: rowHeight * 0.62,
          rx: rowHeight * 0.3,
          fill: ORANGE,
        }),
      )
      fitValue(parent, point.value, x + width * 0.8, baseline, width * 0.2, size * 0.85, ink)
    })
  } else {
    const top = y + size * 2.1
    const bottom = y + height - size * 2.4
    const minX = chart.points[0].x
    const spanX = Math.max(1, chart.points.at(-1)!.x - minX)
    parent.append(node('path', { d: `M${x} ${bottom}H${x + width}`, stroke: ink, opacity: 0.25 }))
    let path = ''
    let connected = false
    for (const point of chart.points) {
      if (point.y == null) {
        connected = false
        continue
      }
      path += `${connected ? 'L' : 'M'}${x + ((point.x - minX) / spanX) * width} ${bottom - (point.y / max) * (bottom - top)} `
      connected = true
    }
    parent.append(
      node('path', {
        d: path,
        fill: 'none',
        stroke: ORANGE,
        'stroke-width': 2,
        'stroke-linejoin': 'round',
      }),
    )
    text(parent, `${Math.round(max)} ${chart.axis}`, x, top, size * 0.75, ink, 400, fonts.mono)
    text(parent, chart.points[0].label, x, bottom + size, size * 0.75, ink, 400, fonts.mono)
    fitValue(
      parent,
      chart.points.at(-1)!.label,
      x + width * 0.85,
      bottom + size,
      width * 0.15,
      size * 0.75,
      ink,
    )
  }
  if (chart.note) fitValue(parent, chart.note, x, y + height, width, size * 0.7, ink)
}

function metricGrid(
  parent: SVGElement,
  stats: ActivityCardStat[],
  x: number,
  y: number,
  width: number,
  height: number,
  ink: string,
  fonts: Fonts,
  boxed = false,
): void {
  const { fitText, fitValue, lines } = createSvgText(fonts)
  if (!stats.length) return
  const columns = Math.min(3, stats.length)
  const rows = Math.ceil(stats.length / columns)
  const rowHeight = boxed ? Math.min(height / rows, width / columns / 2) : height / rows
  const size = Math.min(width / columns / 6.5, rowHeight / 3.5)
  stats.forEach((stat, index) => {
    const left = x + ((index % columns) * width) / columns
    const top = y + Math.floor(index / columns) * rowHeight
    const inset = boxed ? size * 0.35 : 0
    const textWidth = (width / columns) * 0.92 - inset * 2
    if (boxed)
      parent.append(
        node('rect', {
          x: left + 1,
          y: top + 1,
          width: (width / columns) * 0.96 - 2,
          height: rowHeight * 0.9 - 2,
          fill: 'none',
          stroke: ink,
          'stroke-width': 1.5,
        }),
      )
    fitText(parent, stat.label, left + inset, top + size * 1.1, textWidth, size * 0.6, ink, 400)
    const values = boxed ? [stat.value] : lines(stat.value, textWidth, size, 400, fonts.mono)
    values.forEach((value, line) =>
      fitValue(parent, value, left + inset, top + size * (2.3 + line * 1.15), textWidth, size, ink),
    )
  })
}

function paintOverlay(
  parent: SVGElement,
  activity: ActivityCardData,
  model: CardModel,
  x: number,
  y: number,
  width: number,
  height: number,
  palette: Palette,
  fonts: Fonts,
): void {
  const { text, fitText, fitValue, titleRow } = createSvgText(fonts)
  const stats = activity.stats.filter(stat => metricShown(model, activity, stat))
  const route = model.show.map ? cardRoute(activity.route) : undefined
  const charts = activity.charts.filter(chart => model.show[chart.key])
  const chartHeight = charts.length ? (height * 0.28) / charts.length : 0
  const contentHeight = height - chartHeight * charts.length
  const ink = palette.ink
  const brandSize = width * 0.04
  fitText(
    parent,
    'aarnphm.xyz',
    x,
    y + brandSize,
    width * 0.7,
    brandSize,
    ink,
    fonts.headerWeight,
    fonts.header,
  )
  let top = y + brandSize * 2
  if (model.show.title) {
    top +=
      titleRow(parent, activity.title, activity.sport, x, top, width, brandSize * 0.8, ink) +
      brandSize * 0.5
  } else {
    const icon = sportIcon(activity.sport, x, top, brandSize * 1.4, ink)
    if (icon) parent.append(icon)
    top += brandSize * 1.8
  }
  if (model.show.date) {
    text(
      parent,
      cardDate(activity.date),
      x,
      top + brandSize * 0.7,
      brandSize * 0.65,
      ink,
      400,
      fonts.mono,
    )
    top += brandSize * 1.4
  }
  const remaining = y + contentHeight - top
  const layout = model.overlay
  // Stack and chips read stats first, so the route takes the band below them. The stats keep
  // their natural height until the route would fall under 30% of the space.
  const statsFirst = layout === 'stack' || layout === 'chips'
  let statHeight = remaining
  if (route) {
    const natural =
      layout === 'stack'
        ? stats.length * width * 0.2
        : Math.ceil(stats.length / 3) * (width / 3 / 2)
    const routeHeight =
      layout === 'route'
        ? remaining
        : statsFirst
          ? Math.min(remaining * 0.6, Math.max(remaining * 0.3, remaining - natural))
          : remaining * (layout === 'receipt' || layout === 'rail' ? 0.6 : 0.45)
    graphic(
      parent,
      routeSvg(route, ORANGE, 'transparent'),
      x,
      statsFirst ? y + contentHeight - routeHeight : top,
      width * (layout === 'rail' ? 0.78 : 1),
      routeHeight,
    )
    if (statsFirst) statHeight = remaining - routeHeight - brandSize * 0.5
    else if (layout !== 'rail') {
      top += routeHeight
      statHeight -= routeHeight
    }
  }
  if (layout === 'grid')
    metricGrid(parent, stats, x, top + statHeight * 0.1, width, statHeight * 0.85, ink, fonts)
  else if (layout === 'stack') {
    const step = Math.min(statHeight / Math.max(1, stats.length), width * 0.2)
    const size = Math.min(width * 0.09, step / 2.8)
    stats.forEach((stat, index) => {
      const group = node('g', { 'text-anchor': 'middle' })
      fitText(
        group,
        stat.label,
        x + width / 2,
        top + step * index + size,
        width,
        size * 0.45,
        ink,
        600,
      )
      fitValue(group, stat.value, x + width / 2, top + step * index + size * 2.2, width, size, ink)
      parent.append(group)
    })
  } else if (layout === 'receipt') {
    const step = Math.min(statHeight / Math.max(1, stats.length + 1), width * 0.065)
    const start = y + contentHeight - step * (stats.length + 1)
    stats.forEach((stat, index) => {
      const baseline = start + step * (index + 1)
      fitText(parent, stat.label, x, baseline, width * 0.31, step * 0.65, ink)
      parent.append(
        node('path', {
          d: `M${x + width * 0.32} ${baseline - step * 0.15}H${x + width * 0.65}`,
          stroke: ink,
          'stroke-dasharray': '1 6',
        }),
      )
      fitValue(parent, stat.value, x + width * 0.68, baseline, width * 0.32, step * 0.65, ink)
    })
  } else if (layout === 'chips')
    metricGrid(parent, stats, x, top, width, statHeight, ink, fonts, true)
  else if (layout === 'rail') {
    const step = contentHeight / Math.max(1, Math.ceil(stats.length / 2))
    stats.forEach((stat, index) => {
      const group = node('g', {
        transform: `translate(${x + width * 0.92 - (index % 2) * brandSize * 1.8},${y + contentHeight - Math.floor(index / 2) * step}) rotate(-90)`,
      })
      fitValue(group, `${stat.label}  ${stat.value}`, 0, 0, step * 0.9, brandSize * 0.68, ink)
      parent.append(group)
    })
  }
  charts.forEach((chart, index) =>
    paintTrainingChart(
      parent,
      chart,
      x,
      y + contentHeight + index * chartHeight,
      width,
      chartHeight,
      ink,
      fonts,
    ),
  )
}

export function stickerCardSvg(
  model: CardModel,
  format: FormatKey,
  palette: Palette,
  fonts: Fonts,
): string {
  const { fitText, fitValue, lines } = createSvgText(fonts)
  const spec = FORMATS[format]
  const [pt, pr, pb, pl] = spec.pad
  const svg = node('svg', {
    xmlns: 'http://www.w3.org/2000/svg',
    width: spec.width,
    height: spec.height,
    viewBox: `0 0 ${spec.width} ${spec.height}`,
  })
  const title = node('title')
  title.textContent = runsToPlain(model.title)
  svg.append(title)
  if (!model.show.transparent)
    svg.append(node('rect', { width: spec.width, height: spec.height, fill: palette.bg }))
  const blocks: TrainingCardBlock[] = model.activity
    ? [
        {
          id: 'activity',
          kind: 'activity',
          title: model.activity.title,
          url: model.activity.url,
          activities: [model.activity],
          rows: [],
        },
      ]
    : (model.day?.blocks ?? [])
  const columns = blocks.length > 1 ? 2 : 1
  const rows = Math.max(1, Math.ceil(blocks.length / columns))
  const gap = spec.gap * 0.5
  const width = (spec.width - pl - pr - gap * (columns - 1)) / columns
  const height = (spec.height - pt - pb - gap * (rows - 1)) / rows
  blocks.forEach((block, index) => {
    const x = pl + (index % columns) * (width + gap)
    const y = pt + Math.floor(index / columns) * (height + gap)
    if (block.kind === 'activity')
      paintOverlay(svg, block.activities[0], model, x, y, width, height, palette, fonts)
    else {
      const size = Math.min(width / 15, height / 16)
      fitText(svg, block.title, x, y + size, width, size, palette.ink, 700)
      let top = y + size * 2
      if (block.graphic) {
        graphic(svg, block.graphic.svg, x, top, width, height * 0.3)
        top += height * 0.3
      }
      if (block.kind === 'entry' || block.kind === 'memo') {
        const prose = block.text ?? 'mémo audio'
        let font = size
        let proseLines = lines(prose, width, font)
        while (proseLines.length * font * 1.3 > y + height - top && font > 6) {
          font *= 0.9
          proseLines = lines(prose, width, font)
        }
        proseLines.forEach((line, row) =>
          fitText(svg, line, x, top + font * (1 + row * 1.3), width, font, palette.ink),
        )
      } else {
        const values = block.rows.filter(row => !row.key || model.show[row.key])
        const step = Math.min(size * 1.8, (y + height - top) / Math.max(1, values.length))
        values.forEach((row, rowIndex) =>
          fitValue(
            svg,
            `${row.label} · ${row.values.join(' / ')}`,
            x,
            top + step * (rowIndex + 1),
            width,
            step * 0.7,
            palette.ink,
          ),
        )
      }
    }
  })
  return svg.outerHTML
}
