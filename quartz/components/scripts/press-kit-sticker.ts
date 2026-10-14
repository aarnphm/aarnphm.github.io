import type { ActivityCardData, ActivityCardStat } from '../../util/press-kit-activity'
import { triLabel } from '../../util/triathlon-i18n'
import {
  FORMATS,
  cardResolution,
  metricShown,
  type CardModel,
  type FormatKey,
  type Palette,
  type Fonts,
} from './press-kit-card'
import { paintTrainingChart, trainingCharts, trainingChartWeight } from './press-kit-chart'
import { cardBlockDisplay, type BlockDisplay, type TrainingCardBlock } from './press-kit-day'
import { cardLocale } from './press-kit-i18n'
import { svgLayer, type LayerRenderer } from './press-kit-layers'
import { cardRoute, routeSvg } from './press-kit-route'
import { node, createSvgText, graphic, sportIcon } from './press-kit-svg'
import { cardDate, runsToPlain } from './press-kit-text'

const ORANGE = '#fc4c02'

function metricGrid(
  parent: SVGElement,
  stats: ActivityCardStat[],
  x: number,
  y: number,
  width: number,
  height: number,
  ink: string,
  fonts: Fonts,
  scale: number,
  boxed = false,
  gap = 0,
): void {
  const { fitText, fitValue, lines } = createSvgText(fonts, scale)
  if (!stats.length) return
  const columns = Math.min(3, stats.length)
  const rows = Math.ceil(stats.length / columns)
  const spacing = boxed ? Math.max(0, Math.min(gap, width / columns / 3, height / rows / 3)) : 0
  const cellWidth = (width - spacing * (columns - 1)) / columns
  const rowHeight = boxed
    ? Math.min((height - spacing * (rows - 1)) / rows, cellWidth / 2)
    : height / rows
  const size = Math.min(cellWidth / 6.5, rowHeight / 3.5)
  stats.forEach((stat, index) => {
    const metric = svgLayer(parent, `stat/${stat.key}`, stat.label)
    const left = x + (index % columns) * (cellWidth + spacing)
    const top = y + Math.floor(index / columns) * (rowHeight + spacing)
    const inset = boxed ? size * 0.35 : 0
    const textWidth = cellWidth * (boxed ? 1 : 0.92) - inset * 2
    if (boxed)
      metric.append(
        node('rect', {
          x: left,
          y: top,
          width: cellWidth,
          height: rowHeight,
          fill: 'none',
          stroke: ink,
          'stroke-width': 1.5,
        }),
      )
    fitText(metric, stat.label, left + inset, top + size * 1.1, textWidth, size * 0.6, ink, 400)
    const values = boxed ? [stat.value] : lines(stat.value, textWidth, size, 400, fonts.mono)
    values.forEach((value, line) =>
      fitValue(metric, value, left + inset, top + size * (2.3 + line * 1.15), textWidth, size, ink),
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
  display: BlockDisplay,
  scale: number,
  renderers?: Map<string, LayerRenderer>,
): void {
  const { text, fitText, fitValue, metricRow, titleRow } = createSvgText(fonts, scale)
  const stats = display.metrics
    ? activity.stats.filter(stat => metricShown(model, activity, stat))
    : []
  const route = model.show.map ? cardRoute(activity.route) : undefined
  const charts = trainingCharts(activity, model)
  const chartWeight = charts.reduce((total, chart) => total + trainingChartWeight(chart), 0)
  const chartsHeight = height * Math.min(0.72, 0.28 * chartWeight)
  const contentHeight = height - chartsHeight
  const ink = palette.ink
  const headingSize = width * 0.04
  const centered = model.overlay === 'stack'
  let top = y
  if (display.title) {
    top +=
      titleRow(
        svgLayer(parent, 'title', activity.title),
        activity.title,
        activity.sport,
        x,
        top,
        width,
        headingSize * 0.8,
        ink,
        centered ? 'center' : 'left',
      ) +
      headingSize * 0.5
  } else {
    const iconSize = headingSize * 1.4
    const icon = sportIcon(
      activity.sport,
      centered ? x + (width - iconSize) / 2 : x,
      top,
      iconSize,
      ink,
    )
    if (icon) svgLayer(parent, 'title', activity.title).append(icon)
    top += headingSize * 1.8
  }
  if (display.date) {
    const date = svgLayer(parent, 'date', 'date')
    date.setAttribute('text-anchor', centered ? 'middle' : 'start')
    text(
      date,
      cardDate(activity.date, model.locale ?? cardLocale()),
      centered ? x + width / 2 : x,
      top + headingSize * 0.7,
      headingSize * 0.65,
      ink,
      400,
      fonts.mono,
    )
    top += headingSize * 1.4
  }
  const remaining = y + contentHeight - top
  const layout = model.overlay
  // Stack and chips read stats first. On a wide block the stats take the left half and the
  // route the right half; otherwise the route takes the band below the stats, which keep their
  // natural height until the route would fall under 30% of the space.
  const statsFirst = layout === 'stack' || layout === 'chips'
  const beside = statsFirst && !!route && width > remaining * 1.6
  const statWidth = beside ? width * 0.5 : width
  let statHeight = remaining
  if (route) {
    const natural =
      layout === 'stack'
        ? stats.length * width * 0.2
        : Math.ceil(stats.length / 3) * (width / 3 / 2)
    const routeHeight =
      layout === 'route' || beside
        ? remaining
        : statsFirst
          ? Math.min(remaining * 0.6, Math.max(remaining * 0.3, remaining - natural))
          : remaining * (layout === 'receipt' || layout === 'rail' ? 0.6 : 0.45)
    const gap = headingSize
    graphic(
      svgLayer(parent, 'route', 'tracé'),
      routeSvg(route, ORANGE, 'transparent'),
      beside ? x + statWidth + gap : x,
      statsFirst && !beside ? y + contentHeight - routeHeight : top,
      beside ? width - statWidth - gap : width * (layout === 'rail' ? 0.78 : 1),
      routeHeight,
    )
    if (statsFirst) {
      if (!beside) statHeight = remaining - routeHeight - headingSize * 0.5
    } else if (layout !== 'rail') {
      top += routeHeight
      statHeight -= routeHeight
    }
  }
  if (layout === 'grid')
    metricGrid(
      parent,
      stats,
      x,
      top + statHeight * 0.1,
      width,
      statHeight * 0.85,
      ink,
      fonts,
      scale,
    )
  else if (layout === 'stack') {
    const step = Math.min(statHeight / Math.max(1, stats.length), statWidth * 0.2)
    const size = Math.min(statWidth * 0.09, step / 2.8)
    stats.forEach((stat, index) => {
      const group = svgLayer(parent, `stat/${stat.key}`, stat.label)
      group.setAttribute('text-anchor', 'middle')
      fitText(
        group,
        stat.label,
        x + statWidth / 2,
        top + step * index + size,
        statWidth,
        size * 0.45,
        ink,
        600,
      )
      fitValue(
        group,
        stat.value,
        x + statWidth / 2,
        top + step * index + size * 2.2,
        statWidth,
        size,
        ink,
      )
      parent.append(group)
    })
  } else if (layout === 'receipt') {
    const step = Math.min(statHeight / Math.max(1, stats.length + 1), width * 0.065)
    const start = y + contentHeight - step * (stats.length + 1)
    stats.forEach((stat, index) => {
      const metric = svgLayer(parent, `stat/${stat.key}`, stat.label)
      const labels = node('g', { 'text-anchor': 'end' })
      metric.append(labels)
      const baseline = start + step * (index + 1)
      fitText(labels, stat.label, x + width * 0.31, baseline, width * 0.31, step * 0.65, ink)
      metric.append(
        node('path', {
          d: `M${x + width * 0.32} ${baseline - step * 0.15}H${x + width * 0.65}`,
          stroke: ink,
          'stroke-dasharray': '1 6',
        }),
      )
      fitValue(metric, stat.value, x + width * 0.68, baseline, width * 0.32, step * 0.65, ink)
    })
  } else if (layout === 'chips')
    metricGrid(
      parent,
      stats,
      x,
      top,
      statWidth,
      statHeight,
      ink,
      fonts,
      scale,
      true,
      model.metricGap,
    )
  else if (layout === 'rail') {
    const step = contentHeight / Math.max(1, Math.ceil(stats.length / 2))
    stats.forEach((stat, index) => {
      const group = svgLayer(parent, `stat/${stat.key}`, stat.label)
      group.setAttribute(
        'transform',
        `translate(${x + width * 0.92 - (index % 2) * headingSize * 1.8},${y + contentHeight - Math.floor(index / 2) * step}) rotate(-90)`,
      )
      metricRow(group, stat.label, stat.value, 0, 0, step * 0.9, headingSize * 0.68, ink)
      parent.append(group)
    })
  }
  let chartTop = y + contentHeight
  charts.forEach(chart => {
    const chartHeight = (chartsHeight * trainingChartWeight(chart)) / chartWeight
    paintTrainingChart(
      svgLayer(parent, `chart/${chart.key}`, chart.title),
      chart,
      x,
      chartTop,
      width,
      chartHeight,
      ink,
      fonts,
      scale,
      model.locale ?? cardLocale(),
      { renderers, logScale: model.powerCurveScale !== 'linear' },
    )
    chartTop += chartHeight
  })
}

export function stickerCardSvg(
  model: CardModel,
  format: FormatKey,
  palette: Palette,
  fonts: Fonts,
  renderers?: Map<string, LayerRenderer>,
): string {
  const resolution = cardResolution(format)
  const { fitText, fitValue, metricRow, lines } = createSvgText(fonts, resolution.scale)
  const spec = FORMATS[format]
  const [pt, pr, pb, pl] = spec.pad
  const svg = node('svg', {
    xmlns: 'http://www.w3.org/2000/svg',
    width: resolution.width,
    height: resolution.height,
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
  let contentTop = pt
  if (model.day && model.customTitle && model.show.title) {
    const size = spec.meta * 1.6
    fitText(
      svgLayer(svg, 'title', model.customTitle),
      model.customTitle,
      pl,
      contentTop + size,
      spec.width - pl - pr,
      size,
      palette.ink,
      500,
    )
    contentTop += size * 1.4 + spec.gap * 0.5
  }
  const columns = blocks.length > 1 ? 2 : 1
  const rows = Math.max(1, Math.ceil(blocks.length / columns))
  const gap = spec.gap * 0.5
  const width = (spec.width - pl - pr - gap * (columns - 1)) / columns
  const height = (spec.height - contentTop - pb - gap * (rows - 1)) / rows
  blocks.forEach((block, index) => {
    const group = node('g', { 'data-block': block.id })
    svg.append(group)
    const display = cardBlockDisplay(block, model.show)
    const locale = model.locale ?? cardLocale()
    const x = pl + (index % columns) * (width + gap)
    const y = contentTop + Math.floor(index / columns) * (height + gap)
    if (block.kind === 'activity')
      paintOverlay(
        group,
        block.activities[0],
        model,
        x,
        y,
        width,
        height,
        palette,
        fonts,
        display,
        resolution.scale,
        renderers,
      )
    else {
      const size = Math.min(width / 15, height / 16)
      let top = y
      if (display.title) {
        fitText(
          svgLayer(group, 'title', block.title),
          block.title,
          x,
          top + size,
          width,
          size,
          palette.ink,
          500,
        )
        top += size * 2
      }
      if (display.date && block.date) {
        fitValue(
          svgLayer(group, 'date', 'date'),
          block.date,
          x,
          top + size,
          width,
          size * 0.7,
          palette.muted,
        )
        top += size * 1.5
      }
      if (block.graphic) {
        graphic(
          svgLayer(group, 'graphic', 'graphique'),
          block.graphic.svg,
          x,
          top,
          width,
          height * 0.3,
        )
        top += height * 0.3
      }
      if (block.kind === 'entry' || block.kind === 'memo') {
        const prose = block.text ?? (locale === 'fr' ? 'mémo audio' : 'audio memo')
        let font = size
        let proseLines = lines(prose, width, font)
        const proseBottom = y + height - (display.metrics && block.reading ? size * 1.8 : 0)
        while (proseLines.length * font * 1.3 > proseBottom - top && font > 6) {
          font *= 0.9
          proseLines = lines(prose, width, font)
        }
        const proseLayer = svgLayer(group, 'text', 'texte')
        proseLines.forEach((line, row) =>
          fitText(proseLayer, line, x, top + font * (1 + row * 1.3), width, font, palette.ink),
        )
        if (display.metrics && block.reading) {
          fitValue(
            svgLayer(group, 'reading', 'temps de lecture'),
            block.reading,
            x,
            y + height - size * 0.25,
            width,
            size * 0.75,
            palette.muted,
          )
        }
      } else {
        const values = display.metrics
          ? block.rows.filter(row => !row.key || model.show[row.key])
          : []
        const step = Math.min(size * 1.8, (y + height - top) / Math.max(1, values.length))
        values.forEach((row, rowIndex) =>
          metricRow(
            svgLayer(group, `row/${row.key ?? rowIndex}`, triLabel(locale, row.label)),
            triLabel(locale, row.label),
            row.values.join(' / '),
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
