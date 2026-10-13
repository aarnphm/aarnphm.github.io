import type {
  ActivityCardChart,
  ActivityCardData,
  ActivityCardStat,
} from '../../util/press-kit-activity'
import type { Locale } from '../../util/triathlon-presentation'
import { triLabel, triText } from '../../util/triathlon-i18n'
import {
  FORMATS,
  cardResolution,
  metricShown,
  type CardModel,
  type FormatKey,
  type Palette,
  type Fonts,
} from './press-kit-card'
import { cardBlockDisplay, type BlockDisplay, type TrainingCardBlock } from './press-kit-day'
import { cardLocale } from './press-kit-i18n'
import { svgLayer, type LayerRenderer } from './press-kit-layers'
import { cardRoute, routeSvg } from './press-kit-route'
import { node, createSvgText, graphic, sportIcon, svgImage } from './press-kit-svg'
import { cardDate, runsToPlain } from './press-kit-text'

const ORANGE = '#fc4c02'

interface ChartTypography {
  width: number
  columnWidth: number
  columns: number
  size: number
  labelSize: number
}

interface ChartRenderOptions {
  renderers?: Map<string, LayerRenderer>
  typography?: ChartTypography
  color?: string
}

export function paintTrainingChart(
  parent: SVGElement,
  chart: ActivityCardChart,
  x: number,
  y: number,
  width: number,
  height: number,
  ink: string,
  fonts: Fonts,
  scale: number,
  locale: Locale = cardLocale(),
  options?: ChartRenderOptions,
): void {
  const svgText = createSvgText(fonts, scale)
  const { text, measure } = svgText
  const points = chart.points.filter(point => point.y != null)
  if (!points.length) return
  const zones = chart.key === 'runPace'
  const columns =
    chart.key === 'runLaps'
      ? 1
      : (options?.typography?.columns ??
        (chart.kind === 'bars' && !zones && points.length > 8 && width / height > 3 ? 2 : 1))
  const rows = Math.ceil(points.length / columns)
  const size = options?.typography?.size ?? Math.min(width / 30, height / 10)
  const gap = size * 1.5
  const columnWidth = (width - (columns - 1) * gap) / columns
  const rowHeight = (height - size * 3) / rows
  const labelSize =
    options?.typography?.labelSize ?? Math.min(size * 0.8, rowHeight * (columns > 1 ? 0.65 : 0.55))
  const textWidthRatio = (options?.typography?.columnWidth ?? columnWidth) / columnWidth
  // Fit against the original text slots while the chart's geometric slots change size.
  const fitText: typeof svgText.fitText = (...args) => {
    args[4] *= textWidthRatio
    svgText.fitText(...args)
  }
  const fitValue: typeof svgText.fitValue = (...args) => {
    args[4] *= textWidthRatio
    svgText.fitValue(...args)
  }
  const heading = `${triLabel(locale, chart.title)}${chart.key === 'cyclingPower' ? ' · 30 s' : ''}`
  fitText(parent, heading, x, y + size, columnWidth * 0.72, size, ink, 500)
  const max = Math.max(...points.map(point => point.y ?? 0), 1)
  const appearance =
    document.querySelector('.tri-training-zone-grid--pace, .tri-day-embed, .triathlon') ??
    document.documentElement
  const color =
    chart.key === 'runLaps'
      ? ORANGE
      : (options?.color ??
        (getComputedStyle(appearance)
          .getPropertyValue(chart.key === 'cyclingPower' ? '--tri-zone-power' : '--tri-zone-pace')
          .trim() ||
          ink))
  if (chart.kind === 'bars') {
    const grid = node('g', { 'data-chart': chart.key })
    parent.append(grid)
    const total = points.reduce((sum, point) => sum + (point.y ?? 0), 0)
    const top = y + size * 2
    const trackWidth = columnWidth * (zones ? 0.43 : 0.61)
    const headings = node('g', { 'text-anchor': 'end' })
    grid.append(headings)
    if (zones)
      fitText(
        headings,
        locale === 'fr' ? 'part' : 'share',
        x + width * 0.69,
        y + size,
        width * 0.12,
        size * 0.65,
        ink,
      )
    for (let column = 0; column < columns; column++) {
      const cx = x + column * (columnWidth + gap)
      const left = cx + columnWidth * (zones ? 0.1 : 0.14)
      fitText(
        headings,
        triText(locale, 'pace'),
        cx + columnWidth,
        y + size,
        columnWidth * 0.25,
        size * 0.65,
        ink,
      )
      for (let tick = 0; tick <= 4; tick++)
        grid.append(
          node('path', {
            d: `M${left + (trackWidth * tick) / 4} ${top}V${top + rowHeight * rows}`,
            stroke: ink,
            opacity: 0.08,
          }),
        )
      grid.append(
        node('path', {
          d: `M${cx} ${top + rowHeight * rows}H${cx + columnWidth}`,
          stroke: ink,
          opacity: 0.12,
        }),
      )
    }
    points.forEach((point, index) => {
      const column = Math.floor(index / rows)
      const cx = x + column * (columnWidth + gap)
      const left = cx + columnWidth * (zones ? 0.1 : 0.14)
      const rowTop = top + rowHeight * (index % rows)
      const baseline = rowTop + rowHeight / 2 + labelSize * 0.35
      const barHeight = Math.min(rowHeight * 0.4, labelSize * 0.7)
      const barTop = rowTop + (rowHeight - barHeight) / 2
      const share = total > 0 ? ((point.y ?? 0) / total) * 100 : 0
      const row = node('g', {
        'aria-label': zones
          ? `${point.label}, ${point.range ?? '—'}, ${point.value}, ${share.toFixed(1)}%`
          : `${point.label}, ${point.value}`,
      })
      grid.append(row)
      row.append(
        node('path', { d: `M${cx} ${rowTop}H${cx + columnWidth}`, stroke: ink, opacity: 0.12 }),
      )
      fitText(
        row,
        triLabel(locale, point.label),
        cx,
        baseline,
        columnWidth * (zones ? 0.08 : 0.12),
        labelSize,
        ink,
      )
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
          opacity:
            chart.key === 'runLaps' ? 1 : zones ? (index === 5 ? 0.91 : 0.42 + index * 0.1) : 0.7,
        }),
      )
      const values = node('g', { 'text-anchor': 'end' })
      row.append(values)
      if (zones)
        fitValue(
          values,
          `${share.toFixed(1)}%`,
          x + width * 0.69,
          baseline,
          width * 0.12,
          labelSize,
          ink,
        )
      fitValue(
        values,
        zones ? (point.range ?? '—') : point.value,
        cx + columnWidth,
        baseline,
        columnWidth * (zones ? 0.28 : 0.22),
        labelSize,
        ink,
      )
    })
  } else {
    const top = y + size * 2
    const bottom = y + height - size
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
        stroke: color,
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
  if (chart.note && chart.key !== 'cyclingPower')
    svgText.fitValue(
      parent,
      chart.note,
      x,
      y + height,
      options?.typography?.width ?? width,
      size * 0.7,
      ink,
    )

  const id = parent.getAttribute('data-card-layer')
  if (id && options?.renderers) {
    const fittedWidth = (
      value: string,
      slot: number,
      fontSize: number,
      weight = 400,
      family = fonts.body,
    ) => {
      const measured = measure(value, fontSize, weight, family)?.width ?? 0
      const fitted = measured > slot ? (fontSize * slot) / measured : fontSize
      return (measure(value, fitted, weight, family)?.width ?? 0) + 4
    }
    const columnMinimum = Math.max(
      size * 6,
      fittedWidth(heading, columnWidth * 0.72, size, 500) / 0.72,
      ...(chart.kind === 'bars'
        ? [
            fittedWidth(triText(locale, 'pace'), columnWidth * 0.25, size * 0.65) / 0.25,
            ...points.map(point =>
              Math.max(
                fittedWidth(
                  triLabel(locale, point.label),
                  columnWidth * (zones ? 0.08 : 0.12),
                  labelSize,
                ) / (zones ? 0.08 : 0.12),
                fittedWidth(
                  zones ? (point.range ?? '—') : point.value,
                  columnWidth * (zones ? 0.28 : 0.22),
                  labelSize,
                  400,
                  fonts.mono,
                ) / (zones ? 0.28 : 0.22),
              ),
            ),
            ...(zones
              ? [fittedWidth('100.0%', width * 0.12, labelSize, 400, fonts.mono) / 0.12]
              : []),
          ]
        : [
            fittedWidth(chart.points.at(-1)!.label, width * 0.15, size * 0.75, 400, fonts.mono) /
              0.15,
          ]),
    )
    const minWidth = Math.min(
      width,
      Math.max(
        columnMinimum * columns + gap * (columns - 1),
        chart.note && chart.key !== 'cyclingPower'
          ? fittedWidth(chart.note, width, size * 0.7, 400, fonts.mono)
          : 0,
      ),
    )
    const minHeight = Math.min(
      height,
      chart.kind === 'bars' ? size * 3 + rows * labelSize * 1.5 : size * 4,
    )
    const typography = { width, columnWidth, columns, size, labelSize }
    options.renderers.set(id, {
      x,
      y,
      width,
      height,
      minWidth: minWidth + 6,
      minHeight: minHeight + 6,
      render: (targetWidth, targetHeight) => {
        const density = Math.min(scale, 3840 / Math.max(targetWidth, targetHeight))
        const svg = node('svg', {
          xmlns: 'http://www.w3.org/2000/svg',
          width: Math.ceil(targetWidth * density),
          height: Math.ceil(targetHeight * density),
          viewBox: `0 0 ${targetWidth} ${targetHeight}`,
        })
        paintTrainingChart(
          svg,
          chart,
          3,
          3,
          targetWidth - 6,
          targetHeight - 6,
          ink,
          fonts,
          density,
          locale,
          { typography, color },
        )
        return svgImage(svg.outerHTML)
      },
    })
  }
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
  const charts = activity.charts.filter(chart => model.show[chart.key])
  const chartHeight = charts.length ? (height * 0.28) / charts.length : 0
  const contentHeight = height - chartHeight * charts.length
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
  charts.forEach((chart, index) =>
    paintTrainingChart(
      svgLayer(parent, `chart/${chart.key}`, chart.title),
      chart,
      x,
      y + contentHeight + index * chartHeight,
      width,
      chartHeight,
      ink,
      fonts,
      scale,
      model.locale ?? cardLocale(),
      { renderers },
    ),
  )
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
  const columns = blocks.length > 1 ? 2 : 1
  const rows = Math.max(1, Math.ceil(blocks.length / columns))
  const gap = spec.gap * 0.5
  const width = (spec.width - pl - pr - gap * (columns - 1)) / columns
  const height = (spec.height - pt - pb - gap * (rows - 1)) / rows
  blocks.forEach((block, index) => {
    const group = node('g', { 'data-block': block.id })
    svg.append(group)
    const display = cardBlockDisplay(block, model.show)
    const locale = model.locale ?? cardLocale()
    const x = pl + (index % columns) * (width + gap)
    const y = pt + Math.floor(index / columns) * (height + gap)
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
