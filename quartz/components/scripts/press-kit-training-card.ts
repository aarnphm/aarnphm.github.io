import { triLabel, triText } from '../../util/triathlon-i18n'
import {
  FORMATS,
  cardResolution,
  metricShown,
  type CardModel,
  type FormatKey,
  type Palette,
  type Fonts,
  type Frame,
} from './press-kit-card'
import { paintTrainingChart, trainingCharts, trainingChartWeight } from './press-kit-chart'
import { cardBlockDisplay, type TrainingCardBlock } from './press-kit-day'
import { cardLocale } from './press-kit-i18n'
import { svgLayer, type LayerRenderer } from './press-kit-layers'
import { cardRoute, routeSvg } from './press-kit-route'
import { stickerCardSvg } from './press-kit-sticker'
import { node, createSvgText, graphic, sportIcon } from './press-kit-svg'
import { cardDate, runsToPlain } from './press-kit-text'
function activityBlock(model: CardModel): TrainingCardBlock[] {
  return model.activity
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
    : []
}

export function trainingCardSvg(
  model: CardModel,
  format: FormatKey,
  palette: Palette,
  fonts: Fonts,
  frame: Frame,
  renderers?: Map<string, LayerRenderer>,
): string {
  if (model.overlay && model.overlay !== 'card')
    return stickerCardSvg(model, format, palette, fonts, renderers)
  const resolution = cardResolution(format)
  const { text, lines, fitText, fitValue, metricRow, titleRow } = createSvgText(
    fonts,
    resolution.scale,
  )
  const spec = FORMATS[format]
  const [, pr, , pl] = spec.pad
  const width = spec.width - pl - pr
  const blocks = model.day?.blocks ?? activityBlock(model)
  const single = blocks.length === 1 && blocks[0].kind === 'activity'
  const svg = node('svg', {
    xmlns: 'http://www.w3.org/2000/svg',
    width: resolution.width,
    height: resolution.height,
    viewBox: `0 0 ${spec.width} ${spec.height}`,
    role: 'img',
  })
  const title = node('title')
  title.textContent = runsToPlain(model.title)
  svg.append(title)
  if (!model.show.transparent)
    svg.append(node('rect', { width: spec.width, height: spec.height, fill: palette.bg }))
  let top = frame.top
  if (model.day && model.customTitle && model.show.title)
    top +=
      titleRow(
        svgLayer(svg, 'title', model.customTitle),
        model.customTitle,
        undefined,
        pl,
        top,
        width,
        spec.meta * 1.6,
        palette.ink,
      ) +
      spec.gap * 0.5
  const bottom = frame.bottom
  if (!blocks.length) {
    text(
      svgLayer(svg, 'empty', 'aucun bloc sélectionné'),
      (model.locale ?? cardLocale()) === 'fr' ? 'aucun bloc sélectionné' : 'no blocks selected',
      pl,
      (top + bottom) / 2,
      spec.meta,
      palette.muted,
    )
    return svg.outerHTML
  }
  const columns = blocks.length > 1 ? (format === 'landscape' && blocks.length > 4 ? 3 : 2) : 1
  const rows = Math.ceil(blocks.length / columns)
  const gap = spec.gap * 0.65
  const cellWidth = (width - gap * (columns - 1)) / columns
  const cellHeight = (bottom - top - gap * (rows - 1)) / rows
  const centered: { group: SVGElement; row: number; height: number }[] = []
  blocks.forEach((block, index) => {
    const display = cardBlockDisplay(block, model.show)
    const locale = model.locale ?? cardLocale()
    const x = pl + (index % columns) * (cellWidth + gap)
    const y = top + Math.floor(index / columns) * (cellHeight + gap)
    const group = node('g', { 'data-block': block.id, 'data-kind': block.kind })
    const clipId = `panel-${index}`
    const clip = node('clipPath', { id: clipId })
    clip.append(node('rect', { x, y, width: cellWidth, height: cellHeight }))
    svg.append(clip)
    const panel = node('g', { 'clip-path': `url(#${clipId})` })
    panel.append(group)
    const link = node('a', { href: block.url })
    link.append(panel)
    svg.append(link)
    const metricCount = display.metrics
      ? (block.activities[0]?.stats.filter(stat => metricShown(model, block.activities[0], stat))
          .length ?? 0)
      : 0
    const size = Math.min(
      spec.meta * (single ? 1.25 : 0.95),
      cellHeight /
        (block.kind === 'calculator'
          ? 13
          : block.kind === 'comparison'
            ? 10 + block.activities.length * 1.4 + block.rows.length * 1.5
            : Math.max(11, Math.ceil(metricCount / 3) * 4.2 + 4)),
    )
    let cursor = y
    const activity = block.activities[0]
    if (display.title) {
      cursor +=
        titleRow(
          svgLayer(group, 'title', block.title),
          block.kind === 'comparison' ? triText(locale, 'comparison') : block.title,
          activity?.sport,
          x,
          cursor,
          cellWidth,
          size * (single ? 1.6 : 1.12),
          palette.ink,
        ) +
        size * 0.8
    } else {
      const icon = activity
        ? sportIcon(activity.sport, x, cursor, size * 1.4, palette.ink)
        : undefined
      if (icon) svgLayer(group, 'title', block.title).append(icon)
      if (icon) cursor += size * 1.8
    }
    const date = block.date ?? (activity ? cardDate(activity.date, locale) : undefined)
    if (display.date && date) {
      text(
        svgLayer(group, 'date', 'date'),
        date,
        x,
        cursor + size * 0.8,
        size * 0.75,
        palette.muted,
        400,
        fonts.mono,
      )
      cursor += size * 1.4
    }
    if (block.kind === 'memo') {
      text(
        svgLayer(group, 'memo', 'mémo'),
        locale === 'fr' ? 'écouter le mémo ↗' : 'listen to memo ↗',
        x,
        cursor + size,
        size,
        palette.ink,
      )
      return
    }
    if (block.kind === 'entry') {
      const prose = block.text ?? ''
      const proseBottom = y + cellHeight - (display.metrics && block.reading ? size * 1.8 : 0)
      let proseSize = size
      let proseLines = lines(prose, cellWidth, proseSize)
      while (proseLines.length * proseSize * 1.3 > proseBottom - cursor && proseSize > 6) {
        proseSize *= 0.9
        proseLines = lines(prose, cellWidth, proseSize)
      }
      const proseLayer = svgLayer(group, 'text', 'texte')
      proseLines.forEach((line, index) =>
        fitText(
          proseLayer,
          line,
          x,
          cursor + proseSize * (1 + index * 1.3),
          cellWidth,
          proseSize,
          palette.ink,
        ),
      )
      if (display.metrics && block.reading)
        fitValue(
          svgLayer(group, 'reading', 'temps de lecture'),
          block.reading,
          x,
          y + cellHeight - size * 0.25,
          cellWidth,
          size * 0.75,
          palette.muted,
        )
      return
    }
    if (block.kind === 'calculator') {
      const rowHeight = Math.min(
        size * 1.8,
        (y + cellHeight - cursor) / Math.max(1, block.rows.length),
      )
      const visibleRows = display.metrics ? block.rows : []
      visibleRows.forEach((row, rowIndex) => {
        const baseline = cursor + (rowIndex + 1) * rowHeight
        metricRow(
          svgLayer(group, `row/${row.key ?? rowIndex}`, triLabel(locale, row.label)),
          triLabel(locale, row.label),
          row.values.join(' · '),
          x,
          baseline,
          cellWidth,
          size,
          palette.ink,
        )
      })
      return
    }
    const stats = display.metrics
      ? (activity?.stats.filter(stat => metricShown(model, activity, stat)) ?? [])
      : []
    const charts = block.kind === 'activity' ? trainingCharts(activity, model) : []
    const comparisonRows = !display.metrics
      ? []
      : block.rows.length
        ? block.rows.filter(row => row.key && model.show[row.key])
        : stats.map(stat => ({
            key: stat.key,
            label: stat.label,
            values: block.activities.map(
              item => item.stats.find(value => value.key === stat.key)?.value ?? '—',
            ),
          }))
    const comparisonNames = block.activities
      .map(item =>
        [display.date ? cardDate(item.date, locale) : '', display.title ? item.title : '']
          .filter(Boolean)
          .join(' · '),
      )
      .filter(Boolean)
    const statColumns = Math.min(3, stats.length)
    const statWidth = statColumns ? cellWidth / statColumns - size * 0.3 : 0
    const statLayouts = stats.map(stat => ({
      labels: lines(stat.label, statWidth, size * 0.75),
      values: lines(stat.value, statWidth, size * 1.2, 400, fonts.mono),
    }))
    const statRowHeights = Array.from({ length: Math.ceil(stats.length / 3) }, (_, row) =>
      Math.max(
        ...statLayouts
          .slice(row * 3, row * 3 + 3)
          .map(layout => size * (layout.labels.length * 0.95 + layout.values.length * 1.4 + 0.65)),
      ),
    )
    const statsHeight =
      block.kind === 'comparison'
        ? size * (comparisonNames.length * 1.4 + 0.5 + comparisonRows.length * 1.5)
        : statRowHeights.reduce((sum, height) => sum + height, 0)
    const route = model.show.map
      ? cardRoute(block.activities.flatMap(item => item.route))
      : undefined
    const available = y + cellHeight - cursor - statsHeight - size
    const chartWeight = charts.reduce((total, chart) => total + trainingChartWeight(chart), 0)
    const visualHeight = Math.max(
      0,
      Math.min(((cellWidth * 9) / 16) * Math.max(1, chartWeight), available),
    )
    const visualWeights = [
      ...(route ? [1] : []),
      ...(block.graphic ? [1] : []),
      ...charts.map(trainingChartWeight),
    ]
    const visualColumns = format === 'landscape' && visualWeights.length > 1 ? 2 : 1
    const rowWeights = Array.from(
      { length: Math.ceil(visualWeights.length / visualColumns) },
      (_, row) => Math.max(...visualWeights.slice(row * visualColumns, (row + 1) * visualColumns)),
    )
    const totalWeight = rowWeights.reduce((total, weight) => total + weight, 0)
    const visualWidth = (cellWidth - gap * (visualColumns - 1)) / visualColumns
    const visualPart = totalWeight ? visualHeight / totalWeight : 0
    let visualIndex = 0
    const nextVisual = () => {
      const position = {
        x: x + (visualIndex % visualColumns) * (visualWidth + gap),
        y:
          cursor +
          rowWeights
            .slice(0, Math.floor(visualIndex / visualColumns))
            .reduce((total, weight) => total + weight, 0) *
            visualPart,
        height: rowWeights[Math.floor(visualIndex / visualColumns)] * visualPart,
      }
      visualIndex++
      return position
    }
    if (route && visualPart > size * 2) {
      const position = nextVisual()
      graphic(
        svgLayer(group, 'route', 'tracé'),
        routeSvg(route, palette.ink, palette.bg),
        position.x,
        position.y,
        visualWidth,
        position.height,
      )
    }
    if (block.graphic && visualPart > size * 2) {
      const position = nextVisual()
      graphic(
        svgLayer(group, 'graphic', 'graphique'),
        block.graphic.svg,
        position.x,
        position.y,
        visualWidth,
        position.height,
      )
    }
    for (const chart of charts) {
      if (visualPart <= size * 2) continue
      const position = nextVisual()
      paintTrainingChart(
        svgLayer(group, `chart/${chart.key}`, triLabel(locale, chart.title)),
        chart,
        position.x,
        position.y,
        visualWidth,
        position.height,
        palette.ink,
        fonts,
        resolution.scale,
        locale,
        { renderers, logScale: model.powerCurveScale !== 'linear' },
      )
    }
    if (visualPart > size * 2) cursor += visualHeight + size * 0.6
    if (!model.show.map)
      centered.push({ group, row: Math.floor(index / columns), height: cursor - y + statsHeight })
    if (block.kind === 'comparison') {
      const names = comparisonNames
      names.forEach((name, rowIndex) =>
        fitText(
          svgLayer(group, `name/${block.activities[rowIndex]?.url ?? rowIndex}`, name),
          name,
          x,
          cursor + size * (1 + rowIndex * 1.4),
          cellWidth,
          size * 0.8,
          palette.muted,
        ),
      )
      cursor += size * (names.length * 1.4 + 0.5)
      comparisonRows.forEach((row, rowIndex) => {
        metricRow(
          svgLayer(group, `row/${row.key ?? rowIndex}`, triLabel(locale, row.label)),
          triLabel(locale, row.label),
          row.values.join(' / '),
          x,
          cursor + size * (rowIndex + 1) * 1.5,
          cellWidth,
          size,
          palette.ink,
        )
      })
    } else {
      stats.forEach((stat, statIndex) => {
        const metric = svgLayer(group, `stat/${stat.key}`, stat.label)
        const sx = x + ((statIndex % statColumns) * cellWidth) / statColumns
        const sy =
          cursor +
          statRowHeights
            .slice(0, Math.floor(statIndex / 3))
            .reduce((sum, height) => sum + height, 0)
        const layout = statLayouts[statIndex]
        layout.labels.forEach((line, lineIndex) =>
          fitText(
            metric,
            line,
            sx,
            sy + size * (0.85 + lineIndex * 0.95),
            statWidth,
            size * 0.75,
            palette.muted,
          ),
        )
        const valueTop = sy + size * (layout.labels.length * 0.95 + 1.3)
        layout.values.forEach((line, lineIndex) =>
          fitValue(
            metric,
            line,
            sx,
            valueTop + lineIndex * size * 1.4,
            statWidth,
            size * 1.2,
            palette.ink,
          ),
        )
      })
    }
  })
  for (const panel of centered) {
    const height = Math.max(
      ...centered.filter(item => item.row === panel.row).map(item => item.height),
    )
    panel.group.setAttribute('transform', `translate(0 ${Math.max(0, (cellHeight - height) / 2)})`)
  }
  return svg.outerHTML
}
