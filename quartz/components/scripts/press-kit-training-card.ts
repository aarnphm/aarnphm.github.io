import type { TrainingCardBlock } from './press-kit-day'
import {
  FORMATS,
  metricShown,
  type CardModel,
  type FormatKey,
  type Palette,
  type Fonts,
  type Frame,
} from './press-kit-card'
import { cardRoute, routeSvg } from './press-kit-route'
import { stickerCardSvg, paintTrainingChart } from './press-kit-sticker'
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
): string {
  if (model.overlay && model.overlay !== 'card')
    return stickerCardSvg(model, format, palette, fonts)
  const { text, lines, fitText, fitValue, titleRow } = createSvgText(fonts)
  const spec = FORMATS[format]
  const [, pr, , pl] = spec.pad
  const width = spec.width - pl - pr
  const blocks = model.day?.blocks ?? activityBlock(model)
  const single = blocks.length === 1 && blocks[0].kind === 'activity'
  const svg = node('svg', {
    xmlns: 'http://www.w3.org/2000/svg',
    width: spec.width,
    height: spec.height,
    viewBox: `0 0 ${spec.width} ${spec.height}`,
    role: 'img',
  })
  const title = node('title')
  title.textContent = runsToPlain(model.title)
  svg.append(title)
  if (!model.show.transparent)
    svg.append(node('rect', { width: spec.width, height: spec.height, fill: palette.bg }))
  const top = frame.top
  const bottom = frame.bottom
  if (!blocks.length) {
    text(svg, 'aucun bloc sélectionné', pl, (top + bottom) / 2, spec.meta, palette.muted)
    return svg.outerHTML
  }
  const columns = blocks.length > 1 ? (format === 'landscape' && blocks.length > 4 ? 3 : 2) : 1
  const rows = Math.ceil(blocks.length / columns)
  const gap = spec.gap * 0.65
  const cellWidth = (width - gap * (columns - 1)) / columns
  const cellHeight = (bottom - top - gap * (rows - 1)) / rows
  blocks.forEach((block, index) => {
    const x = pl + (index % columns) * (cellWidth + gap)
    const y = top + Math.floor(index / columns) * (cellHeight + gap)
    const group = node('g', { 'data-block': block.id, 'data-kind': block.kind })
    const clipId = `panel-${index}`
    const clip = node('clipPath', { id: clipId })
    clip.append(node('rect', { x, y, width: cellWidth, height: cellHeight }))
    svg.append(clip)
    group.setAttribute('clip-path', `url(#${clipId})`)
    const link = node('a', { href: block.url })
    link.append(group)
    svg.append(link)
    const metricCount =
      block.activities[0]?.stats.filter(stat => metricShown(model, block.activities[0], stat))
        .length ?? 0
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
    if (model.show.title) {
      cursor +=
        titleRow(
          group,
          block.title,
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
      if (icon) group.append(icon)
      cursor += size * 1.8
    }
    if (block.kind === 'memo') {
      text(group, 'écouter le mémo ↗', x, cursor + size, size, palette.ink)
      group.append(
        node('rect', { x, y, width: cellWidth, height: cellHeight, fill: 'transparent' }),
      )
      return
    }
    if (block.kind === 'entry') {
      const prose = block.text ?? ''
      let proseSize = size
      let proseLines = lines(prose, cellWidth, proseSize)
      while (proseLines.length * proseSize * 1.3 > y + cellHeight - cursor && proseSize > 6) {
        proseSize *= 0.9
        proseLines = lines(prose, cellWidth, proseSize)
      }
      proseLines.forEach((line, index) =>
        fitText(
          group,
          line,
          x,
          cursor + proseSize * (1 + index * 1.3),
          cellWidth,
          proseSize,
          palette.ink,
        ),
      )
      return
    }
    if (block.kind === 'calculator') {
      const rowHeight = Math.min(
        size * 1.8,
        (y + cellHeight - cursor) / Math.max(1, block.rows.length),
      )
      block.rows.forEach((row, rowIndex) => {
        const baseline = cursor + (rowIndex + 1) * rowHeight
        fitText(group, row.label, x, baseline, cellWidth * 0.3, size * 0.85, palette.muted)
        fitValue(
          group,
          row.values.join(' · '),
          x + cellWidth * 0.32,
          baseline,
          cellWidth * 0.68,
          size,
          palette.ink,
        )
      })
      return
    }
    if (!single && block.kind === 'activity' && activity && model.show.date) {
      text(
        group,
        `${cardDate(activity.date)} · ${activity.sport}`,
        x,
        cursor + size * 0.8,
        size * 0.75,
        palette.muted,
        400,
        fonts.mono,
      )
      cursor += size * 1.4
    }
    const stats = activity?.stats.filter(stat => metricShown(model, activity, stat)) ?? []
    const charts =
      block.kind === 'activity'
        ? (activity?.charts.filter(chart => model.show[chart.key]) ?? [])
        : []
    const comparisonRows = block.rows.length
      ? block.rows.filter(row => row.key && model.show[row.key])
      : stats.map(stat => ({
          label: stat.label,
          values: block.activities.map(
            item => item.stats.find(value => value.key === stat.key)?.value ?? '—',
          ),
        }))
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
        ? size * (block.activities.length * 1.4 + 0.5 + comparisonRows.length * 1.5)
        : statRowHeights.reduce((sum, height) => sum + height, 0)
    const route = model.show.map
      ? cardRoute(block.activities.flatMap(item => item.route))
      : undefined
    const available = y + cellHeight - cursor - statsHeight - size
    const visualHeight = Math.max(0, Math.min((cellWidth * 9) / 16, available))
    const visuals = (route ? 1 : 0) + (block.graphic ? 1 : 0) + charts.length
    const visualPart = visuals ? visualHeight / visuals : 0
    if (route && visualPart > size * 2) {
      graphic(group, routeSvg(route, palette.ink, palette.bg), x, cursor, cellWidth, visualPart)
      cursor += visualPart
    }
    if (block.graphic && visualPart > size * 2) {
      graphic(group, block.graphic.svg, x, cursor, cellWidth, visualPart)
      cursor += visualPart
    }
    for (const chart of charts) {
      if (visualPart <= size * 2) continue
      paintTrainingChart(group, chart, x, cursor, cellWidth, visualPart, palette.ink, fonts)
      cursor += visualPart
    }
    if (visualPart > size * 2) cursor += size * 0.6
    if (block.kind === 'comparison') {
      const names = block.activities.map(item =>
        model.show.date ? `${cardDate(item.date)} · ${item.title}` : item.title,
      )
      names.forEach((name, rowIndex) =>
        fitText(
          group,
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
        fitValue(
          group,
          `${row.label} · ${row.values.join(' / ')}`,
          x,
          cursor + size * (rowIndex + 1) * 1.5,
          cellWidth,
          size,
          palette.ink,
        )
      })
    } else {
      stats.forEach((_, statIndex) => {
        const sx = x + ((statIndex % statColumns) * cellWidth) / statColumns
        const sy =
          cursor +
          statRowHeights
            .slice(0, Math.floor(statIndex / 3))
            .reduce((sum, height) => sum + height, 0)
        const layout = statLayouts[statIndex]
        layout.labels.forEach((line, lineIndex) =>
          fitText(
            group,
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
            group,
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
  return svg.outerHTML
}

export function trainingCardImage(svg: string): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }))
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => {
      URL.revokeObjectURL(url)
      resolve(image)
    }
    image.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('Unable to render training card'))
    }
    image.src = url
  })
}
