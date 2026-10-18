import { scaleLinear, scaleLog } from 'd3'
import type { ActivityCardChart, ActivityCardData } from '../../util/press-kit-activity'
import type { Locale } from '../../util/triathlon-presentation'
import type { CardModel, Fonts } from './press-kit-card'
import type { LayerRenderer } from './press-kit-layers'
import { triLabel, triText } from '../../util/triathlon-i18n'
import { cardLocale } from './press-kit-i18n'
import { createSvgText, node, svgImage } from './press-kit-svg'

export function trainingCharts(
  activity: ActivityCardData | undefined,
  model: CardModel,
): ActivityCardChart[] {
  return (activity?.charts ?? []).flatMap(chart => {
    if (!model.show[chart.key]) return []
    if (chart.key !== 'cyclingPower' || !chart.series) return [chart]
    const series = chart.series.find(series => series.seconds === (model.powerWindow ?? 30))
    return series ? [{ ...chart, points: series.points, window: series.seconds }] : []
  })
}

export function trainingChartWeight(chart: ActivityCardChart): number {
  return chart.kind === 'intervals' || chart.key === 'runLaps' ? 1.3 : 1
}

interface ChartTypography {
  width: number
  size: number
  labelSize: number
}
interface ChartRenderOptions {
  renderers?: Map<string, LayerRenderer>
  typography?: ChartTypography
  color?: string
  logScale?: boolean
}

const clock = (seconds: number): string => {
  const rounded = Math.round(seconds)
  const hours = Math.floor(rounded / 3600)
  const minutes = Math.floor((rounded % 3600) / 60)
  const rest = String(rounded % 60).padStart(2, '0')
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${rest}` : `${minutes}:${rest}`
}
const duration = (seconds: number): string => (seconds < 60 ? `${seconds} s` : clock(seconds))

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
  const valid = chart.points.filter(
    point => point.y != null && (chart.key !== 'runLaps' || point.y > 0),
  )
  if (!valid.length || width <= 0 || height <= 0) return
  const { text, fitText, fitValue, measure } = createSvgText(fonts, scale)
  const bars = chart.kind === 'bars'
  const zones = chart.key === 'runPace'
  const runLaps = chart.key === 'runLaps'
  const laps = chart.kind === 'intervals'
  const curve = chart.key === 'powerCurve'
  const logarithmic = curve && options?.logScale !== false
  const size = options?.typography?.size ?? Math.min(width / 30, height / 10)
  const labelSize =
    options?.typography?.labelSize ??
    (bars ? Math.min(size * 0.75, (height / (valid.length + 5)) * 0.5) : size * 0.7)
  const referenceWidth = options?.typography?.width ?? width
  const appearance =
    document.querySelector('.tri-training-zone-grid--pace, .tri-day-embed, .triathlon') ??
    document.documentElement
  const color =
    options?.color ??
    (runLaps || laps
      ? '#fc4c02'
      : getComputedStyle(appearance)
          .getPropertyValue(zones ? '--tri-zone-pace' : '--tri-zone-power')
          .trim() || ink)
  const group = node('g', {
    'data-chart': chart.key,
    role: 'img',
    'aria-label': `${triLabel(locale, chart.title)} · ${chart.axis}`,
  })
  parent.append(group)
  const title =
    (runLaps
      ? locale === 'fr'
        ? 'temps intermédiaires'
        : 'splits'
      : triLabel(locale, chart.title)) +
    (chart.window ? ` · ${chart.window === 30 ? '30 s' : '5 min'}` : '')
  fitText(group, title, x, y + size, referenceWidth * 0.78, size, ink, 500)
  const label = (value: string, px: number, py: number, anchor = 'start', fontSize = labelSize) => {
    const holder = node('g', { 'text-anchor': anchor, 'aria-label': value })
    group.append(holder)
    text(holder, value, px, py, fontSize, ink, 400, fonts.mono)
  }
  const line = (x1: number, y1: number, x2: number, y2: number, grid = false) =>
    group.append(
      node('line', {
        x1,
        y1,
        x2,
        y2,
        stroke: ink,
        'stroke-width': grid ? 0.7 : 1,
        opacity: grid ? 0.15 : 0.45,
        ...(grid ? { 'stroke-dasharray': '2 4', 'data-grid': '' } : {}),
      }),
    )
  const labelWidth = (value: string) =>
    measure(value, labelSize, 400, fonts.mono)?.width ?? value.length * labelSize * 0.6
  const maximum = Math.max(1, ...valid.map(point => point.y ?? 0))
  const axis = scaleLinear().domain([0, maximum]).nice(4)
  const ceiling = axis.domain()[1]
  const top = y + size * 2
  const bottom = y + height - size * ((bars ? 2.9 : 1.7) + (chart.note ? 1 : 0))
  const tickText = (value: number) =>
    runLaps ? clock(value) : value.toLocaleString(locale, { maximumFractionDigits: 1 })

  // Keep endpoint labels and omit intermediate labels that no longer fit after resizing.
  const spacedTicks = (
    values: number[],
    project: (value: number) => number,
    format: (value: number) => string,
  ) => {
    if (values.length < 3) return values
    const last = values.at(-1)!
    let edge = project(values[0]) + labelWidth(format(values[0]))
    const end = project(last) - labelWidth(format(last))
    return values.filter((value, index) => {
      if (index === 0 || value === last) return true
      const half = labelWidth(format(value)) / 2
      const center = project(value)
      if (center - half < edge + labelSize || center + half > end - labelSize) return false
      edge = center + half
      return true
    })
  }

  if (runLaps) {
    const imperial = chart.axis === '/mi'
    const headers = {
      distance: imperial ? 'Mi' : 'Km',
      pace: locale === 'fr' ? 'Allure' : 'Pace',
      elevation: `${locale === 'fr' ? 'Élév.' : 'Elev.'} (${imperial ? 'ft' : 'm'})`,
      heartRate: locale === 'fr' ? 'FC' : 'HR',
    }
    const rows = valid.map(point => {
      const distance = (point.split?.distanceKm ?? 0) / (imperial ? 1.609344 : 1)
      const elevation = point.split?.elevationDifferenceM
      return {
        point,
        distance:
          distance > 0 && distance < 0.95
            ? distance.toLocaleString(locale, { maximumFractionDigits: distance < 0.1 ? 2 : 1 })
            : String(point.x),
        pace: clock(point.y ?? 0),
        elevation:
          elevation == null ? '—' : String(Math.round(elevation / (imperial ? 0.3048 : 1))),
        heartRate: point.split?.heartRate == null ? '—' : String(Math.round(point.split.heartRate)),
      }
    })
    const columnWidth = (header: string, values: string[], fraction: number) =>
      Math.max(
        width * fraction,
        Math.max(labelWidth(header), ...values.map(labelWidth)) + labelSize * 1.2,
      )
    const distanceWidth = columnWidth(
      headers.distance,
      rows.map(row => row.distance),
      0.075,
    )
    const paceWidth = columnWidth(
      headers.pace,
      rows.map(row => row.pace),
      0.125,
    )
    const elevationWidth = columnWidth(
      headers.elevation,
      rows.map(row => row.elevation),
      0.13,
    )
    const heartRateWidth = columnWidth(
      headers.heartRate,
      rows.map(row => row.heartRate),
      0.07,
    )
    const paceX = x + distanceWidth
    const left = paceX + paceWidth
    const right = x + width - elevationWidth - heartRateWidth
    const elevationX = x + width - heartRateWidth
    const headerY = y + size * 2.6
    const ruleY = headerY + labelSize * 0.8
    const rowsTop = ruleY + labelSize * 0.6
    const rowsBottom = y + height - labelSize * 0.3
    const rowHeight = (rowsBottom - rowsTop) / rows.length
    const speeds = rows.map(row => 1 / (row.point.y ?? 1))
    const minimum = Math.min(...speeds)
    const maximum = Math.max(...speeds)
    // A local speed range makes the pace differences visible, with faster splits extending farther.
    const barWidth = scaleLinear()
      .domain([minimum, maximum])
      .range([Math.max(0, right - left) * 0.25, Math.max(0, right - left)])
    label(headers.distance, x, headerY)
    label(headers.pace, paceX, headerY)
    label(headers.elevation, elevationX, headerY, 'end')
    label(headers.heartRate, x + width, headerY, 'end')
    line(x, ruleY, x + width, ruleY)
    rows.forEach((row, index) => {
      const middle = rowsTop + (index + 0.5) * rowHeight
      const baseline = middle + labelSize * 0.35
      const holder = node('g', {
        'aria-label': `${row.distance} ${headers.distance}, ${row.pace} ${chart.axis}, ${row.elevation} ${imperial ? 'ft' : 'm'}, ${row.heartRate} bpm`,
      })
      group.append(holder)
      const barHeight = rowHeight * 0.78
      holder.append(
        node('rect', {
          x: left,
          y: middle - barHeight / 2,
          width: barWidth(speeds[index]),
          height: barHeight,
          fill: color,
        }),
      )
      label(row.distance, x, baseline)
      label(row.pace, paceX, baseline)
      label(row.elevation, elevationX, baseline, 'end')
      label(row.heartRate, x + width, baseline, 'end')
    })
  } else if (bars) {
    const left = x + width * (zones ? 0.1 : 0.15)
    const right = x + width * (zones ? 0.58 : 0.73)
    const rowHeight = (bottom - top) / valid.length
    const px = axis.range([left, right])
    const ticks = spacedTicks(
      axis.ticks(Math.max(2, Math.min(4, (right - left) / (labelSize * 5)))),
      px,
      tickText,
    )
    for (const tick of ticks) {
      line(px(tick), top, px(tick), bottom, true)
      label(
        tickText(tick),
        px(tick),
        bottom + size * 0.9,
        tick === 0 ? 'start' : tick === ceiling ? 'end' : 'middle',
      )
    }
    line(left, top, left, bottom)
    line(left, bottom, right, bottom)
    label(chart.axis, (left + right) / 2, bottom + size * 2, 'middle')
    if (zones) label(locale === 'fr' ? 'part' : 'share', x + width * 0.72, y + size, 'end')
    label(triText(locale, 'pace'), x + width, y + size, 'end')
    const total = valid.reduce((sum, point) => sum + (point.y ?? 0), 0)
    valid.forEach((point, index) => {
      const rowTop = top + index * rowHeight
      const middle = rowTop + rowHeight / 2
      line(x, rowTop, x + width, rowTop, true)
      const row = node('g', {
        'aria-label': `${point.label}, ${point.value}, ${point.range ?? ''}`,
      })
      group.append(row)
      const barHeight = Math.min(rowHeight * 0.45, labelSize)
      row.append(
        node('rect', {
          x: left,
          y: middle - barHeight / 2,
          width: Math.max(0, px(point.y ?? 0) - left),
          height: barHeight,
          fill: color,
          opacity: zones ? 0.4 + index * 0.1 : 1,
        }),
      )
      fitText(
        row,
        triLabel(locale, point.label),
        x,
        middle + labelSize * 0.35,
        referenceWidth * (zones ? 0.08 : 0.13),
        labelSize,
        ink,
      )
      if (zones)
        label(
          `${(total ? ((point.y ?? 0) / total) * 100 : 0).toFixed(1)}%`,
          x + width * 0.72,
          middle + labelSize * 0.35,
          'end',
        )
      const value = node('g', { 'text-anchor': 'end' })
      row.append(value)
      fitValue(
        value,
        zones ? (point.range ?? '—') : point.value,
        x + width,
        middle + labelSize * 0.35,
        referenceWidth * 0.25,
        labelSize,
        ink,
      )
    })
  } else {
    const ticks = axis.ticks(Math.max(2, Math.min(5, (bottom - top) / (labelSize * 2.5))))
    const left = x + Math.max(...ticks.map(tick => labelWidth(tickText(tick)))) + labelSize * 0.8
    const right = x + width - labelSize * 0.5
    const py = axis.range([bottom, top])
    const first = chart.points[0].x
    const last = Math.max(first + 1, ...chart.points.map(point => point.end ?? point.x))
    const px = logarithmic
      ? scaleLog()
          .domain([Math.max(1, first), last])
          .range([left, right])
      : scaleLinear().domain([0, last]).range([left, right])
    for (const tick of ticks) {
      line(left, py(tick), right, py(tick), true)
      label(tickText(tick), left - labelSize * 0.6, py(tick) + labelSize * 0.35, 'end')
    }
    label(chart.axis, x + width, y + size, 'end')
    const candidates = logarithmic
      ? [
          first,
          ...[5, 10, 20, 30, 60, 120, 180, 300, 360, 600, 720, 1200, 1800, 3600, 7200].filter(
            value => value > first && value < last,
          ),
          last,
        ]
      : [
          0,
          ...scaleLinear()
            .domain([0, last])
            .ticks(5)
            .filter(value => value > 0 && value < last),
          last,
        ]
    const xTicks = spacedTicks(candidates, px, duration)
    for (const tick of xTicks) {
      line(px(tick), top, px(tick), bottom, true)
      label(
        duration(tick),
        px(tick),
        bottom + size * 0.9,
        tick === candidates[0] ? 'start' : tick === last ? 'end' : 'middle',
      )
    }
    line(left, top, left, bottom)
    line(left, bottom, right, bottom)
    if (laps) {
      for (const point of chart.points) {
        if (point.end == null) continue
        const start = px(point.x)
        const end = px(point.end)
        const span = end - start
        const rect = node('rect', {
          x: start + Math.min(1, span / 4),
          y: point.y == null ? top : py(point.y),
          width: Math.max(0, span - Math.min(2, span / 2)),
          height: point.y == null ? bottom - top : bottom - py(point.y),
          fill: point.y == null ? 'none' : color,
          stroke: point.y == null ? ink : 'none',
          'stroke-dasharray': point.y == null ? '2 4' : '',
          opacity: point.y == null ? 0.35 : 0.8,
        })
        const title = node('title')
        title.textContent = point.detail ?? `${point.label}: ${point.value}`
        rect.append(title)
        group.append(rect)
      }
    } else {
      let path = ''
      let connected = false
      for (const point of chart.points) {
        if (point.y == null) {
          connected = false
          continue
        }
        path += `${connected ? 'L' : 'M'}${px(point.x)} ${py(point.y)} `
        connected = true
      }
      group.append(
        node('path', {
          d: path,
          fill: 'none',
          stroke: color,
          'stroke-width': 2,
          'stroke-linejoin': 'round',
        }),
      )
    }
  }
  if (chart.note)
    fitValue(group, chart.note, x, y + height - size * 0.15, referenceWidth, size * 0.6, ink)
  if (options?.renderers) {
    const id = parent.getAttribute('data-card-layer')
    if (!id) return
    const lapChart = runLaps || laps
    const minimumLabel = Math.min(labelSize, 12 / scale)
    const minimumSize = minimumLabel / (bars ? 0.75 : 0.7)
    const minimumHeight = bars
      ? Math.max(minimumSize * 12, minimumLabel * (valid.length + 5) * 2)
      : minimumSize * 10
    options.renderers.set(id, {
      x,
      y,
      width,
      height,
      minWidth: Math.min(width, lapChart ? minimumSize * 30 : size * (bars ? 24 : 18)) + 6,
      minHeight:
        Math.min(
          height,
          lapChart ? minimumHeight : bars ? size * 5 + valid.length * labelSize * 1.8 : size * 8,
        ) + 6,
      render: (targetWidth, targetHeight) => {
        const contentWidth = targetWidth - 6
        const contentHeight = targetHeight - 6
        const targetSize = lapChart
          ? Math.min(size, contentWidth / 30, contentHeight / (bars ? 12 : 10))
          : size
        const targetLabel = lapChart
          ? Math.min(
              labelSize,
              bars
                ? Math.min(targetSize * 0.75, (contentHeight / (valid.length + 5)) * 0.5)
                : targetSize * 0.7,
            )
          : labelSize
        const typography = {
          width: lapChart ? Math.min(referenceWidth, contentWidth) : referenceWidth,
          size: targetSize,
          labelSize: targetLabel,
        }
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
          contentWidth,
          contentHeight,
          ink,
          fonts,
          density,
          locale,
          { typography, color, logScale: options.logScale },
        )
        return svgImage(svg.outerHTML)
      },
    })
  }
}
