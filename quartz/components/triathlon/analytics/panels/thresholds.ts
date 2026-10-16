import type {
  Analytics,
  LactateThresholdHeartRate,
  SportTrend,
  SportTrendForecastPoint,
} from '../../../../plugins/stores/analytics'
import type { GarminLactateThresholdValue } from '../../../../plugins/stores/garmin'
import type { Sport } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import type { RaceLegSplit } from '../shared'
import { clock } from '../../../../util/triathlon-card'
import { KM_TO_MI } from '../../../../util/triathlon-card'
import { el, svg } from '../../runtime/dom'
import { anaTitle } from '../shared'
import { buildIconLeg } from '../shared'
import { bySport } from '../shared'
import { clampN } from '../shared'
import { fmtSpeedKmh } from '../shared'
import { hms } from '../shared'
import { markGloss } from '../shared'
import { markGlossDefinition } from '../shared'
import { RACE_LABEL } from '../shared'
import { raceLegTip } from '../shared'
import { signedFixed } from '../shared'
import { speedFromKmh } from '../shared'
import { thLabel } from '../shared'

export const renderLegSegments = (
  formatter: TriathlonFormatter,
  track: HTMLElement,
  legs: RaceLegSplit[],
): void => {
  for (const old of track.querySelectorAll('.tri-rdy-leg')) old.remove()
  const legTotalS = legs.reduce((sum, leg) => sum + Math.max(0, leg.splitS), 0)
  if (legTotalS <= 0) return
  let legOffsetS = 0
  for (const leg of legs) {
    const splitS = Math.max(0, leg.splitS)
    if (splitS <= 0) continue
    const hit = el('button', `tri-rdy-leg tri-rdy-leg-${leg.sport}`, undefined, {
      type: 'button',
      'aria-label': raceLegTip(formatter, leg),
      'data-tip': raceLegTip(formatter, leg),
    })
    hit.style.left = `${((legOffsetS / legTotalS) * 100).toFixed(2)}%`
    hit.style.width = `${((splitS / legTotalS) * 100).toFixed(2)}%`
    track.appendChild(hit)
    legOffsetS += splitS
  }
}

export const buildReadiness = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const text = (key: string): string => context.formatter.text(key)
  const block = el('div', 'tri-ana-readiness')
  block.appendChild(anaTitle(context.formatter, 'race readiness', 'score'))
  if (!data.races.length) {
    block.appendChild(el('div', 'tri-ana-empty', '—'))
    return block
  }
  for (const r of data.races) {
    const row = el('div', 'tri-rdy-row')
    row.appendChild(el('span', 'tri-rdy-label', RACE_LABEL[r.distance] ?? r.distance))
    const track = el('div', 'tri-rdy-bar')
    const score = clampN(r.score, 0, 100)
    const fill = el('div', 'tri-rdy-fill')
    fill.style.width = `${Math.max(2, score)}%`
    const bw = Math.min(40, r.bandPct)
    const band = el('div', 'tri-rdy-band')
    band.style.left = `${clampN(score - bw / 2, 0, 100 - bw)}%`
    band.style.width = `${bw}%`
    track.append(fill, band)
    renderLegSegments(context.formatter, track, r.legs)
    row.appendChild(track)
    const meta = el('span', 'tri-rdy-meta')
    meta.append(
      markGloss(el('span', `tri-rdy-bind tri-leg-${r.bindingLeg}`, text(r.bindingLeg)), 'binding'),
      markGloss(el('span', 'tri-rdy-time', hms(r.predictedTotalS)), 'predtime'),
    )
    const gain = r.currentTotalS - r.predictedTotalS
    const showGain = r.projected && Math.abs(gain) >= 1
    meta.appendChild(
      el(
        'span',
        `tri-rdy-delta${showGain ? ` tri-dir-${gain > 0 ? 'up' : 'down'}` : ''}`,
        showGain ? `${gain > 0 ? '−' : '+'}${hms(Math.abs(gain))}` : '',
      ),
    )
    const rangeTxt =
      r.predictedFastS < r.predictedSlowS ? `${hms(r.predictedFastS)}–${hms(r.predictedSlowS)}` : ''
    meta.appendChild(
      markGloss(
        el('span', 'tri-rdy-forecast', rangeTxt, {
          title: text('projected finish range, including both transitions'),
        }),
        'predtime',
      ),
    )
    row.appendChild(meta)
    block.appendChild(row)
  }
  return block
}

export const METHOD_WIKI: Record<string, string> = { kalman: 'Kalman filter' }

export const buildMethod = (
  formatter: TriathlonFormatter,
  method: string,
  n: number,
): HTMLElement => {
  const span = el('span', 'tri-ana-k')
  const title = METHOD_WIKI[method]
  if (!title) {
    span.textContent = `${method} · n=${n}`
    return span
  }
  const a = document.createElement('a')
  a.className = 'internal tri-ana-wiki'
  a.href = `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, '_'))}`
  a.target = '_blank'
  a.rel = 'noopener noreferrer'
  a.dataset.wikipediaLang = 'en'
  a.dataset.wikipediaTitle = title
  a.textContent = formatter.text(method)
  span.append(a, ` · n=${n}`)
  return span
}

export const fmtTrendVal = (formatter: TriathlonFormatter, sport: Sport, v: number): string =>
  sport === 'bike'
    ? fmtSpeedKmh(formatter, v, 0)
    : `${clock(sport === 'run' && formatter.presentation.distance === 'imperial' ? v / KM_TO_MI : v)}${sport === 'swim' ? ' /100m' : formatter.presentation.distance === 'imperial' ? ' /mi' : ' /km'}`

export const fmtTrendShort = (formatter: TriathlonFormatter, sport: Sport, v: number): string =>
  sport === 'bike'
    ? String(Math.round(speedFromKmh(formatter, v)))
    : clock(sport === 'run' && formatter.presentation.distance === 'imperial' ? v / KM_TO_MI : v)

export const fmtKm = (presentation: TriathlonPresentation, km: number): string =>
  presentation.distance === 'imperial' ? `${(km * KM_TO_MI).toFixed(1)} mi` : `${km.toFixed(1)} km`

export const fmtSignedKm = (presentation: TriathlonPresentation, km: number): string =>
  presentation.distance === 'imperial'
    ? `${signedFixed(km * KM_TO_MI, 1)} mi`
    : `${signedFixed(km, 1)} km`

export type LactateThresholdEstimate = Analytics['engine']['lactateThreshold']['sports'][number]

export const fmtLactateValue = (
  formatter: TriathlonFormatter,
  estimate: Pick<LactateThresholdEstimate, 'sport' | 'value'>,
): string =>
  estimate.sport === 'bike'
    ? `${Math.round(estimate.value)} W`
    : fmtTrendVal(formatter, estimate.sport, estimate.value)

/** Method, sample and date of one sport's LT2, in display order. */
export const lactateSourceText = (
  formatter: TriathlonFormatter,
  estimate: LactateThresholdEstimate,
): string => {
  const text = (key: string): string => formatter.text(key)
  const anchor = estimate.heartRateAnchor
  switch (estimate.source) {
    case 'heart-rate-anchored':
      return anchor
        ? `${text('pace at LTHR')} ${anchor.heartRateBpm} bpm · ${anchor.windows} ${text('steady windows')} · ${anchor.runs} ${text('runs')} · ${anchor.lookbackDays} ${text('days')}`
        : text('pace at LTHR')
    case 'garmin':
      return `${text('Garmin running estimate')} · ${estimate.date}${estimate.device?.deltaPct != null ? ` · ${signedFixed(estimate.device.deltaPct, 1)}% ${text('vs pace at LTHR')}` : ''}`
    case 'pace-p90':
      return `${text('training pace P90')} · n=${estimate.sampleSize}`
    case 'pace-max':
      return `${text('97% of fastest session')} · n=${estimate.sampleSize}`
    case 'prior':
      return text('population prior')
    case 'critical-power':
      return `${text('critical power')} · ${estimate.sampleSize} ${text('efforts')} · ${estimate.date}`
    case 'critical-power-year':
      return `${text('critical power')} · ${text('calendar year')} · ${estimate.sampleSize} ${text('efforts')} · ${estimate.date}`
    case 'twenty-minute-power':
      return `${text('95% of 20-min power')} · ${estimate.date}`
    case 'declared-ftp':
      return text('declared FTP')
  }
}

const LT_TREND_POINTS = 8

type DatedValue = { date: string; value: number }

const buildPulseIcon = (): HTMLElement => {
  const wrap = el('span', 'tri-ana-ico tri-leg-run')
  const icon = svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
  icon.appendChild(svg('path', { d: 'M3 12h4l2.5-6 4.5 12 2.5-6H21' }))
  wrap.appendChild(icon)
  return wrap
}

const sampleTrendCheckpoints = (
  points: readonly DatedValue[],
  count = LT_TREND_POINTS,
): DatedValue[] => {
  if (points.length <= 2) return [...points]
  const startMs = Date.parse(`${points[0].date}T00:00:00Z`)
  const endMs = Date.parse(`${points[points.length - 1].date}T00:00:00Z`)
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
    return [points[0], points[points.length - 1]]
  }
  const sampled: DatedValue[] = []
  for (let i = 0; i < count; i++) {
    const targetMs = startMs + ((endMs - startMs) * i) / (count - 1)
    let chosen = points[0]
    for (const point of points) {
      if (Date.parse(`${point.date}T00:00:00Z`) <= targetMs + 1) chosen = point
      else break
    }
    sampled.push(chosen)
  }
  return sampled
}

const buildTrendPolylineSvg = (
  points: readonly DatedValue[],
  fallbackValue: number,
  lineClass: string,
  ariaLabel: string,
): SVGElement => {
  const sampled =
    points.length >= 2
      ? sampleTrendCheckpoints(points)
      : [
          { date: '', value: fallbackValue },
          { date: '', value: fallbackValue },
        ]
  const values = sampled.map(point => point.value)
  const low = Math.min(...values)
  const rawSpan = Math.max(...values) - low
  const span = Math.max(1e-6, rawSpan)
  const coordinates = values.map((value, index) => {
    const x = 6 + (index / (values.length - 1)) * 128
    const y = rawSpan < 1e-6 ? 24 : 42 - ((value - low) / span) * 36
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  const chart = svg('svg', {
    class: 'tri-lt-spark',
    viewBox: '0 0 140 48',
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': ariaLabel,
  })
  chart.append(
    svg('polyline', {
      class: `tri-trend-level ${lineClass}`,
      points: coordinates.join(' '),
      fill: 'none',
    }),
  )
  return chart
}

const fmtPaceDeltaSec = (deltaSec: number): string => {
  const rounded = Math.round(deltaSec)
  const sign = rounded > 0 ? '+' : rounded < 0 ? '−' : ''
  const abs = Math.abs(rounded)
  return abs < 60 ? `${sign}${abs}s` : `${sign}${clock(abs)}`
}

const fmtHeartRateDelta = (
  heartRate: LactateThresholdHeartRate,
  points: readonly DatedValue[],
): string => {
  if (points.length < 2) return `0 ${heartRate.unit}`
  const delta = Math.round(points[points.length - 1].value - points[0].value)
  const sign = delta > 0 ? '+' : delta < 0 ? '−' : ''
  return `${sign}${Math.abs(delta)} ${heartRate.unit}`
}

const fmtSportDelta = (
  formatter: TriathlonFormatter,
  estimate: LactateThresholdEstimate,
  points: readonly DatedValue[],
): string => {
  if (points.length < 2) return estimate.sport === 'bike' ? '0 W' : '0s'
  const first = points[0].value
  const last = points[points.length - 1].value
  if (estimate.sport === 'bike') {
    const delta = Math.round(last - first)
    const sign = delta > 0 ? '+' : delta < 0 ? '−' : ''
    return `${sign}${Math.abs(delta)} W`
  }
  const scale =
    estimate.sport === 'run' && formatter.presentation.distance === 'imperial' ? 1 / KM_TO_MI : 1
  return fmtPaceDeltaSec((last - first) * scale)
}

const fmtSeriesDateRange = (
  formatter: TriathlonFormatter,
  points: readonly DatedValue[],
  fallbackDate?: string | null,
): string => {
  if (points.length >= 2) {
    return `${formatter.shortDate(points[0].date)} → ${formatter.shortDate(points[points.length - 1].date)}`
  }
  if (points.length === 1 && points[0].date) return formatter.shortDate(points[0].date)
  return fallbackDate ? formatter.shortDate(fallbackDate) : '—'
}

const lactateHeartRatePoints = (
  heartRate: LactateThresholdHeartRate,
  series: readonly GarminLactateThresholdValue[],
): DatedValue[] => {
  if (series.length >= 2) return [...series]
  if ('date' in heartRate && heartRate.date)
    return [{ date: heartRate.date, value: heartRate.value }]
  return []
}

const lactateSportPoints = (
  estimate: LactateThresholdEstimate,
  trend: SportTrend | undefined,
): DatedValue[] => {
  if (!trend || trend.history.length < 2 || trend.level == null || !(trend.level > 0)) return []
  const scale = estimate.value / trend.level
  return trend.history.map((point: SportTrendForecastPoint) => ({
    date: point.date,
    value: point.value * scale,
  }))
}

const lactateHeartRateHoverText = (
  formatter: TriathlonFormatter,
  heartRate: LactateThresholdHeartRate,
  series: readonly GarminLactateThresholdValue[],
): string => {
  const base =
    heartRate.source === 'garmin'
      ? `${formatter.text('Garmin running estimate')} · ${heartRate.date}`
      : formatter.text('declared heart-rate anchor')
  if (series.length < 2) return base
  const first = series[0]
  return `${base} · ${formatter.shortDate(first.date)} ${Math.round(first.value)} ${heartRate.unit} → ${heartRate.value} ${heartRate.unit}`
}

const lactateSportHoverText = (
  formatter: TriathlonFormatter,
  estimate: LactateThresholdEstimate,
  trend: SportTrend | undefined,
): string => {
  const parts = [lactateSourceText(formatter, estimate)]
  const device = estimate.device
  if (device && !device.accepted) {
    parts.push(
      `${formatter.text('Garmin device estimate')} ${fmtTrendVal(formatter, 'run', device.value)} · ${device.date}${device.deltaPct != null ? ` · ${signedFixed(device.deltaPct, 1)}% · ${formatter.text('outside ±5%, not used')}` : ''}`,
    )
  }
  if (trend && trend.history.length >= 2 && trend.level != null && trend.level > 0) {
    const scale = estimate.value / trend.level
    const first = trend.history[0]
    const startVal = fmtLactateValue(formatter, {
      sport: estimate.sport,
      value: first.value * scale,
    })
    const curVal = fmtLactateValue(formatter, estimate)
    const endForecast = trend.forecast[trend.forecast.length - 1]
    const projVal = endForecast
      ? ` · +${trend.forecast.length}d ${fmtLactateValue(formatter, { sport: estimate.sport, value: endForecast.value * scale })}`
      : ''
    parts.push(`${formatter.shortDate(first.date)} ${startVal} → ${curVal}${projVal}`)
  }
  return parts.join(' · ')
}

const attachGlossHover = <T extends HTMLElement>(
  node: T,
  termLabel: string,
  hoverText: string,
): T => {
  markGlossDefinition(node, hoverText)
  node.dataset.glossTerm = termLabel
  return node
}

const buildLactateSparkContainer = (
  termLabel: string,
  hoverText: string,
  sparkSvg: SVGElement,
): HTMLElement => {
  const wrap = attachGlossHover(el('div', 'tri-lt-trend'), termLabel, hoverText)
  wrap.appendChild(sparkSvg)
  return wrap
}

const buildLactateHeartRateRow = (
  heartRate: LactateThresholdHeartRate,
  series: readonly GarminLactateThresholdValue[],
  context: TriathlonContext,
): HTMLTableRowElement => {
  const { formatter } = context
  const row = document.createElement('tr')
  row.className = 'tri-lt-row'
  row.dataset.sport = 'lthr'
  row.dataset.source = heartRate.source
  const valueText = `${heartRate.value} ${heartRate.unit}`
  const label = `LTHR ${valueText}`
  const hoverText = lactateHeartRateHoverText(formatter, heartRate, series)
  const points = lactateHeartRatePoints(heartRate, series)

  const th = el('th', undefined, undefined, { scope: 'row' })
  const head = attachGlossHover(el('span', 'tri-lt-head'), label, hoverText)
  head.append(buildPulseIcon(), el('span', 'tri-lt-name', 'LTHR'))
  th.appendChild(head)

  const valTd = el('td', 'tri-lt-val')
  const valWrap = el('span', 'tri-lt-val-inner')
  valWrap.appendChild(attachGlossHover(el('span', 'tri-trend-unit', valueText), label, hoverText))
  valTd.appendChild(valWrap)

  const deltaTd = el('td', 'tri-lt-delta', fmtHeartRateDelta(heartRate, points))
  const heartRateDate = 'date' in heartRate ? heartRate.date : undefined
  const datesTd = el('td', 'tri-lt-dates', fmtSeriesDateRange(formatter, points, heartRateDate))
  const chartTd = el('td', 'tri-lt-chart')
  const spark = buildTrendPolylineSvg(
    points,
    heartRate.value,
    'tri-line-run',
    `${label} · ${hoverText}`,
  )
  chartTd.appendChild(buildLactateSparkContainer(label, hoverText, spark))

  row.append(th, valTd, deltaTd, datesTd, chartTd)
  return row
}

const buildLactateThresholdRow = (
  estimate: LactateThresholdEstimate,
  trend: SportTrend | undefined,
  context: TriathlonContext,
): HTMLTableRowElement => {
  const { formatter } = context
  const row = document.createElement('tr')
  row.className = 'tri-lt-row'
  row.dataset.sport = estimate.sport
  row.dataset.source = estimate.source
  const valueText = fmtLactateValue(formatter, estimate)
  const label = `LT2 ${valueText}`
  const hoverText = lactateSportHoverText(formatter, estimate, trend)
  const points = lactateSportPoints(estimate, trend)

  const th = el('th', undefined, undefined, { scope: 'row' })
  const head = attachGlossHover(el('span', 'tri-lt-head'), label, hoverText)
  head.append(buildIconLeg(formatter, estimate.sport), el('span', 'tri-lt-name', 'LT2'))
  th.appendChild(head)

  const valTd = el('td', 'tri-lt-val')
  const valWrap = el('span', 'tri-lt-val-inner')
  valWrap.appendChild(attachGlossHover(el('span', 'tri-trend-unit', valueText), label, hoverText))
  if (estimate.conf) {
    valWrap.appendChild(
      markGloss(el('span', `tri-ana-conf tri-conf-${estimate.conf}`, estimate.conf), 'conf'),
    )
  }
  valTd.appendChild(valWrap)

  const deltaTd = el('td', 'tri-lt-delta', fmtSportDelta(formatter, estimate, points))
  const datesTd = el('td', 'tri-lt-dates', fmtSeriesDateRange(formatter, points, estimate.date))
  const chartTd = el('td', 'tri-lt-chart')
  const spark = buildTrendPolylineSvg(
    points,
    estimate.value,
    `tri-line-${estimate.sport}`,
    `${label} · ${hoverText}`,
  )
  chartTd.appendChild(buildLactateSparkContainer(label, hoverText, spark))

  row.append(th, valTd, deltaTd, datesTd, chartTd)
  return row
}

export const buildLactateThreshold = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('div', 'tri-ana-lactate')
  block.appendChild(anaTitle(context.formatter, 'lactate threshold', 'lactate'))
  const heartRate = data.engine.lactateThreshold.heartRate
  const heartRateSeries = data.engine.lactateThreshold.heartRateSeries ?? []
  const sports = data.engine.lactateThreshold.sports
  if (!heartRate && !sports.length) {
    block.appendChild(el('div', 'tri-ana-empty', '—'))
    return block
  }
  const table = el('table', 'tri-lt-table')
  const body = el('tbody')
  if (heartRate) {
    body.appendChild(buildLactateHeartRateRow(heartRate, heartRateSeries, context))
  }
  for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
    const estimate = bySport(sports, sport)
    if (estimate) {
      body.appendChild(buildLactateThresholdRow(estimate, bySport(data.trends, sport), context))
    }
  }
  table.appendChild(body)
  block.appendChild(table)
  return block
}

export const buildActions = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('div', 'tri-ana-actions')
  block.appendChild(anaTitle(context.formatter, 'things to improve'))
  const banner = el('div', 'tri-actions-head')
  banner.append(
    el('span', 'tri-actions-weak', context.formatter.text('weakest')),
    buildIconLeg(context.formatter, data.weakestSport),
    el('span', 'tri-ana-k', data.weakestSport),
  )
  block.appendChild(banner)
  if (data.actions.length) {
    const tbl = el('table', 'tri-act-stats')
    const body = document.createElement('tbody')
    data.actions.forEach((a, i) => {
      const tr = document.createElement('tr')
      tr.append(
        el('th', 'tri-act-stat-k', `${i + 1}. ${a.text}`),
        el('td', 'tri-act-stat-v', a.value),
      )
      body.appendChild(tr)
    })
    tbl.appendChild(body)
    block.appendChild(tbl)
  }
  const chips = el('div', 'tri-gauge-chips')
  for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
    const th = bySport(data.thresholds, sport)
    if (th)
      chips.appendChild(
        el(
          'span',
          `tri-ana-chip tri-chip-${sport}`,
          `${sport} ${thLabel(context.formatter, th)} ${th.conf}`,
        ),
      )
  }
  block.appendChild(chips)
  return block
}
