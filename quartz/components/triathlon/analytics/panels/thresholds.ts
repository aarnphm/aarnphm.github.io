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
import { ANA_W } from '../shared'
import { anaTitle } from '../shared'
import { buildIconLeg } from '../shared'
import { bySport } from '../shared'
import { clampN } from '../shared'
import { fmtSpeedKmh } from '../shared'
import { hms } from '../shared'
import { markGloss } from '../shared'
import { markGlossDefinition } from '../shared'
import { polyD } from '../shared'
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

const LT_SPARK_H = 18
const LT_DAY_MS = 86_400_000
const ltDayOf = (iso: string): number => Date.parse(`${iso}T00:00:00Z`) / LT_DAY_MS

type LactateSparkFrame = { today: string; before: number; after: number }

const buildLactateSparkFrame = (data: Analytics): LactateSparkFrame => {
  const live = data.trends.filter(tr => tr.history.length > 0)
  const today = live[0]?.history[live[0].history.length - 1]?.date ?? data.meta.today
  const todayDay = ltDayOf(today)
  const hrSeries = data.engine.lactateThreshold.heartRateSeries ?? []
  const historySpans = [
    ...live.map(tr => Math.max(0, todayDay - ltDayOf(tr.history[0].date))),
    ...(hrSeries.length ? [Math.max(0, todayDay - ltDayOf(hrSeries[0].date))] : []),
  ]
  const before = Math.max(1, ...historySpans)
  const after = Math.max(0, ...live.map(tr => tr.forecast.length))
  return { today, before, after }
}

const ltNowX = (frame: LactateSparkFrame): number =>
  (frame.before / (frame.before + frame.after)) * ANA_W

const ltHistoryXAt = (frame: LactateSparkFrame, startIso: string, iso: string): number => {
  const xNow = ltNowX(frame)
  const span = Math.max(1, ltDayOf(frame.today) - ltDayOf(startIso))
  return clampN(((ltDayOf(iso) - ltDayOf(startIso)) / span) * xNow, 0, xNow)
}

const ltForecastXAt = (frame: LactateSparkFrame, iso: string): number => {
  const xNow = ltNowX(frame)
  if (frame.after <= 0) return xNow
  return clampN(
    xNow + ((ltDayOf(iso) - ltDayOf(frame.today)) / frame.after) * (ANA_W - xNow),
    xNow,
    ANA_W,
  )
}

const ltDot = (x: number, y: number, cls: string, width: number): SVGElement => {
  const path = svg('path', { d: `M ${x.toFixed(2)} ${y.toFixed(2)} l 0.01 0`, class: cls })
  path.style.setProperty('--tri-dot-w', `${width.toFixed(1)}px`)
  return path
}

const buildPulseIcon = (): HTMLElement => {
  const wrap = el('span', 'tri-ana-ico tri-leg-run')
  const icon = svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
  icon.appendChild(svg('path', { d: 'M3 12h4l2.5-6 4.5 12 2.5-6H21' }))
  wrap.appendChild(icon)
  return wrap
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

const buildLactateSparkContainer = (
  termLabel: string,
  hoverText: string,
  sparkSvg: SVGElement,
): HTMLElement => {
  const wrap = markGlossDefinition(el('div', 'tri-lt-trend'), hoverText)
  wrap.dataset.glossTerm = termLabel
  wrap.appendChild(sparkSvg)
  return wrap
}

const buildHeartRateSparkSvg = (
  heartRate: LactateThresholdHeartRate,
  series: readonly GarminLactateThresholdValue[],
  frame: LactateSparkFrame,
  ariaLabel: string,
): SVGElement => {
  const xNow = ltNowX(frame)
  const s = svg('svg', {
    class: 'tri-lt-spark',
    viewBox: `0 0 ${ANA_W} ${LT_SPARK_H}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': ariaLabel,
  })
  if (frame.after > 0) {
    s.appendChild(
      svg('rect', {
        x: xNow,
        y: 0,
        width: ANA_W - xNow,
        height: LT_SPARK_H,
        class: 'tri-trend-future',
      }),
    )
  }
  s.appendChild(
    svg('line', { x1: 0, y1: LT_SPARK_H, x2: ANA_W, y2: LT_SPARK_H, class: 'tri-trend-axis' }),
  )
  const points =
    series.length >= 2
      ? series[series.length - 1].date < frame.today
        ? [...series, { date: frame.today, value: heartRate.value }]
        : [...series]
      : [
          { date: frame.today, value: heartRate.value },
          { date: frame.today, value: heartRate.value },
        ]
  const values = points.map(p => p.value)
  let low = Math.min(...values)
  let high = Math.max(...values)
  const minSpan = Math.max(4, Math.abs(heartRate.value) * 0.04)
  if (high - low < minSpan) {
    const mid = (high + low) / 2
    low = mid - minSpan / 2
    high = mid + minSpan / 2
  }
  const pad = (high - low) * 0.1
  low -= pad
  high += pad
  const span = Math.max(1e-6, high - low)
  const yAt = (value: number): number =>
    clampN((1 - (value - low) / span) * (LT_SPARK_H - 3) + 1.5, 1.2, LT_SPARK_H - 1.2)
  const startIso = points[0].date
  const coords: [number, number][] =
    series.length >= 2
      ? points.map(p => [ltHistoryXAt(frame, startIso, p.date), yAt(p.value)])
      : [
          [0, yAt(heartRate.value)],
          [xNow, yAt(heartRate.value)],
        ]
  s.appendChild(svg('path', { d: polyD(coords), class: 'tri-trend-level tri-line-run' }))
  s.appendChild(svg('line', { x1: xNow, y1: 0, x2: xNow, y2: LT_SPARK_H, class: 'tri-trend-now' }))
  const mark = svg('g', { class: 'tri-trend-mark' })
  const yNow = yAt(heartRate.value)
  mark.append(ltDot(xNow, yNow, 'tri-trend-mark-o', 5), ltDot(xNow, yNow, 'tri-trend-mark-i', 2.2))
  s.appendChild(mark)
  return s
}

const buildSportSparkSvg = (
  estimate: LactateThresholdEstimate,
  trend: SportTrend | undefined,
  frame: LactateSparkFrame,
  ariaLabel: string,
): SVGElement => {
  const { sport } = estimate
  const invert = sport !== 'bike'
  const xNow = ltNowX(frame)
  const s = svg('svg', {
    class: 'tri-lt-spark',
    viewBox: `0 0 ${ANA_W} ${LT_SPARK_H}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': ariaLabel,
  })
  if (frame.after > 0) {
    s.appendChild(
      svg('rect', {
        x: xNow,
        y: 0,
        width: ANA_W - xNow,
        height: LT_SPARK_H,
        class: 'tri-trend-future',
      }),
    )
  }
  s.appendChild(
    svg('line', { x1: 0, y1: LT_SPARK_H, x2: ANA_W, y2: LT_SPARK_H, class: 'tri-trend-axis' }),
  )
  if (!trend || trend.history.length < 2 || trend.level == null || !(trend.level > 0)) {
    const midY = LT_SPARK_H / 2
    s.appendChild(
      svg('path', {
        d: polyD([
          [0, midY],
          [xNow, midY],
        ]),
        class: `tri-trend-level tri-line-${sport}`,
      }),
    )
    s.appendChild(
      svg('line', { x1: xNow, y1: 0, x2: xNow, y2: LT_SPARK_H, class: 'tri-trend-now' }),
    )
    const mark = svg('g', { class: 'tri-trend-mark' })
    mark.append(
      ltDot(xNow, midY, 'tri-trend-mark-o', 5),
      ltDot(xNow, midY, 'tri-trend-mark-i', 2.2),
    )
    s.appendChild(mark)
    return s
  }
  const scale = estimate.value / trend.level
  const scalePoint = (p: SportTrendForecastPoint): SportTrendForecastPoint => ({
    date: p.date,
    value: p.value * scale,
    lo: p.lo * scale,
    hi: p.hi * scale,
  })
  const history = trend.history.map(scalePoint)
  const forecast = trend.forecast.map(scalePoint)
  const startIso = history[0].date
  const xFor = (iso: string): number =>
    iso <= frame.today ? ltHistoryXAt(frame, startIso, iso) : ltForecastXAt(frame, iso)
  const allPoints = [...history, ...forecast]
  const values = allPoints.flatMap(p => [p.lo, p.hi])
  let low = Math.min(...values)
  let high = Math.max(...values)
  const minSpan = Math.abs(estimate.value) * 0.04
  if (high - low < minSpan) {
    const mid = (high + low) / 2
    low = mid - minSpan / 2
    high = mid + minSpan / 2
  }
  const pad = (high - low) * 0.08
  low -= pad
  high += pad
  const span = Math.max(1e-6, high - low)
  const yAt = (value: number): number => {
    const t = (value - low) / span
    return clampN((invert ? t : 1 - t) * (LT_SPARK_H - 2.4) + 1.2, 0.8, LT_SPARK_H - 0.8)
  }
  const bandHi = allPoints.map(p => [xFor(p.date), yAt(p.hi)] as [number, number])
  const bandLo = allPoints.map(p => [xFor(p.date), yAt(p.lo)] as [number, number])
  s.appendChild(
    svg('path', {
      d: `${polyD([...bandHi, ...bandLo.reverse()])} Z`,
      class: `tri-trend-band tri-fill-${sport}`,
    }),
  )
  s.appendChild(
    svg('path', {
      d: polyD(history.map(p => [xFor(p.date), yAt(p.value)])),
      class: `tri-trend-level tri-line-${sport}`,
    }),
  )
  const now = history[history.length - 1]
  if (forecast.length > 0) {
    s.appendChild(
      svg('path', {
        d: polyD([now, ...forecast].map(p => [xFor(p.date), yAt(p.value)])),
        class: `tri-trend-proj tri-line-${sport}`,
      }),
    )
  }
  s.appendChild(svg('line', { x1: xNow, y1: 0, x2: xNow, y2: LT_SPARK_H, class: 'tri-trend-now' }))
  const mark = svg('g', { class: 'tri-trend-mark' })
  const yNow = yAt(now.value)
  mark.append(ltDot(xNow, yNow, 'tri-trend-mark-o', 5), ltDot(xNow, yNow, 'tri-trend-mark-i', 2.2))
  s.appendChild(mark)
  return s
}

const buildLactateHeartRateRow = (
  heartRate: LactateThresholdHeartRate,
  series: readonly GarminLactateThresholdValue[],
  frame: LactateSparkFrame,
  context: TriathlonContext,
): HTMLElement => {
  const { formatter } = context
  const row = el('div', 'tri-lt-row')
  row.dataset.sport = 'lthr'
  row.dataset.source = heartRate.source
  const label = `LTHR ${heartRate.value} ${heartRate.unit}`
  const hoverText = lactateHeartRateHoverText(formatter, heartRate, series)
  const head = el('div', 'tri-trend-head')
  head.append(buildPulseIcon(), markGlossDefinition(el('span', 'tri-trend-unit', label), hoverText))
  const spark = buildHeartRateSparkSvg(heartRate, series, frame, `${label} · ${hoverText}`)
  row.append(head, buildLactateSparkContainer(label, hoverText, spark))
  return row
}

const buildLactateThresholdRow = (
  estimate: LactateThresholdEstimate,
  trend: SportTrend | undefined,
  frame: LactateSparkFrame,
  context: TriathlonContext,
): HTMLElement => {
  const { formatter } = context
  const row = el('div', 'tri-lt-row')
  row.dataset.sport = estimate.sport
  row.dataset.source = estimate.source
  const label = `LT2 ${fmtLactateValue(formatter, estimate)}`
  const hoverText = lactateSportHoverText(formatter, estimate, trend)
  const head = el('div', 'tri-trend-head')
  head.append(
    buildIconLeg(formatter, estimate.sport),
    markGlossDefinition(el('span', 'tri-trend-unit', label), hoverText),
  )
  if (estimate.conf)
    head.appendChild(
      markGloss(el('span', `tri-ana-conf tri-conf-${estimate.conf}`, estimate.conf), 'conf'),
    )
  const spark = buildSportSparkSvg(estimate, trend, frame, `${label} · ${hoverText}`)
  row.append(head, buildLactateSparkContainer(label, hoverText, spark))
  return row
}

export const buildLactateThreshold = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('div', 'tri-ana-lactate')
  block.appendChild(anaTitle(context.formatter, 'lactate threshold', 'lactate'))
  const frame = buildLactateSparkFrame(data)
  const heartRate = data.engine.lactateThreshold.heartRate
  const heartRateSeries = data.engine.lactateThreshold.heartRateSeries ?? []
  if (heartRate) {
    block.appendChild(buildLactateHeartRateRow(heartRate, heartRateSeries, frame, context))
  }
  const sports = data.engine.lactateThreshold.sports
  if (!sports.length) block.appendChild(el('div', 'tri-ana-empty', '—'))
  for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
    const estimate = bySport(sports, sport)
    if (estimate)
      block.appendChild(
        buildLactateThresholdRow(estimate, bySport(data.trends, sport), frame, context),
      )
  }
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
