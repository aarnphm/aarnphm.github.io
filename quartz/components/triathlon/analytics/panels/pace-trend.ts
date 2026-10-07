import type { Analytics } from '../../../../plugins/stores/analytics'
import type { SportTrend, SportTrendForecastPoint } from '../../../../plugins/stores/analytics'
import type { Sport } from '../../../../plugins/stores/strava'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import { trendUnavailableText } from '../../../../util/triathlon-i18n'
import { el } from '../../runtime/dom'
import { svg } from '../../runtime/dom'
import { buildDistancePredictor } from '../../tools/pace-forecast'
import { ANA_H } from '../shared'
import { ANA_W } from '../shared'
import { anaTitle } from '../shared'
import { buildIconLeg } from '../shared'
import { bySport } from '../shared'
import { clampN } from '../shared'
import { markGloss } from '../shared'
import { polyD } from '../shared'
import { signedFixed } from '../shared'
import { thLabel } from '../shared'
import { buildMethod } from './thresholds'
import { fmtTrendShort } from './thresholds'
import { fmtTrendVal } from './thresholds'

const DAY_MS = 86_400_000
const dayOf = (iso: string): number => Date.parse(`${iso}T00:00:00Z`) / DAY_MS

/** One calendar for every sport, so the synced cursor reads the same date in each panel. */
export type TrendFrame = { today: string; before: number; after: number }

export const trendFrame = (trends: readonly SportTrend[]): TrendFrame | null => {
  const live = trends.filter(tr => tr.history.length > 0)
  if (!live.length) return null
  return {
    today: live[0].history[live[0].history.length - 1].date,
    before: Math.max(...live.map(tr => tr.history.length - 1)),
    after: Math.max(...live.map(tr => tr.forecast.length)),
  }
}

const offsetOf = (frame: TrendFrame, iso: string): number => dayOf(iso) - dayOf(frame.today)

const xOf = (frame: TrendFrame, offset: number): number =>
  ((offset + frame.before) / (frame.before + frame.after)) * ANA_W

const dateAt = (frame: TrendFrame, offset: number): string =>
  new Date((dayOf(frame.today) + offset) * DAY_MS).toISOString().slice(0, 10)

export const trendValueText = (formatter: TriathlonFormatter, tr: SportTrend, v: number): string =>
  tr.signal === 'power-at-heart-rate' ? `${Math.round(v)} W` : fmtTrendVal(formatter, tr.sport, v)

export const trendShortText = (formatter: TriathlonFormatter, tr: SportTrend, v: number): string =>
  tr.signal === 'power-at-heart-rate'
    ? String(Math.round(v))
    : fmtTrendShort(formatter, tr.sport, v)

const rangeText = (formatter: TriathlonFormatter, tr: SportTrend, p: SportTrendForecastPoint) =>
  `${trendShortText(formatter, tr, p.lo)}–${trendShortText(formatter, tr, p.hi)}`

/** Weekly change in the signal, in percent; positive means fitter. */
export const trendWeeklyPct = (tr: SportTrend): number | null => {
  if (tr.level == null || tr.slopePerWeek == null || !(tr.level > 0)) return null
  const next = tr.level + tr.slopePerWeek
  if (!(next > 0)) return null
  return ((tr.invert ? tr.level / next : next / tr.level) - 1) * 100
}

/** What the trend measures, for the panel head and the readout. */
export const trendSignalText = (formatter: TriathlonFormatter, tr: SportTrend): string =>
  tr.referenceHeartRate != null ? `@ ${tr.referenceHeartRate} bpm` : formatter.text('session pace')

const dotWidth = (windows: number): number =>
  windows > 0 ? clampN(3 + 1.6 * Math.log(1 + windows), 3.5, 8) : 5

const dot = (x: number, y: number, cls: string, width: number): SVGElement => {
  const path = svg('path', { d: `M ${x.toFixed(2)} ${y.toFixed(2)} l 0.01 0`, class: cls })
  path.style.setProperty('--tri-dot-w', `${width.toFixed(1)}px`)
  return path
}

export const trendChartGeometry = (
  tr: SportTrend,
): { y: (value: number) => number; low: number; high: number } => {
  const values: number[] = []
  for (const p of [...tr.history, ...tr.forecast, ...tr.issued]) values.push(p.lo, p.hi)
  for (const s of tr.sessions) if (!s.outlier) values.push(s.value)
  let low = Math.min(...values)
  let high = Math.max(...values)
  // A flat series still gets a readable span: 4% of the level.
  const minSpan = Math.abs(tr.level ?? high) * 0.04
  if (high - low < minSpan) {
    const mid = (high + low) / 2
    low = mid - minSpan / 2
    high = mid + minSpan / 2
  }
  const pad = (high - low) * 0.08
  low -= pad
  high += pad
  const span = high - low
  const y = (value: number): number => {
    const t = (value - low) / span
    // Down-weighted sessions can sit outside the range; pin them just inside the edge.
    return clampN(tr.invert ? t * ANA_H : (1 - t) * ANA_H, 0.6, ANA_H - 0.6)
  }
  return { y, low, high }
}

const bandPath = (
  frame: TrendFrame,
  points: readonly SportTrendForecastPoint[],
  y: (value: number) => number,
): string => {
  const hi = points.map(p => [xOf(frame, offsetOf(frame, p.date)), y(p.hi)] as [number, number])
  const lo = points.map(p => [xOf(frame, offsetOf(frame, p.date)), y(p.lo)] as [number, number])
  return `${polyD([...hi, ...lo.reverse()])} Z`
}

const linePath = (
  frame: TrendFrame,
  points: readonly SportTrendForecastPoint[],
  y: (value: number) => number,
): string => polyD(points.map(p => [xOf(frame, offsetOf(frame, p.date)), y(p.value)]))

export const appendTrendChart = (
  formatter: TriathlonFormatter,
  wrap: HTMLElement,
  frame: TrendFrame,
  tr: SportTrend,
): void => {
  const { sport } = tr
  const { y, low, high } = trendChartGeometry(tr)
  const now = tr.history[tr.history.length - 1]
  const xNow = xOf(frame, 0)
  const s = svg('svg', {
    class: 'tri-ana-svg tri-trend-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
    'aria-label': trendSummaryText(formatter, tr),
  })
  s.appendChild(
    svg('rect', { x: xNow, y: 0, width: ANA_W - xNow, height: ANA_H, class: 'tri-trend-future' }),
  )
  // Gridlines run back from today every two weeks, or four once the frame passes ten weeks, so the
  // date labels under them stay apart at phone width.
  const step = frame.before > 70 ? 28 : 14
  const grid: number[] = []
  for (let offset = -step; offset > -frame.before; offset -= step) grid.push(offset)
  for (const offset of grid) {
    const x = xOf(frame, offset)
    s.appendChild(svg('line', { x1: x, y1: 0, x2: x, y2: ANA_H, class: 'tri-trend-tick' }))
  }
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-trend-axis' }))
  s.appendChild(svg('line', { x1: 0, y1: ANA_H, x2: ANA_W, y2: ANA_H, class: 'tri-trend-axis' }))
  s.appendChild(
    svg('path', {
      d: bandPath(frame, [...tr.history, ...tr.forecast], y),
      class: `tri-trend-band tri-fill-${sport}`,
    }),
  )
  if (tr.issued.length > 1) {
    s.appendChild(svg('path', { d: bandPath(frame, tr.issued, y), class: 'tri-trend-issued' }))
    s.appendChild(svg('path', { d: linePath(frame, tr.issued, y), class: 'tri-trend-issued-mid' }))
  }
  s.appendChild(
    svg('path', { d: linePath(frame, tr.history, y), class: `tri-trend-level tri-line-${sport}` }),
  )
  s.appendChild(
    svg('path', {
      d: linePath(frame, [now, ...tr.forecast], y),
      class: `tri-trend-proj tri-line-${sport}`,
    }),
  )
  for (const session of tr.sessions) {
    const g = svg('g', {
      class: `tri-trend-dot tri-trend-dot-${sport}${session.outlier ? ' tri-trend-dot--out' : ''}`,
      'data-date': session.date,
    })
    const x = xOf(frame, offsetOf(frame, session.date))
    const width = dotWidth(session.windows)
    g.appendChild(dot(x, y(session.value), 'tri-trend-dot-o', width))
    if (session.outlier) g.appendChild(dot(x, y(session.value), 'tri-trend-dot-i', width * 0.55))
    s.appendChild(g)
  }
  s.appendChild(svg('line', { x1: xNow, y1: 0, x2: xNow, y2: ANA_H, class: 'tri-trend-now' }))
  const mark = svg('g', { class: 'tri-trend-mark' })
  mark.append(
    dot(xNow, y(now.value), 'tri-trend-mark-o', 6.5),
    dot(xNow, y(now.value), 'tri-trend-mark-i', 3),
  )
  s.appendChild(mark)
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))

  const track = el('div', 'tri-trend-track')
  track.appendChild(s)
  const yax = el('div', 'tri-trend-yax')
  yax.append(
    el('span', '', trendShortText(formatter, tr, tr.invert ? low : high)),
    el('span', '', trendShortText(formatter, tr, tr.invert ? high : low)),
  )
  const chart = el('div', 'tri-trend-chart')
  chart.append(yax, track)
  // On a long frame the forecast strip is too narrow at phone width for both "now" and "+2 wk".
  const longFrame = frame.after / (frame.before + frame.after) < 0.15
  const xax = el('div', `tri-trend-xax${longFrame ? ' tri-trend-xax--long' : ''}`)
  const tick = (offset: number, text: string, cls = ''): void => {
    const label = el('span', cls, text)
    label.style.left = `${((xOf(frame, offset) / ANA_W) * 100).toFixed(2)}%`
    xax.appendChild(label)
  }
  if (frame.before + (grid.at(-1) ?? 0) >= step / 2)
    tick(-frame.before, formatter.shortDate(dateAt(frame, -frame.before)), 'tri-trend-x0')
  for (const offset of grid.toReversed()) tick(offset, formatter.shortDate(dateAt(frame, offset)))
  tick(0, formatter.text('now'), 'tri-trend-xnow')
  tick(frame.after, `+${Math.round(frame.after / 7)} ${formatter.text('wk')}`, 'tri-trend-xend')
  wrap.append(chart, xax, el('div', 'tri-chart-readout tri-trend-readout'))
}

export const trendSummaryText = (formatter: TriathlonFormatter, tr: SportTrend): string => {
  const parts: string[] = []
  if (tr.level != null)
    parts.push(
      `${formatter.text(tr.sport)} ${trendValueText(formatter, tr, tr.level)} ${trendSignalText(formatter, tr)}`,
    )
  const pct = trendWeeklyPct(tr)
  if (pct != null) parts.push(`${signedFixed(pct, 1)}%/${formatter.text('wk')}`)
  const end = tr.forecast[tr.forecast.length - 1]
  if (end)
    parts.push(
      `+${Math.round(tr.forecast.length / 7)} ${formatter.text('wk')} ${trendValueText(formatter, tr, end.value)} (${rangeText(formatter, tr, end)})`,
    )
  return parts.join(' · ')
}

/** Readout for a cursor at fraction f of the shared frame. */
export const trendReadout = (
  formatter: TriathlonFormatter,
  frame: TrendFrame,
  tr: SportTrend,
  f: number,
): { text: string; date: string } => {
  const offset = Math.round(clampN(f, 0, 1) * (frame.before + frame.after)) - frame.before
  const date = dateAt(frame, offset)
  if (offset > 0) {
    const p = tr.forecast[offset - 1]
    if (!p) return { text: '', date }
    return {
      text: `+${(offset / 7).toFixed(1)} ${formatter.text('wk')} · ${trendValueText(formatter, tr, p.value)} · ${rangeText(formatter, tr, p)}`,
      date,
    }
  }
  const p = tr.history.find(point => point.date === date)
  const parts = [formatter.shortDate(date)]
  const watts = tr.signal === 'power-at-heart-rate' ? ' W' : ''
  for (const session of tr.sessions.filter(s => s.date === date)) {
    const value = `${trendShortText(formatter, tr, session.value)}${watts}`
    const text =
      session.steady != null && session.heartRate != null
        ? `${formatter.text('steady')} ${trendShortText(formatter, tr, session.steady)}${watts} @ ${session.heartRate} bpm → ${value} @ ${tr.referenceHeartRate} bpm`
        : value
    parts.push(session.outlier ? `${text} (${formatter.text('low weight')})` : text)
  }
  if (p)
    parts.push(
      `${formatter.text('trend')} ${trendValueText(formatter, tr, p.value)} (${rangeText(formatter, tr, p)})`,
    )
  return { text: parts.join(' · '), date }
}

/** The forecast issued two weeks ago, its call for today, and how often such calls held. */
export const buildTrendIssuedCap = (
  formatter: TriathlonFormatter,
  tr: SportTrend,
): HTMLElement | null => {
  const from = tr.issued[0]
  const call = tr.issued[tr.issued.length - 1]
  const now = tr.history[tr.history.length - 1]
  if (!from || !call || !now) return null
  const cap = el('div', 'tri-elev-cap tri-trend-cap tri-trend-calls')
  const held = now.value >= call.lo && now.value <= call.hi
  cap.append(
    el(
      'span',
      'tri-ana-k',
      `${formatter.text('forecast from')} ${formatter.shortDate(from.date)} ${trendValueText(formatter, tr, call.value)} (${rangeText(formatter, tr, call)})`,
    ),
    el(
      'span',
      `tri-ana-k ${held ? 'tri-trend-held' : 'tri-trend-missed'}`,
      `${formatter.text('now')} ${trendValueText(formatter, tr, now.value)}`,
    ),
  )
  if (tr.hindcast)
    cap.appendChild(
      el(
        'span',
        'tri-ana-k',
        `${formatter.text('band held')} ${tr.hindcast.hits}/${tr.hindcast.days} ${formatter.text('days')}`,
      ),
    )
  return cap
}

const buildTrendCalibrationCap = (
  data: Analytics,
  sport: Sport,
  formatter: TriathlonFormatter,
): HTMLElement | null => {
  const cal = bySport(data.calibration.paces, sport)
  if (!cal) return null
  const cap = el('div', 'tri-elev-cap tri-trend-cap')
  if (cal.average != null)
    cap.appendChild(
      el(
        'span',
        'tri-ana-k',
        `${formatter.text('avg')} ${fmtTrendVal(formatter, sport, cal.average)}`,
      ),
    )
  if (cal.projected != null)
    cap.appendChild(el('span', 'tri-ana-k', `proj ${fmtTrendVal(formatter, sport, cal.projected)}`))
  if (cal.deltaPct != null) {
    const cls =
      cal.direction === 'faster'
        ? 'tri-dir-up'
        : cal.direction === 'slower'
          ? 'tri-dir-down'
          : 'tri-dir-flat'
    cap.appendChild(
      el(
        'span',
        `tri-ana-k ${cls}`,
        `${signedFixed(cal.deltaPct, 1)}% vs prev ${data.calibration.windowDays}d`,
      ),
    )
  }
  if (cal.projectedDeltaPct != null)
    cap.appendChild(
      el(
        'span',
        'tri-ana-k',
        `${signedFixed(cal.projectedDeltaPct, 1)}% next ${data.calibration.projectionDays}d`,
      ),
    )
  cap.appendChild(el('span', 'tri-ana-k', `n ${cal.sampleSize}/${cal.previousSampleSize}`))
  if (cal.latestDate)
    cap.appendChild(
      el('span', 'tri-ana-k', `${formatter.text('latest')} ${formatter.shortDate(cal.latestDate)}`),
    )
  return cap
}

export const buildTrendPanel = (
  data: Analytics,
  sport: Sport,
  frame: TrendFrame | null,
  context: TriathlonContext,
): HTMLElement => {
  const { formatter } = context
  const tr = bySport(data.trends, sport)
  const th = bySport(data.thresholds, sport)
  const wrap = el('div', `tri-trend-panel${tr?.stale ? ' tri-trend-stale' : ''}`)
  wrap.dataset.sport = sport
  const head = el('div', 'tri-trend-head')
  head.appendChild(buildIconLeg(formatter, sport))
  wrap.appendChild(head)
  if (!tr || tr.method === 'none' || tr.level == null || !frame || !tr.history.length) {
    head.appendChild(
      markGloss(el('span', 'tri-trend-unit', th ? thLabel(formatter, th) : sport), 'threshold'),
    )
    const daysSinceLastEffort =
      tr?.daysSinceLastEffort ?? (th && th.staleDays > 45 ? th.staleDays : null)
    wrap.appendChild(
      el(
        'div',
        'tri-trend-note',
        trendUnavailableText(
          context.presentation.locale,
          tr?.sampleSize ?? null,
          daysSinceLastEffort,
        ),
      ),
    )
    return wrap
  }
  head.append(
    markGloss(el('span', 'tri-trend-unit', trendValueText(formatter, tr, tr.level)), 'trend'),
    el('span', 'tri-trend-signal', trendSignalText(formatter, tr)),
  )
  const pct = trendWeeklyPct(tr)
  if (pct != null)
    head.appendChild(
      el(
        'span',
        `tri-trend-rate tri-dir-${Math.abs(pct) < 0.25 ? 'flat' : pct > 0 ? 'up' : 'down'}`,
        `${signedFixed(pct, 1)}%/${formatter.text('wk')}`,
      ),
    )
  appendTrendChart(formatter, wrap, frame, tr)
  const issued = buildTrendIssuedCap(formatter, tr)
  if (issued) wrap.appendChild(issued)
  const calibration = buildTrendCalibrationCap(data, sport, formatter)
  if (calibration) wrap.appendChild(calibration)
  const note = el('div', 'tri-trend-note')
  note.appendChild(buildMethod(formatter, tr.method, tr.sampleSize))
  if (tr.heartRateSlopePct != null)
    note.appendChild(
      el(
        'span',
        'tri-ana-k',
        `${tr.heartRateSlopePct.toFixed(2)}% ${tr.signal === 'power-at-heart-rate' ? 'W' : formatter.text('speed')}/bpm`,
      ),
    )
  wrap.appendChild(note)
  return wrap
}

export const buildTrend = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount: () => () => void } => {
  const block = el('div', 'tri-ana-trend')
  block.appendChild(anaTitle(context.formatter, 'pace trend + forecast', 'trend'))
  const frame = trendFrame(data.trends)
  for (const sport of ['swim', 'bike', 'run'] as Sport[])
    block.appendChild(buildTrendPanel(data, sport, frame, context))
  const predictor = buildDistancePredictor(context.pace, context, {
    min: data.meta.windowFrom,
    max: data.meta.today,
  })
  block.appendChild(predictor.element)
  return { element: block, mount: predictor.mount }
}
