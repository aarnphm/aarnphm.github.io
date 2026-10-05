import type { Analytics } from '../../../../plugins/stores/analytics'
import type { SportTrend } from '../../../../plugins/stores/analytics'
import type { Sport } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import type { RaceLegSplit } from '../shared'
import { clock } from '../../../../util/triathlon-card'
import { KM_TO_MI } from '../../../../util/triathlon-card'
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
import { fmtSpeedKmh } from '../shared'
import { hms } from '../shared'
import { markGloss } from '../shared'
import { polyD } from '../shared'
import { RACE_LABEL } from '../shared'
import { raceLegTip } from '../shared'
import { signedFixed } from '../shared'
import { speedFromKmh } from '../shared'
import { thLabel } from '../shared'
import { trendDir } from '../shared'

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

export const METHOD_WIKI: Record<string, string> = {
  ols: 'Ordinary least squares',
  ewma: 'Exponential smoothing',
}

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

export type TrendSamples = { centers: number[]; los: number[]; his: number[]; days: number }

export const trendSamples = (tr: SportTrend): TrendSamples | null => {
  if (tr.level == null || tr.forecast.length < 1) return null
  const lvl = tr.level
  return {
    centers: [lvl, ...tr.forecast.map(p => p.value)],
    los: [lvl, ...tr.forecast.map(p => p.lo)],
    his: [lvl, ...tr.forecast.map(p => p.hi)],
    days: tr.forecast.length,
  }
}

export const sampleTrend = (
  s: TrendSamples,
  f: number,
): { value: number; lo: number; hi: number; days: number } => {
  const q = clampN(f, 0, 1) * s.days
  const i0 = Math.floor(q)
  const i1 = Math.min(s.days, i0 + 1)
  const t = q - i0
  const at = (a: number[]): number => a[i0] + (a[i1] - a[i0]) * t
  return { value: at(s.centers), lo: at(s.los), hi: at(s.his), days: q }
}

export const trendChartGeometry = (
  invert: boolean,
  samples: TrendSamples,
  capBand = true,
): { line: string; band: string; low: number; high: number } => {
  const { centers, los, his, days: M } = samples
  const level = centers[0]
  let cLo = Infinity
  let cHi = -Infinity
  for (let i = 0; i <= M; i++) {
    if (centers[i] > cHi) cHi = centers[i]
    if (centers[i] < cLo) cLo = centers[i]
  }
  const scale = Math.max(cHi - cLo, Math.abs(level) * 0.05, 1e-6)
  const coneMax = scale * 0.5
  const lowAt = (i: number): number => (capBand ? Math.max(los[i], centers[i] - coneMax) : los[i])
  const highAt = (i: number): number => (capBand ? Math.min(his[i], centers[i] + coneMax) : his[i])
  let lo = cLo
  let hi = cHi
  for (let i = 0; i <= M; i++) {
    hi = Math.max(hi, highAt(i))
    lo = Math.min(lo, lowAt(i))
  }
  const pad = scale * 0.3
  lo -= pad
  hi += pad
  const span = Math.max(1e-6, hi - lo)
  const xOf = (i: number): number => (i / M) * ANA_W
  const y = (value: number): number => {
    const t = (value - lo) / span
    return invert ? t * ANA_H : (1 - t) * ANA_H
  }
  const hiPts: [number, number][] = []
  const loPts: [number, number][] = []
  const midPts: [number, number][] = []
  for (let i = 0; i <= M; i++) {
    hiPts.push([xOf(i), y(highAt(i))])
    loPts.push([xOf(i), y(lowAt(i))])
    midPts.push([xOf(i), y(centers[i])])
  }
  return {
    line: polyD(midPts),
    band: `${polyD([...hiPts, ...loPts.reverse()])} Z`,
    low: lo,
    high: hi,
  }
}

export const appendTrendChart = (
  formatter: TriathlonFormatter,
  wrap: HTMLElement,
  sport: Sport,
  invert: boolean,
  samples: TrendSamples,
  capBand = true,
): void => {
  const geometry = trendChartGeometry(invert, samples, capBand)
  const s = svg('svg', {
    class: 'tri-ana-svg tri-trend-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
  })
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-trend-axis' }))
  s.appendChild(svg('line', { x1: 0, y1: ANA_H, x2: ANA_W, y2: ANA_H, class: 'tri-trend-axis' }))
  s.appendChild(svg('path', { d: geometry.band, class: `tri-trend-band tri-fill-${sport}` }))
  s.appendChild(svg('path', { d: geometry.line, class: `tri-trend-proj tri-line-${sport}` }))
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
  const track = el('div', 'tri-trend-track')
  track.appendChild(s)
  const yax = el('div', 'tri-trend-yax')
  yax.append(
    el('span', '', fmtTrendShort(formatter, sport, invert ? geometry.low : geometry.high)),
    el('span', '', fmtTrendShort(formatter, sport, invert ? geometry.high : geometry.low)),
  )
  const chart = el('div', 'tri-trend-chart')
  chart.append(yax, track)
  const xax = el('div', 'tri-trend-xax')
  xax.append(
    el('span', '', formatter.text('now')),
    el('span', '', `+${Math.round(samples.days / 7)} wk`),
  )
  wrap.append(chart, xax, el('div', 'tri-chart-readout tri-trend-readout'))
}

export const buildTrendPanel = (
  data: Analytics,
  sport: Sport,
  context: TriathlonContext,
): HTMLElement => {
  const tr = bySport(data.trends, sport)
  const th = bySport(data.thresholds, sport)
  const wrap = el('div', `tri-trend-panel${tr?.stale ? ' tri-trend-stale' : ''}`)
  wrap.dataset.sport = sport
  const head = el('div', 'tri-trend-head')
  head.append(
    buildIconLeg(context.formatter, sport),
    markGloss(
      el('span', 'tri-trend-unit', th ? thLabel(context.formatter, th) : sport),
      'threshold',
    ),
  )
  if (th)
    head.appendChild(markGloss(el('span', `tri-ana-conf tri-conf-${th.conf}`, th.conf), 'conf'))
  wrap.appendChild(head)
  if (!tr || tr.method === 'none') {
    const daysSinceLastEffort =
      tr?.daysSinceLastEffort ?? (th && th.staleDays > 45 ? th.staleDays : null)
    const msg = trendUnavailableText(
      context.presentation.locale,
      tr?.sampleSize ?? null,
      daysSinceLastEffort,
    )
    wrap.appendChild(el('div', 'tri-trend-note', msg))
    return wrap
  }
  const samples = trendSamples(tr)
  if (samples) appendTrendChart(context.formatter, wrap, sport, tr.invert, samples)
  const cal = bySport(data.calibration.paces, sport)
  if (cal) {
    const cap = el('div', 'tri-elev-cap tri-trend-cap')
    if (cal.average != null)
      cap.appendChild(
        el(
          'span',
          'tri-ana-k',
          `${context.formatter.text('avg')} ${fmtTrendVal(context.formatter, sport, cal.average)}`,
        ),
      )
    if (cal.projected != null)
      cap.appendChild(
        el('span', 'tri-ana-k', `proj ${fmtTrendVal(context.formatter, sport, cal.projected)}`),
      )
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
        el(
          'span',
          'tri-ana-k',
          `${context.formatter.text('latest')} ${context.formatter.shortDate(cal.latestDate)}`,
        ),
      )
    wrap.appendChild(cap)
  }
  const dir = trendDir(tr.invert, tr.slopePerWeek)
  const note = el('div', 'tri-trend-note')
  note.append(
    markGloss(
      el(
        'span',
        `tri-trend-dir tri-dir-${dir > 0 ? 'up' : dir < 0 ? 'down' : 'flat'}`,
        dir > 0 ? 'faster' : dir < 0 ? 'slower' : 'flat',
      ),
      'trend',
    ),
    buildMethod(context.formatter, tr.method, tr.sampleSize),
  )
  wrap.appendChild(note)
  return wrap
}

export const buildTrend = (
  data: Analytics,
  context: TriathlonContext,
): { element: HTMLElement; mount: () => () => void } => {
  const block = el('div', 'tri-ana-trend')
  block.appendChild(anaTitle(context.formatter, 'pace trend + forecast'))
  for (const sport of ['swim', 'bike', 'run'] as Sport[])
    block.appendChild(buildTrendPanel(data, sport, context))
  const predictor = buildDistancePredictor(context.pace, context, {
    min: data.meta.windowFrom,
    max: data.meta.today,
  })
  block.appendChild(predictor.element)
  return { element: block, mount: predictor.mount }
}

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

const buildLactateThresholdRow = (
  estimate: LactateThresholdEstimate,
  context: TriathlonContext,
): HTMLElement => {
  const { formatter } = context
  const row = el('div', 'tri-lt-row')
  row.dataset.sport = estimate.sport
  row.dataset.source = estimate.source
  const head = el('div', 'tri-trend-head')
  head.append(
    buildIconLeg(formatter, estimate.sport),
    markGloss(
      el('span', 'tri-trend-unit', `LT2 ${fmtLactateValue(formatter, estimate)}`),
      'lactate',
    ),
  )
  if (estimate.conf)
    head.appendChild(
      markGloss(el('span', `tri-ana-conf tri-conf-${estimate.conf}`, estimate.conf), 'conf'),
    )
  row.append(head, el('div', 'tri-trend-note', lactateSourceText(formatter, estimate)))
  const device = estimate.device
  if (device && !device.accepted)
    row.appendChild(
      el(
        'div',
        'tri-trend-note tri-lt-device',
        `${formatter.text('Garmin device estimate')} ${fmtTrendVal(formatter, 'run', device.value)} · ${device.date}${device.deltaPct != null ? ` · ${signedFixed(device.deltaPct, 1)}% · ${formatter.text('outside ±5%, not used')}` : ''}`,
      ),
    )
  return row
}

export const buildLactateThreshold = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('div', 'tri-ana-lactate')
  block.appendChild(anaTitle(context.formatter, 'lactate threshold', 'lactate'))
  const heartRate = data.engine.lactateThreshold.heartRate
  if (heartRate) {
    const cap = el('div', 'tri-elev-cap')
    cap.append(
      markGloss(el('span', 'tri-ana-k', `LTHR ${heartRate.value} ${heartRate.unit}`), 'lactate'),
      el(
        'span',
        'tri-ana-k',
        heartRate.source === 'garmin'
          ? `${context.formatter.text('Garmin running estimate')} · ${heartRate.date}`
          : context.formatter.text('declared heart-rate anchor'),
      ),
    )
    block.appendChild(cap)
  }
  const sports = data.engine.lactateThreshold.sports
  if (!sports.length) block.appendChild(el('div', 'tri-ana-empty', '—'))
  for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
    const estimate = bySport(sports, sport)
    if (estimate) block.appendChild(buildLactateThresholdRow(estimate, context))
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
