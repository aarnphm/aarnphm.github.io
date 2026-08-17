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

export const appendTrendChart = (
  formatter: TriathlonFormatter,
  wrap: HTMLElement,
  sport: Sport,
  invert: boolean,
  samples: TrendSamples,
  capBand = true,
): void => {
  const { centers, los, his, days: M } = samples
  const level = centers[0]
  const weeks = M / 7
  let cLo = Infinity
  let cHi = -Infinity
  for (let i = 0; i <= M; i++) {
    if (centers[i] > cHi) cHi = centers[i]
    if (centers[i] < cLo) cLo = centers[i]
  }
  const scale = Math.max(cHi - cLo, Math.abs(level) * 0.05, 1e-6)
  const coneMax = scale * 0.5
  const halfAt = (i: number): number =>
    capBand ? Math.min((his[i] - los[i]) / 2, coneMax) : (his[i] - los[i]) / 2
  let lo = cLo
  let hi = cHi
  for (let i = 0; i <= M; i++) {
    const half = halfAt(i)
    if (centers[i] + half > hi) hi = centers[i] + half
    if (centers[i] - half < lo) lo = centers[i] - half
  }
  const pad = scale * 0.3
  lo -= pad
  hi += pad
  const span = Math.max(1e-6, hi - lo)
  const top = 4
  const bot = 24
  const xOf = (i: number): number => (i / M) * ANA_W
  const y = (value: number): number => {
    const t = (value - lo) / span
    return invert ? top + t * (bot - top) : bot - t * (bot - top)
  }
  const yClamped = (value: number): number => clampN(y(value), 0.5, ANA_H - 0.5)
  const s = svg('svg', {
    class: 'tri-ana-svg tri-trend-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
  })
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-trend-axis' }))
  s.appendChild(svg('line', { x1: 0, y1: ANA_H, x2: ANA_W, y2: ANA_H, class: 'tri-trend-axis' }))
  const hiPts: [number, number][] = []
  const loPts: [number, number][] = []
  const midPts: [number, number][] = []
  for (let i = 0; i <= M; i++) {
    const half = halfAt(i)
    hiPts.push([xOf(i), yClamped(centers[i] + half)])
    loPts.push([xOf(i), yClamped(centers[i] - half)])
    midPts.push([xOf(i), yClamped(centers[i])])
  }
  s.appendChild(
    svg('path', {
      d: `${polyD([...hiPts, ...loPts.reverse()])} Z`,
      class: `tri-trend-band tri-fill-${sport}`,
    }),
  )
  s.appendChild(svg('path', { d: polyD(midPts), class: `tri-trend-proj tri-line-${sport}` }))
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
  const track = el('div', 'tri-trend-track')
  const dot = el('span', `tri-trend-dot tri-bg-${sport}`)
  dot.style.left = '0%'
  dot.style.top = `${clampN((y(level) / ANA_H) * 100, 4, 96)}%`
  track.append(s, dot)
  const yax = el('div', 'tri-trend-yax')
  yax.append(
    el('span', '', fmtTrendShort(formatter, sport, invert ? lo : hi)),
    el('span', '', fmtTrendShort(formatter, sport, invert ? hi : lo)),
  )
  const chart = el('div', 'tri-trend-chart')
  chart.append(yax, track)
  const xax = el('div', 'tri-trend-xax')
  xax.append(el('span', '', formatter.text('now')), el('span', '', `+${Math.round(weeks)} wk`))
  wrap.append(chart, xax, el('div', 'tri-chart-readout'))
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
  const predictor = buildDistancePredictor(context.pace, context)
  block.appendChild(predictor.element)
  return { element: block, mount: predictor.mount }
}

export type LactateThresholdProjection = Analytics['engine']['lactateThreshold']['sports'][number]

export const lactateThresholdSamples = (
  projection: LactateThresholdProjection,
): TrendSamples | null => {
  if (projection.points.length < 2) return null
  return {
    centers: projection.points.map(point => point.value),
    los: projection.points.map(point => point.lo),
    his: projection.points.map(point => point.hi),
    days: projection.points.length - 1,
  }
}

export const buildLactateThresholdPanel = (
  data: Analytics,
  sport: Sport,
  context: TriathlonContext,
): HTMLElement => {
  const projection = bySport(data.engine.lactateThreshold.sports, sport)
  const threshold = bySport(data.thresholds, sport)
  const wrap = el(
    'div',
    `tri-trend-panel tri-lt-panel${projection?.projected == null ? ' tri-trend-stale' : ''}`,
  )
  wrap.dataset.sport = sport
  const head = el('div', 'tri-trend-head')
  head.append(
    buildIconLeg(context.formatter, sport),
    markGloss(
      el(
        'span',
        'tri-trend-unit',
        projection
          ? `LT2 ${fmtTrendVal(context.formatter, sport, projection.current)}`
          : (threshold?.paceLabel ?? sport),
      ),
      'lactate',
    ),
  )
  if (projection)
    head.appendChild(
      markGloss(el('span', `tri-ana-conf tri-conf-${projection.conf}`, projection.conf), 'conf'),
    )
  wrap.appendChild(head)
  if (!projection || projection.projected == null) {
    wrap.appendChild(
      el(
        'div',
        'tri-trend-note',
        trendUnavailableText(
          context.presentation.locale,
          projection?.sampleSize ?? null,
          threshold?.staleDays ?? null,
        ),
      ),
    )
    return wrap
  }
  const samples = lactateThresholdSamples(projection)
  if (samples) appendTrendChart(context.formatter, wrap, sport, sport !== 'bike', samples, false)
  const cap = el('div', 'tri-elev-cap tri-trend-cap')
  cap.append(
    el(
      'span',
      'tri-ana-k',
      `${context.formatter.text('projected')} ${fmtTrendVal(context.formatter, sport, projection.projected)}`,
    ),
    el(
      'span',
      `tri-ana-k tri-dir-${(projection.deltaPct ?? 0) > 0 ? 'up' : (projection.deltaPct ?? 0) < 0 ? 'down' : 'flat'}`,
      `${signedFixed(projection.deltaPct ?? 0, 1)}% · ${projection.horizonDays}d`,
    ),
  )
  if (projection.low != null && projection.high != null)
    cap.appendChild(
      el(
        'span',
        'tri-ana-k',
        `${context.formatter.text('80% range')} ${fmtTrendShort(context.formatter, sport, projection.low)}–${fmtTrendShort(context.formatter, sport, projection.high)}`,
      ),
    )
  cap.appendChild(el('span', 'tri-ana-k', `n ${projection.sampleSize}`))
  wrap.appendChild(cap)
  const note = el('div', 'tri-trend-note')
  note.append(
    el('span', 'tri-ana-k', context.formatter.text('training-derived LT2 proxy')),
    buildMethod(context.formatter, projection.method, projection.sampleSize),
  )
  wrap.appendChild(note)
  return wrap
}

export const buildLactateThreshold = (data: Analytics, context: TriathlonContext): HTMLElement => {
  const block = el('div', 'tri-ana-lactate')
  block.appendChild(anaTitle(context.formatter, 'lactate threshold projection', 'lactate'))
  const heartRate = data.engine.lactateThreshold.heartRate
  if (heartRate) {
    const cap = el('div', 'tri-elev-cap')
    cap.append(
      markGloss(el('span', 'tri-ana-k', `LTHR ${heartRate.value} ${heartRate.unit}`), 'lactate'),
      el('span', 'tri-ana-k', context.formatter.text('declared heart-rate anchor')),
    )
    block.appendChild(cap)
  }
  for (const sport of ['swim', 'bike', 'run'] as Sport[])
    block.appendChild(buildLactateThresholdPanel(data, sport, context))
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
