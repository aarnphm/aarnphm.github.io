import type { Analytics } from '../../../../plugins/stores/analytics'
import type { Sport } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import type { RaceLegSplit } from '../shared'
import { clock } from '../../../../util/triathlon-card'
import { KM_TO_MI } from '../../../../util/triathlon-card'
import { el } from '../../runtime/dom'
import { anaTitle } from '../shared'
import { buildIconLeg } from '../shared'
import { bySport } from '../shared'
import { clampN } from '../shared'
import { fmtSpeedKmh } from '../shared'
import { hms } from '../shared'
import { markGloss } from '../shared'
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
