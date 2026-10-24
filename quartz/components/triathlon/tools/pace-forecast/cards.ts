import type { PaceDayState, PaceLegSpec, PaceSport } from '../../../../util/pace-features'
import type { PaceForecaster } from '../../../../util/pace-forecast'
import type { TriathlonContext } from '../../runtime/context'
import type { TriathlonFormatter } from '../../runtime/formatter'
import type { PredCompareKey } from '../pace-forecast-model'
import type { PaceRuntime } from '../pace-runtime'
import { isPaceSport } from '../../../../util/pace-features'
import { Z80 } from '../../../../util/pace-forecast'
import { DATE_PICKER_RENDER_EVENT } from '../../../controls/date-picker'
import { hms, raceLegPace } from '../../analytics/shared'
import { el } from '../../runtime/dom'
import {
  paceForecastBounds,
  paceForecastComparisonDate,
  type PaceForecastBounds,
} from '../pace-forecast-model'

export const PRED_SPORTS: { sport: PaceSport; dists: { km: number; label: string }[] }[] = [
  {
    sport: 'swim',
    dists: [
      { km: 0.75, label: '750m' },
      { km: 1.5, label: '1.5K' },
      { km: 1, label: '1K' },
      { km: 1.9, label: '1.9K' },
      { km: 2, label: '2K' },
      { km: 3.8, label: '3.8K' },
    ],
  },
  {
    sport: 'bike',
    dists: [
      { km: 20, label: '20K' },
      { km: 40, label: '40K' },
      { km: 90, label: '90K' },
      { km: 80, label: '80K' },
      { km: 180, label: '180K' },
    ],
  },
  {
    sport: 'run',
    dists: [
      { km: 5, label: '5K' },
      { km: 10, label: '10K' },
      { km: 9, label: '9K' },
      { km: 21.0975, label: 'half' },
      { km: 18, label: '18K' },
      { km: 42.195, label: 'marathon' },
    ],
  },
]

export const PRED_COMPARE_OPTIONS = [
  { key: '7', label: '7d', days: 7 },
  { key: '14', label: '14d', days: 14 },
  { key: '30', label: '30d', days: 30 },
  { key: '60', label: '60d', days: 60 },
  { key: 'custom', label: 'custom', days: null },
] as const

export interface PredComparison {
  day: PaceDayState | null
  label: string
}

export interface PredResult {
  card: HTMLElement
  nowSec: number
  fastSec: number
  slowSec: number
  paceText: string
  delta: number | null
  compareLabel: string
}

export const PRED_AXIS_FRACS = [0, 0.25, 0.5, 0.75, 1]

export const PRED_DEFAULT_COMPARE: PredCompareKey = '7'

export const isPredCompareKey = (value: string | undefined): value is PredCompareKey =>
  PRED_COMPARE_OPTIONS.some(option => option.key === value)

export const predCompareKey = (block: HTMLElement): PredCompareKey => {
  const key = block.dataset.compareMode
  return isPredCompareKey(key) ? key : PRED_DEFAULT_COMPARE
}

export const predCompareOption = (key: PredCompareKey): (typeof PRED_COMPARE_OPTIONS)[number] =>
  PRED_COMPARE_OPTIONS.find(option => option.key === key) ?? PRED_COMPARE_OPTIONS[0]

export const predHistoryBounds = (
  block: HTMLElement,
  f: PaceForecaster,
): PaceForecastBounds | null => {
  const min = block.dataset.historyMin
  const max = block.dataset.historyMax
  return min && max ? paceForecastBounds(f.dayBounds(), { min, max }) : null
}

export const predDayAt = (
  f: PaceForecaster,
  bounds: PaceForecastBounds | null,
  date: string | null,
): PaceDayState | null => {
  if (!bounds || !date || date < bounds.min || date > bounds.max) return null
  const day = f.dayStateOnOrBefore(date)
  return day && day.date >= bounds.min && day.date <= bounds.max ? day : null
}

export const syncPredDateControl = (
  formatter: TriathlonFormatter,
  block: HTMLElement,
  f: PaceForecaster,
): void => {
  const trigger = block.querySelector<HTMLButtonElement>('.g-datepicker-trigger')
  const text = block.querySelector<HTMLElement>('.g-datepicker-text')
  const bounds = predHistoryBounds(block, f)
  if (!trigger || !text) return
  trigger.disabled = bounds == null
  if (!bounds) {
    text.textContent = formatter.text('no data')
    return
  }
  const selected = block.dataset.compareDate
  const fallback = predDayAt(f, bounds, paceForecastComparisonDate(bounds, 30))?.date ?? bounds.min
  const date = selected && selected >= bounds.min && selected <= bounds.max ? selected : fallback
  block.dataset.compareMin = bounds.min
  block.dataset.compareMax = bounds.max
  block.dataset.compareDate = date
  trigger.dataset.value = date
  text.textContent = formatter.longDate(date)
  const panel = block.querySelector<HTMLElement>('.g-datepicker-panel')
  if (panel?.matches(':popover-open'))
    panel.dispatchEvent(new CustomEvent(DATE_PICKER_RENDER_EVENT))
}

export const predComparison = (
  formatter: TriathlonFormatter,
  f: PaceForecaster,
  block: HTMLElement,
): PredComparison => {
  const key = predCompareKey(block)
  const bounds = predHistoryBounds(block, f)
  if (key === 'custom') {
    const day = predDayAt(f, bounds, block.dataset.compareDate ?? null)
    return {
      day,
      label: day?.date
        ? `vs ${formatter.shortDate(day.date)}`
        : formatter.text('custom date missing'),
    }
  }
  const days = predCompareOption(key).days ?? 30
  const day = predDayAt(f, bounds, bounds ? paceForecastComparisonDate(bounds, days) : null)
  return {
    day,
    label: day?.date
      ? `vs ${days}d (${formatter.shortDate(day.date)})`
      : `vs ${days}d (${formatter.text('no data')})`,
  }
}

export const renderPredAxis = (track: HTMLElement, maxSec: number): void => {
  const ticks = track.querySelectorAll<HTMLElement>('.tri-pred-axis-tick')
  if (ticks.length === PRED_AXIS_FRACS.length) {
    PRED_AXIS_FRACS.forEach((fr, i) => (ticks[i].textContent = hms(maxSec * fr)))
    return
  }
  track.replaceChildren(
    ...PRED_AXIS_FRACS.map(fr => {
      const tick = el('span', 'tri-pred-axis-tick', hms(maxSec * fr))
      tick.style.left = `${fr * 100}%`
      if (fr === 0) tick.dataset.edge = 'start'
      else if (fr === 1) tick.dataset.edge = 'end'
      return tick
    }),
  )
}

export const resetPredCard = (card: HTMLElement, preserveVisual: boolean): void => {
  const stale = card.dataset.stale === '1'
  const paceEl = card.querySelector('.tri-pred-pace')
  if (card.dataset.sport !== 'swim' && paceEl) paceEl.textContent = ''
  delete card.dataset.filled
  delete card.dataset.error
  card.dataset.pending = '1'
  if (stale || (preserveVisual && card.dataset.tipD)) {
    card.dataset.tipH = card.dataset.label ?? ''
    card.dataset.tipD = 'updating'
    return
  }
  delete card.dataset.tipD
  delete card.dataset.tipH
  const bar = card.querySelector<HTMLElement>('.tri-pred-bar')
  if (bar) bar.style.width = '0%'
  const range = card.querySelector<HTMLElement>('.tri-pred-bar-range')
  if (range) {
    range.style.left = '0%'
    range.style.width = '0%'
  }
  const timeEl = card.querySelector('.tri-pred-time')
  if (timeEl) timeEl.textContent = '…'
  if (paceEl) paceEl.textContent = ''
  const deltaEl = card.querySelector('.tri-pred-delta')
  if (deltaEl) {
    deltaEl.textContent = ''
    deltaEl.classList.remove('tri-pred-delta--up', 'tri-pred-delta--down', 'tri-pred-delta--na')
  }
}

export const failPredCard = (card: HTMLElement, reason = 'model unavailable'): void => {
  delete card.dataset.pending
  delete card.dataset.stale
  card.dataset.error = '1'
  const timeEl = card.querySelector('.tri-pred-time')
  if (timeEl) timeEl.textContent = '—'
  const paceEl = card.querySelector('.tri-pred-pace')
  if (paceEl) paceEl.textContent = ''
  const deltaEl = card.querySelector('.tri-pred-delta')
  if (deltaEl) deltaEl.textContent = ''
  card.dataset.tipH = card.dataset.label ?? ''
  card.dataset.tipD = reason
}

export const applyPredResult = (r: PredResult, maxSec: number): void => {
  const pct = (s: number): number => (s / maxSec) * 100
  delete r.card.dataset.pending
  delete r.card.dataset.stale
  delete r.card.dataset.error
  r.card.dataset.filled = '1'
  const badge = r.card.querySelector('.tri-pred-badge')
  if (badge && r.card.dataset.label) badge.textContent = r.card.dataset.label
  const bar = r.card.querySelector<HTMLElement>('.tri-pred-bar')
  if (bar) bar.style.width = `${Math.max(2, pct(r.nowSec))}%`
  const range = r.card.querySelector<HTMLElement>('.tri-pred-bar-range')
  if (range) {
    range.style.left = `${pct(r.fastSec)}%`
    range.style.width = `${Math.max(0, pct(r.slowSec) - pct(r.fastSec))}%`
  }
  const timeEl = r.card.querySelector('.tri-pred-time')
  if (timeEl) timeEl.textContent = hms(r.nowSec)
  const paceEl = r.card.querySelector('.tri-pred-pace')
  if (paceEl) paceEl.textContent = r.paceText
  let tip = `${hms(r.nowSec)}${r.paceText ? ` · ${r.paceText}` : ''} · ${hms(r.fastSec)}–${hms(r.slowSec)} band`
  const deltaEl = r.card.querySelector('.tri-pred-delta')
  if (deltaEl) {
    deltaEl.classList.remove('tri-pred-delta--up', 'tri-pred-delta--down', 'tri-pred-delta--na')
    if (r.delta == null) {
      deltaEl.textContent = '—'
      deltaEl.classList.add('tri-pred-delta--na')
      tip += ` · ${r.compareLabel}`
    } else if (Math.abs(r.delta) >= 1) {
      const faster = r.delta < 0
      deltaEl.textContent = `${faster ? '▾' : '▴'}${hms(Math.abs(r.delta))}`
      deltaEl.classList.add(faster ? 'tri-pred-delta--up' : 'tri-pred-delta--down')
      tip += ` · ${faster ? '▾' : '▴'}${hms(Math.abs(r.delta))} ${r.compareLabel}`
    } else {
      deltaEl.textContent = ''
      tip += ` · ${r.compareLabel}`
    }
  }
  r.card.dataset.tipH = r.card.dataset.label ?? ''
  r.card.dataset.tipD = tip
}

export const inferPredCard = async (
  context: TriathlonContext,
  f: PaceForecaster,
  day: PaceDayState,
  comparison: PredComparison,
  card: HTMLElement,
): Promise<PredResult | null> => {
  const km = Number(card.dataset.km)
  const sport = card.dataset.sport
  if (!km || !isPaceSport(sport)) return null
  const leg: PaceLegSpec = { sport, distanceKm: km, elevationM: 0, tempC: null, windKph: null }
  const [now, then] = await Promise.all([
    f.forecastLegAt(day, leg),
    comparison.day ? f.forecastLegAt(comparison.day, leg) : Promise.resolve(null),
  ])
  if (!now || now.mu <= 0) return null
  const meters = km * 1000
  const nowSec = meters / now.mu
  const timeSd = (meters / (now.mu * now.mu)) * now.sigma
  const fastSec = Math.max(0, nowSec - Z80 * timeSd)
  const slowSec = nowSec + Z80 * timeSd
  const delta = then && then.mu > 0 ? nowSec - meters / then.mu : null
  const paceText = raceLegPace(context.formatter, { sport, legKm: km, splitS: nowSec })
  return { card, nowSec, fastSec, slowSec, paceText, delta, compareLabel: comparison.label }
}

export async function fillDistancePredictor(
  scope: ParentNode,
  runtime: PaceRuntime,
  context: TriathlonContext,
): Promise<void> {
  const f = runtime.forecaster
  const block =
    scope instanceof HTMLElement
      ? (scope.closest<HTMLElement>('.tri-pred') ?? scope.querySelector<HTMLElement>('.tri-pred'))
      : null
  if (!block) return
  if (!f?.ready || !f.day) return
  syncPredDateControl(context.formatter, block, f)
  const bounds = predHistoryBounds(block, f)
  const day = predDayAt(f, bounds, bounds?.max ?? null)
  if (!day) {
    for (const card of block.querySelectorAll<HTMLElement>('.tri-pred-card'))
      failPredCard(card, context.formatter.text('no data'))
    return
  }
  const comparison = predComparison(context.formatter, f, block)
  const cards = Array.from(scope.querySelectorAll<HTMLElement>('.tri-pred-card')).filter(
    c => !c.dataset.filled,
  )
  if (!cards.length) return
  const runId = String(++runtime.sequence)
  block.dataset.predRun = runId
  const hasStale = cards.some(card => card.dataset.stale === '1')
  if (!hasStale) block.querySelector<HTMLElement>('.tri-pred-axis-track')?.replaceChildren()
  for (const card of cards) resetPredCard(card, hasStale)
  const results: PredResult[] = []
  const render = (): void => {
    if (block.dataset.predRun !== runId) return
    const maxSec = Math.max(...results.map(r => r.slowSec), 1)
    for (const result of results) applyPredResult(result, maxSec)
    const axis = block.querySelector<HTMLElement>('.tri-pred-axis-track')
    if (axis && results.length) renderPredAxis(axis, maxSec)
  }
  render()
  await Promise.all(
    cards.map(async card => {
      const result = await inferPredCard(context, f, day, comparison, card)
      if (block.dataset.predRun !== runId) return
      if (result) results.push(result)
      else failPredCard(card)
      render()
    }),
  )
}
