import type { PaceDayState } from '../../../util/pace-features'
import type { PaceLegSpec } from '../../../util/pace-features'
import type { PaceSport } from '../../../util/pace-features'
import type { TriathlonContext } from '../runtime/context'
import type { TriathlonFormatter } from '../runtime/formatter'
import type { PredCompareKey } from './pace-forecast-model'
import type { PaceRuntime } from './pace-runtime'
import { start } from '../../../functional'
import { isPaceSport } from '../../../util/pace-features'
import { PaceForecaster } from '../../../util/pace-forecast'
import { Z80 } from '../../../util/pace-forecast'
import { buildIcon } from '../activity/primitives'
import { hms } from '../analytics/shared'
import { raceLegPace } from '../analytics/shared'
import { el } from '../runtime/dom'
import { buildPredDatePicker } from './date-picker'
import {
  initialPaceForecastModel,
  updatePaceForecast,
  type PaceForecastEffect,
  type PaceForecastMessage,
  type PaceForecastModel,
} from './pace-forecast-model'

export const PRED_SPORTS: { sport: PaceSport; dists: { km: number; label: string }[] }[] = [
  {
    sport: 'swim',
    dists: [
      { km: 0.75, label: '750m' },
      { km: 1.5, label: '1.5K' },
      { km: 1.9, label: '1.9K' },
      { km: 3.8, label: '3.8K' },
    ],
  },
  {
    sport: 'bike',
    dists: [
      { km: 20, label: '20K' },
      { km: 40, label: '40K' },
      { km: 90, label: '90K' },
      { km: 180, label: '180K' },
    ],
  },
  {
    sport: 'run',
    dists: [
      { km: 5, label: '5K' },
      { km: 10, label: '10K' },
      { km: 21.0975, label: 'half' },
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

export const syncPredDateControl = (
  formatter: TriathlonFormatter,
  block: HTMLElement,
  f: PaceForecaster,
): void => {
  const trigger = block.querySelector<HTMLButtonElement>('.tri-pred-date')
  const text = block.querySelector<HTMLElement>('.tri-pred-date-text')
  const bounds = f.dayBounds()
  if (!trigger || !text || !bounds) return
  const selected = block.dataset.compareDate
  const fallback = f.dayStateAgo(30)?.date ?? bounds.min
  const date = selected && selected >= bounds.min && selected <= bounds.max ? selected : fallback
  block.dataset.compareMin = bounds.min
  block.dataset.compareMax = bounds.max
  block.dataset.compareDate = date
  trigger.dataset.value = date
  text.textContent = formatter.longDate(date)
  const panel = block.querySelector<HTMLElement>('.tri-pred-calendar')
  if (panel?.matches(':popover-open')) panel.dispatchEvent(new CustomEvent('tri:date-render'))
}

export const predComparison = (
  formatter: TriathlonFormatter,
  f: PaceForecaster,
  block: HTMLElement,
): PredComparison => {
  const key = predCompareKey(block)
  if (key === 'custom') {
    const day = f.dayStateOnOrBefore(block.dataset.compareDate ?? '')
    return {
      day,
      label: day?.date
        ? `vs ${formatter.shortDate(day.date)}`
        : formatter.text('custom date missing'),
    }
  }
  const days = predCompareOption(key).days ?? 30
  const day = f.dayStateAgo(days)
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
  if (preserveVisual || stale) {
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

export const failPredCard = (card: HTMLElement): void => {
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
  card.dataset.tipD = 'model unavailable'
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
  const day = f.day
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

export const buildDistancePredictor = (
  runtime: PaceRuntime,
  context: TriathlonContext,
): { element: HTMLElement; mount: () => () => void } => {
  const block = el('div', 'tri-pred')
  block.dataset.compareMode = PRED_DEFAULT_COMPARE
  const head = el('div', 'tri-pred-head')
  const headMain = el('div', 'tri-pred-head-main')
  headMain.append(el('span', 'tri-pred-title', 'pace predictor'))
  const controls = el('div', 'tri-pred-controls')
  const compare = el('div', 'tri-pred-compare', undefined, {
    role: 'tablist',
    'aria-label': 'comparison range',
  })
  const updateCompareControls = (): void => {
    const mode = predCompareKey(block)
    for (const button of compare.querySelectorAll<HTMLButtonElement>('.tri-pred-compare-btn')) {
      const active = button.dataset.compareMode === mode
      button.classList.toggle('tri-pred-compare-btn--on', active)
      button.setAttribute('aria-selected', String(active))
    }
    if (runtime.forecaster?.ready) syncPredDateControl(context.formatter, block, runtime.forecaster)
  }
  head.append(headMain, controls)
  block.appendChild(head)
  const tabs = el('div', 'tri-pred-tabs', undefined, {
    role: 'group',
    'aria-label': 'predictor sport',
  })
  const grid = el('div', 'tri-pred-grid')
  let dispatchMessage: (message: PaceForecastMessage) => void = () => {}
  const renderSport = (sport: PaceSport): void => {
    const cfg = PRED_SPORTS.find(s => s.sport === sport)
    if (!cfg) return
    let cards = Array.from(grid.querySelectorAll<HTMLElement>('.tri-pred-card'))
    if (cards.length !== cfg.dists.length) {
      cards = cfg.dists.map(d => {
        const card = el('div', 'tri-pred-card')
        const track = el('div', 'tri-pred-bar-track')
        const result = el('span', 'tri-pred-result')
        result.append(el('span', 'tri-pred-time', '—'), el('span', 'tri-pred-pace'))
        track.append(el('div', 'tri-pred-bar-range'), el('div', 'tri-pred-bar'))
        card.append(
          el('span', 'tri-pred-badge', d.label),
          track,
          result,
          el('span', 'tri-pred-delta'),
        )
        return card
      })
      grid.replaceChildren(...cards)
    }
    cfg.dists.forEach((d, i) => {
      const card = cards[i]
      card.dataset.km = String(d.km)
      card.dataset.sport = sport
      card.dataset.label = d.label
      if (card.dataset.filled) card.dataset.stale = '1'
      delete card.dataset.filled
      delete card.dataset.pending
      delete card.dataset.error
      if (!card.dataset.stale) {
        delete card.dataset.tipD
        delete card.dataset.tipH
      }
    })
    if (!runtime.forecaster?.ready) {
      if (runtime.unavailable) for (const card of cards) failPredCard(card)
    }
  }
  for (const option of PRED_COMPARE_OPTIONS) {
    const button = el(
      'button',
      `tri-pred-compare-btn${option.key === PRED_DEFAULT_COMPARE ? ' tri-pred-compare-btn--on' : ''}`,
      option.label,
      {
        type: 'button',
        role: 'tab',
        'aria-selected': String(option.key === PRED_DEFAULT_COMPARE),
        'data-compare-mode': option.key,
      },
    )
    compare.appendChild(button)
  }
  const activateCustomCompare = (): void => {
    dispatchMessage({ type: 'select-comparison', comparison: 'custom' })
  }
  const selectPredDate = (date: string): void => {
    dispatchMessage({ type: 'select-date', date })
  }
  const clearPredDate = (): void => {
    dispatchMessage({ type: 'clear-date' })
  }
  const datePicker = buildPredDatePicker(
    context.formatter,
    block,
    activateCustomCompare,
    selectPredDate,
    clearPredDate,
  )
  for (const s of PRED_SPORTS) {
    const on = s.sport === 'run'
    const tab = el(
      'button',
      `tri-pred-tab tri-pred-tab--${s.sport}${on ? ' tri-pred-tab--on' : ''}`,
      undefined,
      {
        type: 'button',
        'aria-label': s.sport,
        'aria-pressed': String(on),
        'aria-selected': String(on),
        title: s.sport,
        'data-sport': s.sport,
      },
    )
    tab.appendChild(buildIcon(context.presentation, s.sport))
    tabs.appendChild(tab)
  }
  controls.append(tabs, compare, datePicker.wrap)
  const axis = el('div', 'tri-pred-axis')
  axis.append(
    el('span', 'tri-pred-axis-pad'),
    el('div', 'tri-pred-axis-track'),
    el('span', 'tri-pred-axis-end'),
  )
  block.append(grid, axis)
  renderSport('run')
  updateCompareControls()

  const onCompareClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const button = target.closest<HTMLButtonElement>('.tri-pred-compare-btn[data-compare-mode]')
    if (!button || !compare.contains(button)) return
    const comparison = button.dataset.compareMode
    if (
      comparison !== '7' &&
      comparison !== '14' &&
      comparison !== '30' &&
      comparison !== '60' &&
      comparison !== 'custom'
    )
      return
    dispatchMessage({ type: 'select-comparison', comparison })
  }
  const onSportClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const button = target.closest<HTMLButtonElement>('.tri-pred-tab[data-sport]')
    if (!button || !tabs.contains(button)) return
    const sport = button.dataset.sport
    if (!isPaceSport(sport)) return
    dispatchMessage({ type: 'select-sport', sport })
  }
  return {
    element: block,
    mount: () => {
      const program = start<PaceForecastModel, PaceForecastMessage, PaceForecastEffect>({
        init: () => ({
          model: initialPaceForecastModel(),
          effects: [{ type: 'render', generation: 0 }],
        }),
        reduce: updatePaceForecast,
        effects: (effect, { dispatch, retrieve }) => {
          const model = retrieve()
          block.dataset.compareMode = model.comparison
          if (model.comparisonDate) block.dataset.compareDate = model.comparisonDate
          else delete block.dataset.compareDate
          block.dataset.predGeneration = String(effect.generation)
          for (const button of tabs.querySelectorAll<HTMLElement>('.tri-pred-tab')) {
            const active = button.getAttribute('aria-label') === model.sport
            button.classList.toggle('tri-pred-tab--on', active)
            button.setAttribute('aria-pressed', String(active))
            button.setAttribute('aria-selected', String(active))
          }
          updateCompareControls()
          if (model.comparison === 'custom' && !model.comparisonDate && block.dataset.compareDate) {
            dispatch({ type: 'select-date', date: block.dataset.compareDate })
            return
          }
          renderSport(model.sport)
          if (!runtime.forecaster?.ready) return
          void fillDistancePredictor(grid, runtime, context)
        },
      })
      dispatchMessage = program.dispatch
      compare.addEventListener('click', onCompareClick)
      tabs.addEventListener('click', onSportClick)
      const datePickerCleanup = datePicker.mount()
      return () => {
        dispatchMessage = () => {}
        compare.removeEventListener('click', onCompareClick)
        tabs.removeEventListener('click', onSportClick)
        datePickerCleanup()
        program.stop()
      }
    },
  }
}

export const wirePredTip = (root: HTMLElement): (() => void) => {
  document.body.querySelector('.tri-pred-tip')?.remove()
  const tip = el('div', 'tri-gloss tri-pred-tip')
  tip.setAttribute('role', 'tooltip')
  document.body.appendChild(tip)
  let cur: HTMLElement | null = null
  const move = (e: MouseEvent): void => {
    const card = (e.target as HTMLElement | null)?.closest<HTMLElement>('.tri-pred-card')
    if (!card?.dataset.tipD) {
      cur = null
      tip.classList.remove('tri-gloss--on')
      return
    }
    if (card !== cur) {
      cur = card
      tip.replaceChildren(
        el('span', 'tri-gloss-h', card.dataset.tipH ?? ''),
        el('span', 'tri-gloss-def', card.dataset.tipD ?? ''),
      )
      tip.classList.add('tri-gloss--on')
    }
    const pr = tip.getBoundingClientRect()
    const left =
      e.clientX + 14 + pr.width > window.innerWidth - 8 ? e.clientX - 14 - pr.width : e.clientX + 14
    const top =
      e.clientY + 14 + pr.height > window.innerHeight - 8
        ? e.clientY - 14 - pr.height
        : e.clientY + 14
    tip.style.left = `${Math.max(8, left).toFixed(0)}px`
    tip.style.top = `${Math.max(8, top).toFixed(0)}px`
  }
  const leave = (): void => {
    cur = null
    tip.classList.remove('tri-gloss--on')
  }
  root.addEventListener('mousemove', move)
  root.addEventListener('mouseleave', leave)
  return () => {
    root.removeEventListener('mousemove', move)
    root.removeEventListener('mouseleave', leave)
    tip.remove()
  }
}

export const paceModelBaseCandidates = (): string[] => {
  const bases = [location.origin]
  const port = Number(location.port)
  if (
    (location.hostname === 'localhost' || location.hostname === '127.0.0.1') &&
    Number.isInteger(port) &&
    port > 0 &&
    port + 707 <= 65535
  )
    bases.push(`${location.protocol}//${location.hostname}:${port + 707}`)
  return Array.from(new Set(bases))
}

export const initPaceForecaster = async (forecaster: PaceForecaster): Promise<boolean> => {
  for (const base of paceModelBaseCandidates())
    if (await forecaster.init(base, 'pace', '/static/triathlon/data.jsonl')) return true
  return false
}

export const markDistancePredictorUnavailable = (root: HTMLElement): void => {
  for (const block of root.querySelectorAll<HTMLElement>('.tri-pred')) {
    for (const card of block.querySelectorAll<HTMLElement>('.tri-pred-card'))
      if (!card.dataset.filled) failPredCard(card)
  }
}

export const setupPaceForecast = (
  root: HTMLElement,
  context: TriathlonContext,
): (() => void) | null => {
  if (!root.dataset.analyticsPath && !root.querySelector('.tri-calc')) return null
  let worker: Worker
  try {
    worker = new Worker(new URL('/pace.worker.js', import.meta.url), { type: 'module' })
  } catch {
    return null
  }
  const forecaster = new PaceForecaster(worker)
  context.pace.forecaster = forecaster
  context.pace.unavailable = false
  initPaceForecaster(forecaster)
    .then(ok => {
      if (context.pace.forecaster !== forecaster) return
      context.pace.unavailable = !ok
      if (!ok) {
        markDistancePredictorUnavailable(root)
        return
      }
      void fillDistancePredictor(root, context.pace, context)
    })
    .catch(() => {
      if (context.pace.forecaster !== forecaster) return
      context.pace.unavailable = true
      markDistancePredictorUnavailable(root)
    })
  const tipCleanup = wirePredTip(root)
  return () => {
    tipCleanup()
    forecaster.dispose()
    if (context.pace.forecaster === forecaster) context.pace.forecaster = null
  }
}
