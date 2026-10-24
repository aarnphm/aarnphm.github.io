import type { PaceSport } from '../../../../util/pace-features'
import type { TriathlonContext } from '../../runtime/context'
import type { PaceRuntime } from '../pace-runtime'
import { start } from '../../../../functional'
import { isPaceSport } from '../../../../util/pace-features'
import { PaceForecaster } from '../../../../util/pace-forecast'
import { buildDatePicker } from '../../../controls/date-picker'
import { buildIcon } from '../../activity/primitives'
import { rafEvent } from '../../analytics/scrub-primitives'
import { el, setText } from '../../runtime/dom'
import {
  initialPaceForecastModel,
  updatePaceForecast,
  type PaceForecastBounds,
  type PaceForecastEffect,
  type PaceForecastMessage,
  type PaceForecastModel,
} from '../pace-forecast-model'
import {
  failPredCard,
  fillDistancePredictor,
  PRED_COMPARE_OPTIONS,
  PRED_DEFAULT_COMPARE,
  PRED_SPORTS,
  predCompareKey,
  syncPredDateControl,
} from './cards'

export const buildDistancePredictor = (
  runtime: PaceRuntime,
  context: TriathlonContext,
  bounds: PaceForecastBounds,
): { element: HTMLElement; mount: () => () => void } => {
  const block = el('div', 'tri-pred')
  block.dataset.historyMin = bounds.min
  block.dataset.historyMax = bounds.max
  block.dataset.compareMin = bounds.min
  block.dataset.compareMax = bounds.max
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
  const makePredCard = (label: string): HTMLElement => {
    const card = el('div', 'tri-pred-card')
    const track = el('div', 'tri-pred-bar-track')
    const result = el('span', 'tri-pred-result')
    result.append(el('span', 'tri-pred-time', '—'), el('span', 'tri-pred-pace'))
    track.append(el('div', 'tri-pred-bar-range'), el('div', 'tri-pred-bar'))
    card.append(el('span', 'tri-pred-badge', label), track, result, el('span', 'tri-pred-delta'))
    return card
  }
  const renderSport = (sport: PaceSport): void => {
    const cfg = PRED_SPORTS.find(s => s.sport === sport)
    if (!cfg) return
    const cards = Array.from(grid.querySelectorAll<HTMLElement>('.tri-pred-card'))
    while (cards.length > cfg.dists.length) cards.pop()?.remove()
    while (cards.length < cfg.dists.length) {
      const card = makePredCard(cfg.dists[cards.length].label)
      grid.appendChild(card)
      cards.push(card)
    }
    cfg.dists.forEach((d, i) => {
      const card = cards[i]
      const badge = card.querySelector('.tri-pred-badge')
      if (badge) badge.textContent = d.label
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
  const datePicker = buildDatePicker({
    id: 'tri-pred-comparison-date',
    label: context.formatter.text('comparison date'),
    labels: context.formatter,
    selected: () => block.dataset.compareDate,
    min: () => block.dataset.compareMin,
    max: () => block.dataset.compareMax,
    onOpen: activateCustomCompare,
    onSelect: selectPredDate,
    onClear: clearPredDate,
  })
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
  const headSpan = el('span', 'tri-gloss-h')
  const defSpan = el('span', 'tri-gloss-def')
  tip.append(headSpan, defSpan)
  document.body.appendChild(tip)
  let cur: HTMLElement | null = null
  const sizeCache = new WeakMap<HTMLElement, { width: number; height: number }>()
  const rafMove = rafEvent((e: MouseEvent): void => {
    const card = (e.target as HTMLElement | null)?.closest<HTMLElement>('.tri-pred-card')
    if (!card?.dataset.tipD) {
      cur = null
      tip.classList.remove('tri-gloss--on')
      return
    }
    if (card !== cur) {
      cur = card
      setText(headSpan, card.dataset.tipH ?? '')
      setText(defSpan, card.dataset.tipD ?? '')
      tip.classList.add('tri-gloss--on')
    }
    let size = sizeCache.get(card)
    if (!size || size.width <= 0) {
      const pr = tip.getBoundingClientRect()
      size = { width: pr.width, height: pr.height }
      if (size.width > 0) sizeCache.set(card, size)
    }
    const left =
      e.clientX + 14 + size.width > window.innerWidth - 8
        ? e.clientX - 14 - size.width
        : e.clientX + 14
    const top =
      e.clientY + 14 + size.height > window.innerHeight - 8
        ? e.clientY - 14 - size.height
        : e.clientY + 14
    tip.style.left = `${Math.max(8, left).toFixed(0)}px`
    tip.style.top = `${Math.max(8, top).toFixed(0)}px`
  })
  const leave = (): void => {
    rafMove.cancel()
    cur = null
    tip.classList.remove('tri-gloss--on')
  }
  root.addEventListener('mousemove', rafMove.onEvent)
  root.addEventListener('mouseleave', leave)
  return () => {
    rafMove.cancel()
    root.removeEventListener('mousemove', rafMove.onEvent)
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
  worker.onerror = () => forecaster.dispose()
  context.pace.forecaster = forecaster
  context.pace.unavailable = false
  context.pace.loading = initPaceForecaster(forecaster)
    .then(ok => {
      if (context.pace.forecaster !== forecaster) return false
      context.pace.unavailable = !ok
      if (!ok) {
        markDistancePredictorUnavailable(root)
        return false
      }
      void fillDistancePredictor(root, context.pace, context)
      return true
    })
    .catch(() => {
      if (context.pace.forecaster !== forecaster) return false
      context.pace.unavailable = true
      markDistancePredictorUnavailable(root)
      return false
    })
  const tipCleanup = wirePredTip(root)
  return () => {
    tipCleanup()
    forecaster.dispose()
    if (context.pace.forecaster === forecaster) {
      context.pace.forecaster = null
      context.pace.loading = null
    }
  }
}
