import type {
  RaceProjectionBaseline,
  RaceProjectionSpec,
  RaceScenario,
  RaceTimeProjection,
} from '../../../util/race-projection'
import type { TriathlonContext } from '../runtime/context'
import { shiftIsoDay } from '../../../util/local-date'
import {
  boundedRaceDay,
  parseRaceProjectionSpec,
  predictRaceTime,
  projectRaceDay,
  raceProjectionBaseline,
  raceTrendDates,
} from '../../../util/race-projection'
import { applyI18n, el, svg } from '../runtime/dom'
import { calendarToday } from './display'

const minuteTime = (seconds: number): string => {
  const minutes = Math.round(seconds / 60)
  return minutes < 60
    ? `${minutes} min`
    : `${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, '0')}`
}
const finiteInput = (input: HTMLInputElement | null): number | null => {
  if (!input || input.value.trim() === '' || !input.validity.valid) return null
  return Number.isFinite(input.valueAsNumber) ? input.valueAsNumber : null
}
const inputFor = (panel: HTMLElement, key: string): HTMLInputElement | null =>
  panel.querySelector<HTMLInputElement>(`[data-race-scenario="${key}"]`)

const adjustedSpec = (
  row: HTMLElement,
  original: RaceProjectionSpec,
): RaceProjectionSpec | null => {
  const extraInput = row.querySelector<HTMLInputElement>('[data-race-extra]')
  const extraMinutes = finiteInput(extraInput)
  const division = row.querySelector<HTMLSelectElement>('[data-race-division]')?.value
  if (original.kind !== 'hyrox' && extraMinutes === null) return null
  if (original.kind === 'hyrox' && division && extraInput?.value && extraMinutes === null)
    return null
  const penaltyInput = row.querySelector<HTMLInputElement>('[data-race-run-penalty]')
  const penalty = penaltyInput ? finiteInput(penaltyInput) : 0
  if (penalty === null) return null
  const legs = []
  for (const leg of original.legs) {
    const intensity = finiteInput(
      row.querySelector<HTMLInputElement>(`[data-race-intensity="${leg.sport}"]`),
    )
    if (intensity === null) return null
    legs.push({ ...leg, intensity })
  }
  return {
    ...original,
    legs,
    extraMinutes: original.kind === 'hyrox' && !division ? null : extraMinutes,
    runPenaltyPct: penalty,
  }
}

const projectionTable = (
  className: string,
  rows: { label: string; values: string[] }[],
): HTMLElement => {
  const table = el('table', `tri-calendar-projection-table ${className}`)
  const body = el('tbody')
  for (const { label, values } of rows) {
    const row = el('tr')
    row.append(el('th', undefined, label, { scope: 'row' }))
    for (const value of values) row.append(el('td', undefined, value))
    body.append(row)
  }
  table.append(body)
  return table
}

const trendTable = (
  context: TriathlonContext,
  points: { date: string; prediction: RaceTimeProjection }[],
): HTMLElement | null => {
  if (points.length < 2) return null
  const first = points[0]
  const last = points[points.length - 1]
  const values = points.map(point => point.prediction.midSec)
  const low = Math.min(...values)
  const span = Math.max(60, Math.max(...values) - low)
  const coordinates = values.map(
    (value, index) =>
      `${8 + (index / (values.length - 1)) * 124},${32 - ((value - low) / span) * 24}`,
  )
  const delta = Math.round((last.prediction.midSec - first.prediction.midSec) / 60)
  const description = `${context.formatter.text('recent model trend')} · ${context.formatter.shortDate(first.date)} → ${context.formatter.shortDate(last.date)} · ${delta > 0 ? '+' : ''}${delta} min`
  const chart = svg('svg', { viewBox: '0 0 140 40', role: 'img', 'aria-label': description })
  chart.append(svg('polyline', { points: coordinates.join(' '), fill: 'none' }))
  const table = projectionTable('tri-calendar-race-trend', [
    {
      label: context.formatter.text('recent model trend'),
      values: [
        `${delta > 0 ? '+' : ''}${delta} min`,
        `${context.formatter.shortDate(first.date)} → ${context.formatter.shortDate(last.date)}`,
      ],
    },
  ])
  const chartCell = el('td')
  chartCell.append(chart)
  table.querySelector('tr')?.append(chartCell)
  return table
}

export const mountRaceProjections = (
  calendar: HTMLElement,
  context: TriathlonContext,
): { select: () => void; dispose: () => void } => {
  const slot = calendar.querySelector<HTMLElement>('[data-race-projection-slot]')
  const path = context.root?.dataset.analyticsPath
  if (!slot) return { select: () => {}, dispose: () => {} }
  const savedPanels = new Map<string, HTMLElement>()
  let panel: HTMLElement | null = null
  let rows: { row: HTMLElement; spec: RaceProjectionSpec; output: HTMLElement }[] = []
  let basis: HTMLElement | null = null
  let baseline: RaceProjectionBaseline | null = null
  let active = true
  let failed = false
  let generation = 0
  let timer: number | undefined
  const text = (key: string): string => context.formatter.text(key)
  const message = (output: HTMLElement, key: string): void => {
    output.replaceChildren(el('p', undefined, text(key)))
    output.removeAttribute('aria-busy')
    for (const key of ['projectionTime', 'projectionTss', 'projectionCurrent', 'projectionTrend'])
      delete output.dataset[key]
  }
  const unavailable = (): void => {
    failed = true
    if (basis) basis.textContent = text('model unavailable')
    for (const { output } of rows) message(output, 'model unavailable')
  }
  const render = async (): Promise<void> => {
    const forecaster = context.pace.forecaster
    if (!active || !baseline || !panel || !forecaster?.ready) return
    const run = ++generation
    const weeklyLoad = finiteInput(inputFor(panel, 'weeklyLoad'))
    const taperDays = finiteInput(inputFor(panel, 'taperDays'))
    const paceGainPct = finiteInput(inputFor(panel, 'paceGainPct'))
    if (weeklyLoad === null || taperDays === null || paceGainPct === null) {
      for (const { output } of rows) message(output, 'Enter a value within the displayed limits.')
      return
    }
    const scenario: RaceScenario = { weeklyLoad, taperDays, paceGainPct }
    const currentBaseline = baseline
    if (basis)
      basis.textContent = `${text('training through')} ${context.formatter.longDate(baseline.day.date)} · ${text('computed weekly load')} ${Math.round(baseline.weeklyLoad)} · ${text('pace model')} v${forecaster.modelVersion ?? '?'} · ${text('current thresholds held fixed')}`
    await Promise.all(
      rows.map(async ({ row, spec: original, output }) => {
        if (!original.date) {
          message(output, 'date pending')
          return
        }
        if (original.date < calendarToday() || original.date <= currentBaseline.day.date) {
          message(output, 'past event · see race results')
          return
        }
        const spec = adjustedSpec(row, original)
        if (!spec) {
          message(output, 'Enter a value within the displayed limits.')
          return
        }
        output.setAttribute('aria-busy', 'true')
        const horizon = Math.round(
          (Date.parse(`${spec.date}T00:00:00Z`) -
            Date.parse(`${currentBaseline.day.date}T00:00:00Z`)) /
            86_400_000,
        )
        const projected = projectRaceDay(currentBaseline, original.date, scenario)
        const bounded = boundedRaceDay(currentBaseline, projected)
        const enduranceExtrapolation = spec.legs.some(
          leg => leg.distanceKm > currentBaseline.longest[leg.sport] * 1.5,
        )
        const history: { date: string; prediction: RaceTimeProjection }[] = []
        for (const date of raceTrendDates(currentBaseline.day.date)) {
          if (!active || run !== generation) return
          const day = forecaster.dayStateOnOrBefore(date)
          if (!day) continue
          const prediction = await predictRaceTime(forecaster, day, spec)
          if (prediction) history.push({ date: day.date, prediction })
        }
        if (!active || run !== generation) return
        const future = await predictRaceTime(
          forecaster,
          bounded.day,
          spec,
          scenario.paceGainPct,
          horizon,
          enduranceExtrapolation,
        )
        if (!active || run !== generation) return
        const current = history.at(-1)?.prediction
        if (!future || !current) {
          message(output, 'model unavailable')
          output.removeAttribute('aria-busy')
          return
        }
        const metrics = projectionTable('tri-calendar-race-metrics', [
          {
            label: text(future.complete ? 'race scenario' : 'running only'),
            values: [minuteTime(future.midSec)],
          },
          { label: text('current fitness'), values: [minuteTime(current.midSec)] },
          {
            label: text(spec.kind === 'hyrox' ? 'HYROX TSS proxy' : 'estimated race TSS'),
            values: [
              future.tss === null ? text('station baseline needed') : `≈ ${Math.round(future.tss)}`,
            ],
          },
          {
            label: text('planning range'),
            values: [`${minuteTime(future.fastSec)}–${minuteTime(future.slowSec)}`],
          },
        ])
        const fitness = projectionTable('tri-calendar-race-fitness', [
          { label: text('scenario fitness'), values: [`${Math.round(projected.ctl)} CTL`] },
          {
            label: text('form'),
            values: [`${Math.round(projected.tsb) > 0 ? '+' : ''}${Math.round(projected.tsb)} TSB`],
          },
        ])
        const splits = projectionTable(
          'tri-calendar-race-splits',
          future.splits.map((split, index) => ({
            label: text(split.sport),
            values: [
              minuteTime(split.seconds),
              `${spec.legs[index].distanceKm.toLocaleString(context.presentation.locale, { maximumFractionDigits: 3 })} km`,
            ],
          })),
        )
        const summary = el('div', 'tri-calendar-race-summary')
        summary.append(metrics, splits)
        const chart = trendTable(context, history)
        const notes: string[] = []
        if (!future.complete)
          notes.push(text('Add a division and station/Roxzone time for a full HYROX estimate.'))
        if (bounded.limited) notes.push(text('Model fitness inputs limited to the observed range.'))
        if (enduranceExtrapolation) notes.push(text('Distance exceeds recent endurance evidence.'))
        if (spec.legs.some(leg => currentBaseline.recentSessions[leg.sport] < 5))
          notes.push(text('Limited recent sport data.'))
        if (calendarToday() > shiftIsoDay(currentBaseline.day.date, 7))
          notes.push(text('Training data is over a week old.'))
        output.replaceChildren(
          summary,
          ...(chart ? [chart] : []),
          ...(notes.length ? [el('p', 'tri-calendar-race-note', notes.join(' '))] : []),
        )
        row.querySelector('[data-race-projection-fitness]')?.replaceChildren(fitness)
        output.removeAttribute('aria-busy')
        output.dataset.projectionTime = String(future.midSec)
        output.dataset.projectionTss = future.tss === null ? '' : String(future.tss)
        output.dataset.projectionCurrent = String(current.midSec)
        output.dataset.projectionTrend = JSON.stringify(
          history.map(point => ({ date: point.date, seconds: point.prediction.midSec })),
        )
      }),
    )
  }
  const refresh = (): void => {
    generation++
    window.clearTimeout(timer)
    timer = window.setTimeout(() => {
      void render()
    }, 250)
  }
  const select = (): void => {
    if (!active) return
    generation++
    window.clearTimeout(timer)
    panel = null
    basis = null
    rows = []
    slot.hidden = true
    slot.replaceChildren()
    const id = calendar.dataset.calendarDetail
    if (!id) return
    const source = calendar.querySelector<HTMLTemplateElement>(
      `template[data-race-projection-template="${CSS.escape(id)}"]`,
    )
    const node = savedPanels.get(id) ?? source?.content.firstElementChild?.cloneNode(true)
    if (!(node instanceof HTMLElement)) return
    const spec = parseRaceProjectionSpec(node.dataset.raceProjection ?? '')
    if (!spec) return
    panel = node
    slot.replaceChildren(panel)
    slot.hidden = false
    savedPanels.set(spec.id, panel)
    applyI18n(panel, context.presentation)
    basis = panel.querySelector<HTMLElement>('[data-race-projection-basis]')
    const output = panel.querySelector<HTMLElement>('[data-race-projection-output]')
    if (output) rows = [{ row: panel, spec, output }]
    const input = inputFor(panel, 'weeklyLoad')
    if (input && baseline) {
      input.disabled = false
      if (!input.value) input.value = String(Math.round(baseline.weeklyLoad))
    }
    if (failed) unavailable()
    else refresh()
  }
  const onChange = (event: Event): void => {
    if (
      !(event.target instanceof HTMLSelectElement) ||
      !event.target.matches('[data-race-division]')
    )
      return
    const extra = event.target
      .closest('[data-race-projection]')
      ?.querySelector<HTMLInputElement>('[data-race-extra]')
    if (extra) {
      extra.value = ''
      extra.disabled = !event.target.value
    }
    refresh()
  }
  const onReset = (event: MouseEvent): void => {
    if (
      !(event.target instanceof Element) ||
      !event.target.closest('[data-race-scenario-reset]') ||
      !baseline ||
      !panel
    )
      return
    for (const [key, value] of Object.entries({
      weeklyLoad: Math.round(baseline.weeklyLoad),
      taperDays: 7,
      paceGainPct: 0,
    })) {
      const input = inputFor(panel, key)
      if (input) input.value = String(value)
    }
    refresh()
  }
  slot.addEventListener('input', refresh)
  slot.addEventListener('change', onChange)
  slot.addEventListener('click', onReset)
  window.addEventListener('tri:locale', refresh)
  if (!path || !context.pace.loading) unavailable()
  else
    void Promise.all([context.pace.loading, context.resources.analytics.load(path)]).then(
      ([ready, result]) => {
        if (!active) return
        const day = context.pace.forecaster?.day
        baseline =
          ready && day && result.status === 'ready'
            ? raceProjectionBaseline(result.value, day)
            : null
        if (!baseline) {
          unavailable()
          return
        }
        select()
      },
    )
  const dispose = (): void => {
    active = false
    generation++
    window.clearTimeout(timer)
    slot.removeEventListener('input', refresh)
    slot.removeEventListener('change', onChange)
    slot.removeEventListener('click', onReset)
    window.removeEventListener('tri:locale', refresh)
    savedPanels.clear()
  }
  return { select, dispose }
}
