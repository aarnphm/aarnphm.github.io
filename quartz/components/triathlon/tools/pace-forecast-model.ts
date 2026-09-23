import type { PaceSport } from '../../../util/pace-features'
import { none, type Cmd } from '../../../functional'
import { shiftIsoDay } from '../../../util/local-date'

export type PredCompareKey = '7' | '14' | '30' | '60' | 'custom'

export interface PaceForecastBounds {
  min: string
  max: string
}

export const paceForecastBounds = (
  available: PaceForecastBounds | null,
  selected: PaceForecastBounds,
): PaceForecastBounds | null => {
  if (!available) return null
  const min = available.min > selected.min ? available.min : selected.min
  const max = available.max < selected.max ? available.max : selected.max
  return min <= max ? { min, max } : null
}

export const paceForecastComparisonDate = (
  bounds: PaceForecastBounds,
  days: number,
): string | null => {
  const date = shiftIsoDay(bounds.max, -days)
  return date >= bounds.min && date <= bounds.max ? date : null
}

export interface PaceForecastModel {
  sport: PaceSport
  comparison: PredCompareKey
  comparisonDate: string | undefined
  generation: number
}

export type PaceForecastMessage =
  | { type: 'select-sport'; sport: PaceSport }
  | { type: 'select-comparison'; comparison: PredCompareKey }
  | { type: 'select-date'; date: string }
  | { type: 'clear-date' }
  | { type: 'refresh' }

export type PaceForecastEffect = { type: 'render'; generation: number }

export const initialPaceForecastModel = (): PaceForecastModel => ({
  sport: 'run',
  comparison: '7',
  comparisonDate: undefined,
  generation: 0,
})

const render = (
  model: PaceForecastModel,
  next: Omit<PaceForecastModel, 'generation'>,
): { model: PaceForecastModel; effects: Cmd<PaceForecastEffect> } => {
  const generation = model.generation + 1
  return { model: { ...next, generation }, effects: [{ type: 'render', generation }] }
}

export const updatePaceForecast = (
  model: PaceForecastModel,
  message: PaceForecastMessage,
): { model: PaceForecastModel; effects: Cmd<PaceForecastEffect> } => {
  switch (message.type) {
    case 'select-sport':
      return message.sport === model.sport
        ? { model, effects: none() }
        : render(model, { ...model, sport: message.sport })
    case 'select-comparison':
      return message.comparison === model.comparison
        ? { model, effects: none() }
        : render(model, { ...model, comparison: message.comparison })
    case 'select-date':
      return render(model, { ...model, comparison: 'custom', comparisonDate: message.date })
    case 'clear-date':
      return render(model, { ...model, comparison: '7', comparisonDate: undefined })
    case 'refresh':
      return render(model, model)
  }
}
