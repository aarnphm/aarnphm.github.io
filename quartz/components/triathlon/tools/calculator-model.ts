import type { TriathlonCalcInput } from '../../../util/triathlon-calculator'
import { none, type Cmd } from '../../../functional'

export type CalculatorSource = 'manual' | 'avg' | 'pred' | 'projection'

export interface CalculatorProjectionModel {
  active: boolean
  zone: number
  key: string
  pendingKey: string
  distribution: { mu: number; sigma: number } | null
}

export interface CalculatorModel {
  input: TriathlonCalcInput
  source: CalculatorSource
  userEdited: boolean
  projection: CalculatorProjectionModel
}

export type CalculatorMessage =
  | { type: 'sync-input'; input: TriathlonCalcInput; userEdited?: boolean; forceTarget?: boolean }
  | { type: 'select-source'; source: Exclude<CalculatorSource, 'manual'> }
  | { type: 'select-zone'; zone: number }
  | { type: 'projection-requested'; key: string }
  | { type: 'projection-loaded'; key: string; mu: number; sigma: number }
  | { type: 'projection-failed'; key: string }
  | { type: 'projection-invalidated' }

export type CalculatorEffect =
  | { type: 'compute'; forceTarget: boolean }
  | { type: 'apply-source'; source: 'avg' | 'pred' }
  | { type: 'render-projection' }
  | { type: 'sync-source' }

export const initialCalculatorModel = (input: TriathlonCalcInput): CalculatorModel => ({
  input,
  source: 'manual',
  userEdited: false,
  projection: { active: false, zone: 2, key: '', pendingKey: '', distribution: null },
})

const projectionKey = (input: TriathlonCalcInput): string =>
  `${input.swimKm}-${input.bikeKm}-${input.runKm}`

export const updateCalculator = (
  model: CalculatorModel,
  message: CalculatorMessage,
): { model: CalculatorModel; effects: Cmd<CalculatorEffect> } => {
  switch (message.type) {
    case 'sync-input': {
      const effects: CalculatorEffect[] = [
        { type: 'compute', forceTarget: message.forceTarget ?? false },
      ]
      if (message.userEdited) effects.push({ type: 'sync-source' })
      const projection =
        model.projection.key && model.projection.key !== projectionKey(message.input)
          ? { ...model.projection, key: '', pendingKey: '', distribution: null }
          : model.projection
      return {
        model: {
          ...model,
          input: message.input,
          source: message.userEdited ? 'manual' : model.source,
          userEdited: message.userEdited ?? model.userEdited,
          projection,
        },
        effects,
      }
    }
    case 'select-source':
      return message.source === 'projection'
        ? {
            model: {
              ...model,
              source: 'projection',
              projection: { ...model.projection, active: true },
            },
            effects: [{ type: 'sync-source' }, { type: 'render-projection' }],
          }
        : {
            model: {
              ...model,
              source: message.source,
              userEdited: false,
              projection: { ...model.projection, active: false },
            },
            effects: [{ type: 'sync-source' }, { type: 'apply-source', source: message.source }],
          }
    case 'select-zone':
      return message.zone === model.projection.zone
        ? { model, effects: none() }
        : {
            model: { ...model, projection: { ...model.projection, zone: message.zone } },
            effects: [{ type: 'render-projection' }],
          }
    case 'projection-requested':
      return {
        model: { ...model, projection: { ...model.projection, pendingKey: message.key } },
        effects: none(),
      }
    case 'projection-loaded':
      return message.key === model.projection.pendingKey
        ? {
            model: {
              ...model,
              projection: {
                ...model.projection,
                key: message.key,
                pendingKey: '',
                distribution: { mu: message.mu, sigma: message.sigma },
              },
            },
            effects: model.projection.active ? [{ type: 'render-projection' }] : none(),
          }
        : { model, effects: none() }
    case 'projection-failed':
      return message.key === model.projection.pendingKey
        ? {
            model: { ...model, projection: { ...model.projection, pendingKey: '' } },
            effects: none(),
          }
        : { model, effects: none() }
    case 'projection-invalidated':
      return model.projection.distribution
        ? {
            model: {
              ...model,
              projection: { ...model.projection, key: '', pendingKey: '', distribution: null },
            },
            effects: none(),
          }
        : { model, effects: none() }
  }
}
