import { none, type Cmd } from '../../../functional'

export type AnalyticsLoadStatus = 'idle' | 'loading' | 'ready' | 'failed'
export type AnalyticsMode = 'main' | 'search' | 'detail' | 'compare'

export interface AnalyticsModel {
  status: AnalyticsLoadStatus
  request: number
  mode: AnalyticsMode
  query: string
  selectedResult: number
  selectedActivityId: string | null
  comparisonActivityIds: readonly string[]
}

export type AnalyticsMessage =
  | { type: 'open' }
  | { type: 'close' }
  | { type: 'load' }
  | { type: 'loaded'; request: number }
  | { type: 'failed'; request: number }
  | { type: 'query'; value: string }
  | { type: 'select-result'; index: number }
  | { type: 'show-activity'; id: string }
  | { type: 'toggle-compare' }
  | { type: 'toggle-comparison-activity'; id: string }
  | { type: 'remove-comparison-activity'; id: string }
  | { type: 'set-comparison-activities'; ids: readonly string[] }
  | { type: 'submit-comparison' }
  | { type: 'reset' }

export type AnalyticsEffect =
  | { type: 'load-artifact'; request: number }
  | { type: 'render-panels' }
  | { type: 'render-search' }
  | { type: 'render-activity'; id: string }
  | { type: 'render-comparison'; ids: readonly string[] }
  | { type: 'restore-focus' }

export const initialAnalyticsModel = (): AnalyticsModel => ({
  status: 'idle',
  request: 0,
  mode: 'main',
  query: '',
  selectedResult: -1,
  selectedActivityId: null,
  comparisonActivityIds: [],
})

const removeActivity = (ids: readonly string[], id: string): readonly string[] =>
  ids.filter(candidate => candidate !== id)

export const updateAnalytics = (
  model: AnalyticsModel,
  message: AnalyticsMessage,
): { model: AnalyticsModel; effects: Cmd<AnalyticsEffect> } => {
  switch (message.type) {
    case 'open':
      return model.status === 'idle' || model.status === 'failed'
        ? updateAnalytics(model, { type: 'load' })
        : { model, effects: none() }
    case 'close':
      return {
        model: {
          ...model,
          mode: 'main',
          query: '',
          selectedResult: -1,
          selectedActivityId: null,
          comparisonActivityIds: [],
        },
        effects: [{ type: 'restore-focus' }],
      }
    case 'load': {
      const request = model.request + 1
      return {
        model: { ...model, status: 'loading', request },
        effects: [{ type: 'load-artifact', request }],
      }
    }
    case 'loaded':
      return message.request === model.request
        ? { model: { ...model, status: 'ready' }, effects: [{ type: 'render-panels' }] }
        : { model, effects: none() }
    case 'failed':
      return message.request === model.request
        ? { model: { ...model, status: 'failed' }, effects: none() }
        : { model, effects: none() }
    case 'query':
      return {
        model: {
          ...model,
          mode: model.mode === 'compare' ? 'compare' : message.value.trim() ? 'search' : 'main',
          query: message.value,
          selectedResult: message.value.trim() ? 0 : -1,
          selectedActivityId: null,
        },
        effects: [{ type: 'render-search' }],
      }
    case 'select-result':
      return { model: { ...model, selectedResult: message.index }, effects: none() }
    case 'show-activity':
      return {
        model: { ...model, mode: 'detail', selectedActivityId: message.id },
        effects: [{ type: 'render-activity', id: message.id }],
      }
    case 'toggle-compare':
      return {
        model: {
          ...model,
          mode: model.mode === 'compare' ? 'main' : 'compare',
          comparisonActivityIds: model.mode === 'compare' ? [] : model.comparisonActivityIds,
        },
        effects: none(),
      }
    case 'toggle-comparison-activity': {
      const ids = model.comparisonActivityIds.includes(message.id)
        ? removeActivity(model.comparisonActivityIds, message.id)
        : [...model.comparisonActivityIds, message.id]
      return {
        model: { ...model, mode: 'compare', comparisonActivityIds: ids },
        effects: [{ type: 'render-search' }],
      }
    }
    case 'remove-comparison-activity': {
      const ids = removeActivity(model.comparisonActivityIds, message.id)
      return {
        model: { ...model, comparisonActivityIds: ids },
        effects: ids.length >= 2 ? [{ type: 'render-comparison', ids }] : none(),
      }
    }
    case 'set-comparison-activities':
      return {
        model: { ...model, mode: 'compare', comparisonActivityIds: [...message.ids] },
        effects: [{ type: 'render-search' }],
      }
    case 'submit-comparison':
      return model.comparisonActivityIds.length >= 2
        ? {
            model: { ...model, mode: 'compare' },
            effects: [{ type: 'render-comparison', ids: model.comparisonActivityIds }],
          }
        : { model, effects: none() }
    case 'reset':
      return {
        model: {
          ...model,
          mode: 'main',
          query: '',
          selectedResult: -1,
          selectedActivityId: null,
          comparisonActivityIds: [],
        },
        effects: none(),
      }
  }
}
