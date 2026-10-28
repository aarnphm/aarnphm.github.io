import type { MapAnalysisRange } from './geo'
import type { OverviewDrawOptions, OverviewMode } from './overview'
import { none, type Cmd } from '../../../../functional'
import { ROUTE_SPORTS, type ActivityKind } from '../../../../plugins/stores/strava'

export type MapLoadStatus = 'idle' | 'loading' | 'ready' | 'failed'
export type MapStyle = 'mono' | 'streets' | 'satellite'

export interface MapModel {
  status: MapLoadStatus
  request: number
  mode: OverviewMode
  enabledSports: ReadonlySet<ActivityKind>
  selectedRouteId: string | null
  metric: number
  analysisRange: MapAnalysisRange | null
  style: MapStyle
}

export type MapMessage =
  | { type: 'load' }
  | { type: 'loaded'; request: number }
  | { type: 'failed'; request: number }
  | { type: 'set-mode'; mode: OverviewMode }
  | { type: 'toggle-sport'; sport: ActivityKind }
  | { type: 'set-sports'; sports: ReadonlySet<ActivityKind> }
  | { type: 'select-route'; id: string; metric?: number }
  | { type: 'select-metric'; metric: number }
  | { type: 'select-range'; range: MapAnalysisRange | null }
  | { type: 'clear-route' }
  | { type: 'set-style'; style: MapStyle }
  | { type: 'reset' }

export type MapEffect =
  | { type: 'load-artifacts'; request: number }
  | { type: 'draw-overview'; options?: OverviewDrawOptions }
  | { type: 'draw-route'; id: string; metric: number }
  | { type: 'apply-metric'; id: string; metric: number }
  | { type: 'draw-range'; range: MapAnalysisRange | null }
  | { type: 'apply-mode'; mode: OverviewMode }
  | { type: 'apply-style'; style: MapStyle }

export const initialMapModel = (style: MapStyle = 'mono'): MapModel => ({
  status: 'idle',
  request: 0,
  mode: 'heat',
  enabledSports: new Set(ROUTE_SPORTS),
  selectedRouteId: null,
  metric: 0,
  analysisRange: null,
  style,
})

export const updateMap = (
  model: MapModel,
  message: MapMessage,
): { model: MapModel; effects: Cmd<MapEffect> } => {
  switch (message.type) {
    case 'load': {
      const request = model.request + 1
      return {
        model: { ...model, status: 'loading', request },
        effects: [{ type: 'load-artifacts', request }],
      }
    }
    case 'loaded':
      return message.request === model.request
        ? { model: { ...model, status: 'ready' }, effects: [{ type: 'draw-overview' }] }
        : { model, effects: none() }
    case 'failed':
      return message.request === model.request
        ? { model: { ...model, status: 'failed' }, effects: none() }
        : { model, effects: none() }
    case 'set-mode':
      return {
        model: { ...model, mode: message.mode },
        effects: [{ type: 'apply-mode', mode: message.mode }],
      }
    case 'toggle-sport': {
      const enabledSports = new Set(model.enabledSports)
      if (enabledSports.has(message.sport)) enabledSports.delete(message.sport)
      else enabledSports.add(message.sport)
      return {
        model: { ...model, enabledSports },
        effects: [
          {
            type: 'draw-overview',
            options: { fit: false, redrawStreetMap: message.sport !== 'swim' },
          },
        ],
      }
    }
    case 'set-sports': {
      if (
        message.sports.size === model.enabledSports.size &&
        [...message.sports].every(sport => model.enabledSports.has(sport))
      )
        return { model, effects: none() }
      return {
        model: { ...model, enabledSports: new Set(message.sports) },
        effects: [{ type: 'draw-overview' }],
      }
    }
    case 'select-route': {
      const metric = message.metric ?? 0
      return {
        model: { ...model, selectedRouteId: message.id, metric, analysisRange: null },
        effects: [{ type: 'draw-route', id: message.id, metric }],
      }
    }
    case 'select-metric':
      return model.selectedRouteId
        ? {
            model: { ...model, metric: message.metric },
            effects: [{ type: 'apply-metric', id: model.selectedRouteId, metric: message.metric }],
          }
        : { model: { ...model, metric: message.metric }, effects: none() }
    case 'select-range':
      return {
        model: { ...model, analysisRange: message.range },
        effects: [{ type: 'draw-range', range: message.range }],
      }
    case 'clear-route':
      return {
        model: { ...model, selectedRouteId: null, metric: 0, analysisRange: null },
        effects: [{ type: 'draw-overview', options: { fit: false } }],
      }
    case 'set-style':
      return {
        model: { ...model, style: message.style },
        effects: [{ type: 'apply-style', style: message.style }],
      }
    case 'reset':
      return {
        model: {
          ...model,
          mode: 'heat',
          enabledSports: new Set(ROUTE_SPORTS),
          selectedRouteId: null,
          metric: 0,
          analysisRange: null,
        },
        effects: [{ type: 'draw-overview' }],
      }
  }
}
