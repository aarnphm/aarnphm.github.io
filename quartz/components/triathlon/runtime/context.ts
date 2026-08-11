import type { Analytics } from '../../../plugins/stores/analytics'
import type { OuraDayDetail } from '../../../plugins/stores/oura'
import type { TrainingPayload } from '../../../plugins/stores/training'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { DetailPayload } from '../activity/data'
import type { TriathlonFormatter } from './formatter'
import { createPaceRuntime, type PaceRuntime } from '../tools/pace-runtime'
import { createNavigationResource, type NavigationResource } from './data'
import { createTriathlonEventBus, type TriathlonEventBus } from './events'
import { createPreferenceController, type PreferenceController } from './preferences'

export interface TriathlonResources {
  detail: NavigationResource<DetailPayload>
  analytics: NavigationResource<Analytics>
  oura: NavigationResource<Record<string, OuraDayDetail>>
  training: NavigationResource<TrainingPayload>
}

export interface TriathlonContext {
  signal: AbortSignal
  root: HTMLElement | null
  scope: Document
  presentation: TriathlonPresentation
  formatter: TriathlonFormatter
  preferences: PreferenceController
  resources: TriathlonResources
  events: TriathlonEventBus
  pace: PaceRuntime
}

const readJson = async <T>(response: Response): Promise<T> => {
  const value: T = await response.json()
  return value
}

export const createTriathlonContext = (signal: AbortSignal): TriathlonContext => {
  const events = createTriathlonEventBus()
  const preferences = createPreferenceController(events)
  return {
    signal,
    root: document.querySelector<HTMLElement>('.triathlon'),
    scope: document,
    get presentation() {
      return preferences.current()
    },
    get formatter() {
      return preferences.formatter()
    },
    preferences,
    events,
    pace: createPaceRuntime(),
    resources: {
      detail: createNavigationResource(signal, readJson<DetailPayload>),
      analytics: createNavigationResource(signal, readJson<Analytics>),
      oura: createNavigationResource(signal, readJson<Record<string, OuraDayDetail>>),
      training: createNavigationResource(signal, readJson<TrainingPayload>),
    },
  }
}
