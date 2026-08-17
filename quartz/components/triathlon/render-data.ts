import type { Analytics } from '../../plugins/stores/analytics'
import type { TrainingPlan } from '../../plugins/stores/training'
import type { WeatherSnapshot } from '../../plugins/stores/weather'

export interface TriathlonRenderData {
  analytics: Analytics
  plans: readonly TrainingPlan[]
  weather: WeatherSnapshot | null
}
