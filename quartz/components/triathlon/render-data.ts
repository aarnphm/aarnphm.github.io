import type { Analytics } from '../../plugins/stores/analytics'
import type { TrainingPlan } from '../../plugins/stores/training'
import type { WeatherSnapshot } from '../../plugins/stores/weather'
import type { TriathlonEquipmentUsage } from '../../util/triathlon-equipment'

export interface TriathlonRenderData {
  analytics: Analytics
  plans: readonly TrainingPlan[]
  weather: WeatherSnapshot | null
  equipment?: Record<string, TriathlonEquipmentUsage>
}
