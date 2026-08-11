import type { Analytics } from '../../plugins/stores/analytics'
import type { TrainingPlan } from '../../plugins/stores/training'

export interface TriathlonRenderData {
  analytics: Analytics
  plans: readonly TrainingPlan[]
}
