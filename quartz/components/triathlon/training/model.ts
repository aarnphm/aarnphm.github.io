import type { TrainingPlan } from '../../../plugins/stores/training'
import { none, type Cmd } from '../../../functional'

export type TrainingLoadState = 'idle' | 'loading' | 'ready' | 'failed'

export interface TrainingModel {
  status: TrainingLoadState
  plans: readonly TrainingPlan[]
  query: string
  selectedPlanId: string | null
  selectedTreeTarget: string | null
}

export type TrainingMessage =
  | { type: 'load' }
  | { type: 'loaded'; plans: readonly TrainingPlan[] }
  | { type: 'failed' }
  | { type: 'query'; value: string }
  | { type: 'select-plan'; id: string }
  | { type: 'select-tree-target'; id: string }
  | { type: 'reset' }

export type TrainingEffect =
  | { type: 'load-artifact' }
  | { type: 'render-plans' }
  | { type: 'render-search' }
  | { type: 'render-plan'; id: string }
  | { type: 'scroll-to'; id: string }

export const initialTrainingModel = (plans: readonly TrainingPlan[] = []): TrainingModel => ({
  status: plans.length > 0 ? 'ready' : 'idle',
  plans,
  query: '',
  selectedPlanId: plans[0]?.id ?? null,
  selectedTreeTarget: null,
})

const selectedPlanId = (plans: readonly TrainingPlan[], requested: string | null): string | null =>
  requested && plans.some(plan => plan.id === requested) ? requested : (plans[0]?.id ?? null)

export const updateTraining = (
  model: TrainingModel,
  message: TrainingMessage,
): { model: TrainingModel; effects: Cmd<TrainingEffect> } => {
  switch (message.type) {
    case 'load':
      return { model: { ...model, status: 'loading' }, effects: [{ type: 'load-artifact' }] }
    case 'loaded':
      return {
        model: {
          ...model,
          status: 'ready',
          plans: message.plans,
          selectedPlanId: selectedPlanId(message.plans, model.selectedPlanId),
        },
        effects: [{ type: 'render-plans' }],
      }
    case 'failed':
      return { model: { ...model, status: 'failed' }, effects: none() }
    case 'query':
      return { model: { ...model, query: message.value }, effects: [{ type: 'render-search' }] }
    case 'select-plan': {
      const selected = selectedPlanId(model.plans, message.id)
      return {
        model: { ...model, selectedPlanId: selected, selectedTreeTarget: null },
        effects: selected ? [{ type: 'render-plan', id: selected }] : none(),
      }
    }
    case 'select-tree-target':
      return {
        model: { ...model, selectedTreeTarget: message.id },
        effects: [{ type: 'scroll-to', id: message.id }],
      }
    case 'reset':
      return {
        model: { ...model, query: '', selectedTreeTarget: null },
        effects: [{ type: 'render-plans' }],
      }
  }
}

export const filterTrainingPlans = (
  plans: readonly TrainingPlan[],
  query: string,
): readonly TrainingPlan[] => {
  const normalized = query.trim().toLocaleLowerCase()
  if (!normalized) return plans
  return plans.filter(plan =>
    `${plan.meta} ${plan.distance} ${plan.target} ${plan.date}`
      .toLocaleLowerCase()
      .includes(normalized),
  )
}
