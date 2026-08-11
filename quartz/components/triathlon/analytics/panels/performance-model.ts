export interface PerformanceModel {
  futureDailyLoad: number
  activeIndex: number
  lockedIndex: number | null
}

export type PerformanceMessage =
  | { type: 'set-load'; load: number }
  | { type: 'hover'; index: number }
  | { type: 'leave' }
  | { type: 'toggle-lock'; index: number }

export interface PerformanceBounds {
  lastObservedIndex: number
  maximumIndex: number
  maximumLoad: number
}

export const initialPerformanceModel = (
  futureDailyLoad: number,
  bounds: PerformanceBounds,
): PerformanceModel => ({
  futureDailyLoad: Math.min(bounds.maximumLoad, Math.max(0, futureDailyLoad)),
  activeIndex: bounds.lastObservedIndex,
  lockedIndex: null,
})

const index = (value: number, bounds: PerformanceBounds): number =>
  Math.round(Math.min(bounds.maximumIndex, Math.max(0, value)))

export const updatePerformance = (
  model: PerformanceModel,
  message: PerformanceMessage,
  bounds: PerformanceBounds,
): PerformanceModel => {
  if (message.type === 'set-load')
    return { ...model, futureDailyLoad: Math.min(bounds.maximumLoad, Math.max(0, message.load)) }
  if (message.type === 'hover')
    return model.lockedIndex == null
      ? { ...model, activeIndex: index(message.index, bounds) }
      : model
  if (message.type === 'leave')
    return model.lockedIndex == null ? { ...model, activeIndex: bounds.lastObservedIndex } : model
  const lockedIndex = index(message.index, bounds)
  return model.lockedIndex === lockedIndex
    ? { ...model, activeIndex: bounds.lastObservedIndex, lockedIndex: null }
    : { ...model, activeIndex: lockedIndex, lockedIndex }
}
