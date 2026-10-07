import type {
  TrainingPeaksCalendarMetrics,
  TrainingPeaksCalendarWorkout,
} from './trainingpeaks-calendar'

export type TrainingPeaksExecutionStatus =
  | 'on-target'
  | 'off-target'
  | 'far-off-target'
  | 'missed'
  | 'planned'
  | 'unplanned'
  | 'unavailable'

export interface TrainingPeaksExecution {
  status: TrainingPeaksExecutionStatus
  metric: keyof TrainingPeaksCalendarMetrics | null
  ratio: number | null
}

const priority: readonly (keyof TrainingPeaksCalendarMetrics)[] = [
  'durationSeconds',
  'distanceMeters',
  'tss',
]

export function trainingPeaksExecution(
  workout: TrainingPeaksCalendarWorkout,
  today: string,
): TrainingPeaksExecution {
  if (workout.status === 'planned')
    return { status: workout.date < today ? 'missed' : 'planned', metric: null, ratio: null }

  for (const metric of priority) {
    const planned = workout.planned[metric]
    const actual = workout.actual[metric]
    if (planned === null || planned <= 0 || actual === null) continue
    const ratio = actual / planned
    return {
      status:
        ratio >= 0.8 && ratio <= 1.2
          ? 'on-target'
          : ratio >= 0.5 && ratio <= 1.5
            ? 'off-target'
            : 'far-off-target',
      metric,
      ratio,
    }
  }

  return {
    status: priority.some(metric => (workout.planned[metric] ?? 0) > 0)
      ? 'unavailable'
      : 'unplanned',
    metric: null,
    ratio: null,
  }
}
