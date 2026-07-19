import type { AppleWorkout } from '../plugins/stores/apple'

export interface RunActivityCandidate {
  start: string
  distanceM: number
}

const START_TOLERANCE_MS = 10_000
const DISTANCE_TOLERANCE_M = 200

export function runningDynamicsCount(workout: AppleWorkout): number {
  return (
    (workout.strideLengthM?.length ?? 0) +
    (workout.groundContactTimeMs?.length ?? 0) +
    (workout.verticalOscillationCm?.length ?? 0)
  )
}

export function matchAppleRun(
  activity: RunActivityCandidate,
  workouts: Iterable<AppleWorkout>,
): AppleWorkout | null {
  const activityStartMs = Date.parse(activity.start)
  if (!Number.isFinite(activityStartMs) || !Number.isFinite(activity.distanceM)) return null
  const candidates = [...workouts]
    .filter(workout => workout.activity === 'running' && runningDynamicsCount(workout) > 0)
    .map(workout => ({
      workout,
      startDiffMs: Math.abs(Date.parse(workout.start) - activityStartMs),
      distanceDiffM: Math.abs((workout.distanceM ?? activity.distanceM) - activity.distanceM),
    }))
    .filter(
      candidate =>
        candidate.startDiffMs <= START_TOLERANCE_MS &&
        candidate.distanceDiffM <=
          Math.max(DISTANCE_TOLERANCE_M, Math.abs(activity.distanceM) * 0.02),
    )
    .sort(
      (left, right) =>
        runningDynamicsCount(right.workout) - runningDynamicsCount(left.workout) ||
        left.startDiffMs - right.startDiffMs ||
        left.distanceDiffM - right.distanceDiffM ||
        left.workout.id.localeCompare(right.workout.id),
    )
  return candidates[0]?.workout ?? null
}
