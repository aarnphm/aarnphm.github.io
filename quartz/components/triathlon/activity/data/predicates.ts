import { RECORDING_PAUSE_MIN_S, type ElapsedPause } from '../../../../util/elapsed-axis'
import { isRecord } from '../../../../util/type-guards'

export const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

export const nullableFinite = (value: unknown): value is number | null =>
  value === null || finite(value)

export const bounded = (value: unknown, minimum: number, maximum: number): value is number =>
  finite(value) && value >= minimum && value <= maximum

export const nullableBounded = (value: unknown, minimum: number, maximum: number): boolean =>
  value === null || bounded(value, minimum, maximum)

export const optionalNullableBounded = (
  value: unknown,
  minimum: number,
  maximum: number,
): boolean => value === undefined || nullableBounded(value, minimum, maximum)

/** Ordered, disjoint recording pauses inside the activity, each at least the builder threshold. */
export const isElapsedPauses = (value: unknown, elapsedTimeS: number): value is ElapsedPause[] => {
  if (!Array.isArray(value)) return false
  let previousEndS = 0
  for (const pause of value) {
    if (
      !isRecord(pause) ||
      !bounded(pause.startS, previousEndS, elapsedTimeS) ||
      !bounded(pause.endS, 0, elapsedTimeS) ||
      pause.endS - pause.startS < RECORDING_PAUSE_MIN_S
    )
      return false
    previousEndS = pause.endS
  }
  return true
}
