import type { WahooData, WahooStreams } from '../plugins/stores/wahoo'
import { staminaDepletionPerHour } from './heart-rate-physiology'

export const GARDEN_CYCLING_STAMINA_METHOD = 'garden-stamina-v2'

export interface CyclingStaminaSample {
  elapsedS: number
  stamina: number
  potentialStamina: number
}

export interface CyclingStaminaEstimate {
  method: typeof GARDEN_CYCLING_STAMINA_METHOD
  ftpWatts: number
  maxHeartRateBpm: number
  samples: CyclingStaminaSample[]
}

// Current stamina model; within 0.2 points RMSE of a refit on 71 Garmin native rides with power.
const CURRENT_DEPLETION_RATE = 800
const CURRENT_DEPLETION_EXPONENT = 1.5
const CURRENT_RECOVERY_TIME_S = 360
const MAX_SAMPLE_GAP_S = 5
const MIN_INPUT_COVERAGE = 0.8

const clampPercentage = (value: number): number => Math.min(100, Math.max(0, value))

const validPower = (value: number | null | undefined): value is number =>
  value !== null && value !== undefined && Number.isFinite(value) && value >= 0

const validHeartRate = (value: number | null | undefined): value is number =>
  value !== null && value !== undefined && Number.isFinite(value) && value >= 35 && value <= 240

const validModelInput = (value: number | null | undefined): value is number =>
  value !== null && value !== undefined && Number.isFinite(value) && value > 0

// A session trace from 100%. The stamina ledger carries the day's state into it.
function estimateActivity(
  stream: WahooStreams,
  ftpWatts: number,
  maxHeartRateBpm: number,
): CyclingStaminaEstimate | null {
  const sampleCount = stream.time.length
  if (
    sampleCount < 2 ||
    stream.watts.length !== sampleCount ||
    stream.heartrate.length !== sampleCount
  )
    return null
  const validIndices: number[] = []
  let previousTime = Number.NEGATIVE_INFINITY
  for (let index = 0; index < sampleCount; index++) {
    const elapsedS = stream.time[index]
    if (!Number.isFinite(elapsedS) || elapsedS < 0 || elapsedS <= previousTime) continue
    previousTime = elapsedS
    if (validPower(stream.watts[index]) && validHeartRate(stream.heartrate[index]))
      validIndices.push(index)
  }
  if (validIndices.length < 2 || validIndices.length / sampleCount < MIN_INPUT_COVERAGE) return null

  let potentialStamina = 100
  let currentDeficit = 0
  const first = validIndices[0]
  const samples: CyclingStaminaSample[] = [
    { elapsedS: stream.time[first], stamina: 100, potentialStamina },
  ]
  let previous = first
  for (let offset = 1; offset < validIndices.length; offset++) {
    const index = validIndices[offset]
    const elapsedS = stream.time[index]
    const durationS = elapsedS - stream.time[previous]
    if (durationS <= MAX_SAMPLE_GAP_S) {
      const previousPower = stream.watts[previous]
      const nextPower = stream.watts[index]
      const previousHeartRate = stream.heartrate[previous]
      const nextHeartRate = stream.heartrate[index]
      if (
        !validPower(previousPower) ||
        !validPower(nextPower) ||
        !validHeartRate(previousHeartRate) ||
        !validHeartRate(nextHeartRate)
      )
        return null
      const relativePower = (previousPower + nextPower) / 2 / ftpWatts
      const heartRate = (previousHeartRate + nextHeartRate) / 2
      potentialStamina = clampPercentage(
        potentialStamina -
          (staminaDepletionPerHour('bike', heartRate, maxHeartRateBpm) * durationS) / 3600,
      )
      const excess = Math.max(0, relativePower - 1)
      currentDeficit =
        excess > 0
          ? currentDeficit +
            (CURRENT_DEPLETION_RATE * excess ** CURRENT_DEPLETION_EXPONENT * durationS) / 3600
          : currentDeficit * Math.exp(-durationS / CURRENT_RECOVERY_TIME_S)
    } else {
      currentDeficit *= Math.exp(-durationS / CURRENT_RECOVERY_TIME_S)
    }
    currentDeficit = Math.min(potentialStamina, Math.max(0, currentDeficit))
    samples.push({ elapsedS, potentialStamina, stamina: potentialStamina - currentDeficit })
    previous = index
  }
  return { method: GARDEN_CYCLING_STAMINA_METHOD, ftpWatts, maxHeartRateBpm, samples }
}

export function estimateWahooCyclingStamina(
  wahoo: WahooData | null,
  ftpWatts: number | null,
  maxHeartRateBpm: number | null,
): ReadonlyMap<string, CyclingStaminaEstimate> {
  const estimates = new Map<string, CyclingStaminaEstimate>()
  if (!wahoo || !validModelInput(ftpWatts) || !validModelInput(maxHeartRateBpm)) return estimates
  for (const activity of Object.values(wahoo.activities)) {
    if (activity.sport !== 'bike') continue
    const stream = wahoo.streams[activity.id]
    const estimate = stream ? estimateActivity(stream, ftpWatts, maxHeartRateBpm) : null
    if (estimate) estimates.set(activity.id, estimate)
  }
  return estimates
}
