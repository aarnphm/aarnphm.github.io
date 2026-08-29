import type { GarminActivity, GarminCache } from '../plugins/stores/garmin'
import type { WahooActivity, WahooCache, WahooStreams } from '../plugins/stores/wahoo'

export const GARDEN_CYCLING_STAMINA_METHOD = 'garden-stamina-v1'

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

interface StaminaState {
  timeMs: number
  stamina: number
  potentialStamina: number
}

interface GarminStaminaState {
  activity: GarminActivity
  state: StaminaState
}

const POTENTIAL_POWER_RATE = 17.14
const POTENTIAL_POWER_EXPONENT = 2.5
const POTENTIAL_HEART_RATE_RATE = 98.85
const POTENTIAL_HEART_RATE_EXPONENT = 10
const CURRENT_DEPLETION_RATE = 800
const CURRENT_DEPLETION_EXPONENT = 1.5
const CURRENT_RECOVERY_TIME_S = 360
const BETWEEN_ACTIVITY_RECOVERY_TIME_S = 6 * 60 * 60
const MAX_SAMPLE_GAP_S = 5
const MIN_INPUT_COVERAGE = 0.8
const START_TOLERANCE_MS = 20 * 60 * 1000
const DISTANCE_TOLERANCE_RATIO = 0.08
const DISTANCE_TOLERANCE_M = 1500
const DURATION_TOLERANCE_RATIO = 0.12
const DURATION_TOLERANCE_S = 10 * 60

const clampPercentage = (value: number): number => Math.min(100, Math.max(0, value))

const validPercentage = (value: number | undefined): value is number =>
  value !== undefined && Number.isFinite(value) && value >= 0 && value <= 100

const validPower = (value: number | null | undefined): value is number =>
  value !== null && value !== undefined && Number.isFinite(value) && value >= 0

const validHeartRate = (value: number | null | undefined): value is number =>
  value !== null && value !== undefined && Number.isFinite(value) && value >= 35 && value <= 240

const validModelInput = (value: number | null | undefined): value is number =>
  value !== null && value !== undefined && Number.isFinite(value) && value > 0

function garminStaminaStates(cache: GarminCache | null): GarminStaminaState[] {
  if (!cache?.streams) return []
  const states: GarminStaminaState[] = []
  for (const [id, stream] of Object.entries(cache.streams)) {
    const activity = cache.activities[id]
    if (!activity || activity.sport !== 'bike') continue
    const stamina = stream.stamina
    const potential = stream.potentialStamina
    if (!stamina || !potential || stamina.length !== potential.length) continue
    let last = -1
    for (let index = 0; index < stamina.length; index++)
      if (validPercentage(stamina[index]) && validPercentage(potential[index])) last = index
    if (last < 0) continue
    const startMs = Date.parse(activity.startDate)
    if (!Number.isFinite(startMs)) continue
    const streamElapsedS = stream.time?.[last]
    const elapsedS =
      streamElapsedS !== undefined && Number.isFinite(streamElapsedS) && streamElapsedS >= 0
        ? streamElapsedS
        : (activity.elapsedTimeS ?? activity.movingTimeS ?? 0)
    const potentialStamina = clampPercentage(potential[last])
    states.push({
      activity,
      state: {
        timeMs: startMs + elapsedS * 1000,
        potentialStamina,
        stamina: Math.min(potentialStamina, clampPercentage(stamina[last])),
      },
    })
  }
  return states.sort((left, right) => left.state.timeMs - right.state.timeMs)
}

function matchingNativeActivity(
  wahoo: WahooActivity,
  native: readonly GarminStaminaState[],
): boolean {
  const wahooStartMs = Date.parse(wahoo.startDate)
  if (!Number.isFinite(wahooStartMs)) return false
  return native.some(({ activity }) => {
    const garminStartMs = Date.parse(activity.startDate)
    if (
      !Number.isFinite(garminStartMs) ||
      Math.abs(garminStartMs - wahooStartMs) > START_TOLERANCE_MS
    )
      return false
    if (wahoo.distanceM != null && activity.distanceM != null && wahoo.distanceM > 0) {
      const difference = Math.abs(wahoo.distanceM - activity.distanceM)
      if (
        difference > DISTANCE_TOLERANCE_M &&
        difference / wahoo.distanceM > DISTANCE_TOLERANCE_RATIO
      )
        return false
    }
    const wahooDuration = wahoo.elapsedTimeS ?? wahoo.movingTimeS
    const garminDuration = activity.elapsedTimeS ?? activity.movingTimeS
    if (wahooDuration != null && garminDuration != null && wahooDuration > 0) {
      const difference = Math.abs(wahooDuration - garminDuration)
      if (
        difference > DURATION_TOLERANCE_S &&
        difference / wahooDuration > DURATION_TOLERANCE_RATIO
      )
        return false
    }
    return true
  })
}

function recoveredState(previous: StaminaState | null, startMs: number): StaminaState {
  if (!previous || previous.timeMs > startMs)
    return { timeMs: startMs, stamina: 100, potentialStamina: 100 }
  const recoveryS = (startMs - previous.timeMs) / 1000
  const potentialStamina =
    100 -
    (100 - previous.potentialStamina) * Math.exp(-recoveryS / BETWEEN_ACTIVITY_RECOVERY_TIME_S)
  const deficit =
    (previous.potentialStamina - previous.stamina) * Math.exp(-recoveryS / CURRENT_RECOVERY_TIME_S)
  return {
    timeMs: startMs,
    potentialStamina: clampPercentage(potentialStamina),
    stamina: clampPercentage(Math.min(potentialStamina, potentialStamina - deficit)),
  }
}

function estimateActivity(
  activity: WahooActivity,
  stream: WahooStreams,
  ftpWatts: number,
  maxHeartRateBpm: number,
  initial: StaminaState,
): { estimate: CyclingStaminaEstimate; state: StaminaState } | null {
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

  let potentialStamina = initial.potentialStamina
  let currentDeficit = potentialStamina - initial.stamina
  const first = validIndices[0]
  const samples: CyclingStaminaSample[] = [
    { elapsedS: stream.time[first], stamina: initial.stamina, potentialStamina },
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
      const power = (previousPower + nextPower) / 2
      const heartRate = (previousHeartRate + nextHeartRate) / 2
      const relativePower = power / ftpWatts
      const relativeHeartRate = heartRate / maxHeartRateBpm
      const potentialRate =
        POTENTIAL_POWER_RATE * relativePower ** POTENTIAL_POWER_EXPONENT +
        POTENTIAL_HEART_RATE_RATE * relativeHeartRate ** POTENTIAL_HEART_RATE_EXPONENT
      potentialStamina = clampPercentage(potentialStamina - (potentialRate * durationS) / 3600)
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

  const startMs = Date.parse(activity.startDate)
  if (!Number.isFinite(startMs)) return null
  const lastElapsedS = samples.at(-1)?.elapsedS ?? 0
  const endElapsedS = Math.max(lastElapsedS, activity.elapsedTimeS ?? activity.movingTimeS ?? 0)
  currentDeficit *= Math.exp(-(endElapsedS - lastElapsedS) / CURRENT_RECOVERY_TIME_S)
  return {
    estimate: { method: GARDEN_CYCLING_STAMINA_METHOD, ftpWatts, maxHeartRateBpm, samples },
    state: {
      timeMs: startMs + endElapsedS * 1000,
      potentialStamina,
      stamina: potentialStamina - currentDeficit,
    },
  }
}

export function estimateWahooCyclingStamina(
  wahoo: WahooCache | null,
  garmin: GarminCache | null,
  ftpWatts: number | null,
  maxHeartRateBpm: number | null,
): ReadonlyMap<string, CyclingStaminaEstimate> {
  const estimates = new Map<string, CyclingStaminaEstimate>()
  if (!wahoo || !validModelInput(ftpWatts) || !validModelInput(maxHeartRateBpm)) return estimates
  const native = garminStaminaStates(garmin)
  const activities = Object.values(wahoo.activities)
    .filter(activity => activity.sport === 'bike')
    .sort((left, right) => left.startDate.localeCompare(right.startDate))
  let nativeIndex = 0
  let state: StaminaState | null = null
  for (const activity of activities) {
    const startMs = Date.parse(activity.startDate)
    if (!Number.isFinite(startMs)) continue
    while (nativeIndex < native.length && native[nativeIndex].state.timeMs <= startMs) {
      if (!state || native[nativeIndex].state.timeMs > state.timeMs)
        state = native[nativeIndex].state
      nativeIndex++
    }
    if (matchingNativeActivity(activity, native) || (state && state.timeMs > startMs)) continue
    const stream = wahoo.streams[activity.id]
    if (!stream) continue
    const result = estimateActivity(
      activity,
      stream,
      ftpWatts,
      maxHeartRateBpm,
      recoveredState(state, startMs),
    )
    if (!result) continue
    estimates.set(activity.id, result.estimate)
    state = result.state
  }
  return estimates
}
