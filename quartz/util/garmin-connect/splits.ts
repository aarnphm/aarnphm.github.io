import type {
  GarminClimbSegment,
  GarminRunWalkData,
  GarminRunWalkState,
} from '../../plugins/stores/garmin'
import { isRecord, readNumber, readString } from '../type-guards'
import { firstNumber, normalizeDate, rounded, roundedFloat } from './scalars'

const CLIMB_SPLIT_TYPE = 'CLIMB_PRO_CYCLING_CLIMB'

export function garminConnectClimbSegments(raw: unknown): GarminClimbSegment[] {
  if (!isRecord(raw) || !Array.isArray(raw.splits)) return []
  const out: GarminClimbSegment[] = []
  for (const split of raw.splits) {
    if (!isRecord(split) || readString(split, 'type') !== CLIMB_SPLIT_TYPE) continue
    const startDate = normalizeDate(
      readString(split, 'startTimeGMT') ?? readString(split, 'startTimeLocal') ?? null,
    )
    const endDate = normalizeDate(
      readString(split, 'endTimeGMT') ?? readString(split, 'endTimeLocal') ?? null,
    )
    const distanceM = roundedFloat(firstNumber([split], ['distance']), 2)
    const durationS = roundedFloat(firstNumber([split], ['duration']), 3)
    if (!startDate || !endDate || distanceM == null || durationS == null) continue
    out.push({
      startDate,
      endDate,
      distanceM,
      durationS,
      movingTimeS: roundedFloat(firstNumber([split], ['movingDuration']), 3),
      elapsedTimeS: roundedFloat(firstNumber([split], ['elapsedDuration']), 3),
      elevationGainM: roundedFloat(firstNumber([split], ['elevationGain']), 1),
      elevationLossM: roundedFloat(firstNumber([split], ['elevationLoss']), 1),
      startElevationM: roundedFloat(firstNumber([split], ['startElevation']), 1),
      avgGradePct: roundedFloat(firstNumber([split], ['averageGrade']), 2),
      maxGradePct: roundedFloat(firstNumber([split], ['maxGrade']), 2),
      avgSpeedMps: roundedFloat(firstNumber([split], ['averageSpeed']), 3),
      avgHeartRate: rounded(firstNumber([split], ['averageHR'])),
      maxHeartRate: rounded(firstNumber([split], ['maxHR'])),
      avgPower: rounded(firstNumber([split], ['averagePower'])),
      normalizedPower: rounded(firstNumber([split], ['normalizedPower'])),
      maxPower: rounded(firstNumber([split], ['maxPower'])),
      avgCadence: rounded(firstNumber([split], ['averageBikeCadence'])),
      difficulty: readString(split, 'climbProDifficulty') ?? null,
    })
  }
  return out.sort((a, b) => a.startDate.localeCompare(b.startDate))
}

function garminRunWalkState(type: string | null): GarminRunWalkState | null {
  if (type === 'RWD_RUN') return 'run'
  if (type === 'RWD_WALK') return 'walk'
  if (type === 'RWD_STAND') return 'idle'
  return null
}

export function garminConnectRunWalk(raw: unknown): GarminRunWalkData | null {
  if (!isRecord(raw) || !Array.isArray(raw.splits)) return null
  const segments: GarminRunWalkData['segments'] = []
  let elapsedTimeS = 0
  let runTimeS = 0
  let walkTimeS = 0
  let idleTimeS = 0
  for (const split of raw.splits) {
    if (!isRecord(split)) continue
    const state = garminRunWalkState(readString(split, 'type') ?? null)
    const durationS = roundedFloat(readNumber(split, 'duration') ?? null, 3)
    if (!state || durationS == null || durationS <= 0) continue
    const startElapsedS = elapsedTimeS
    elapsedTimeS += durationS
    if (state === 'run') runTimeS += durationS
    else if (state === 'walk') walkTimeS += durationS
    else idleTimeS += durationS
    segments.push({
      state,
      startElapsedS: roundedFloat(startElapsedS, 3) ?? startElapsedS,
      endElapsedS: roundedFloat(elapsedTimeS, 3) ?? elapsedTimeS,
    })
  }
  if (segments.length === 0) return null
  return {
    source: 'garmin',
    elapsedTimeS: roundedFloat(elapsedTimeS, 3) ?? elapsedTimeS,
    runTimeS: roundedFloat(runTimeS, 3) ?? runTimeS,
    walkTimeS: roundedFloat(walkTimeS, 3) ?? walkTimeS,
    idleTimeS: roundedFloat(idleTimeS, 3) ?? idleTimeS,
    segments,
  }
}
