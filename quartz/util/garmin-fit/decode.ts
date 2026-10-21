import type { FitMessages, RecordMesg } from '@garmin/fitsdk'
import type { SwimLocation, SwimStroke } from '../../plugins/stores/apple'
import type {
  GarminCyclingDynamics,
  GarminFitTrainingEffect,
  GarminGearShift,
  GarminRiderPosition,
  GarminRiderPositionChange,
  GarminSwimData,
} from '../../plugins/stores/garmin'

const MAX_CYCLING_DYNAMICS_SAMPLES = 900

function positiveGearField(value: number | undefined): number | null {
  return value != null && Number.isInteger(value) && value > 0 && value < 255 ? value : null
}

export function fitGearShifts(messages: FitMessages): GarminGearShift[] {
  const shifts: GarminGearShift[] = []
  for (const event of messages.eventMesgs ?? []) {
    if (event.event !== 'frontGearChange' && event.event !== 'rearGearChange') continue
    const timestamp = event.timestamp instanceof Date ? event.timestamp : null
    const frontGearNum = positiveGearField(event.frontGearNum)
    const frontTeeth = positiveGearField(event.frontGear)
    const rearGearNum = positiveGearField(event.rearGearNum)
    const rearTeeth = positiveGearField(event.rearGear)
    if (
      !timestamp ||
      !Number.isFinite(timestamp.getTime()) ||
      frontGearNum == null ||
      frontTeeth == null ||
      rearGearNum == null ||
      rearTeeth == null
    )
      continue
    const shift = {
      timestamp: timestamp.toISOString(),
      frontGearNum,
      frontTeeth,
      rearGearNum,
      rearTeeth,
    }
    const previous = shifts[shifts.length - 1]
    if (
      previous?.frontGearNum === shift.frontGearNum &&
      previous.frontTeeth === shift.frontTeeth &&
      previous.rearGearNum === shift.rearGearNum &&
      previous.rearTeeth === shift.rearTeeth
    )
      continue
    shifts.push(shift)
  }
  return shifts.sort((left, right) => left.timestamp.localeCompare(right.timestamp))
}

function roundedFit(value: number, dp: number): number {
  const factor = 10 ** dp
  return Math.round(value * factor) / factor
}

function fitTimestamp(value: Date | number | 'min' | undefined): number | null {
  const timestamp = value instanceof Date ? value.getTime() : NaN
  return Number.isFinite(timestamp) ? timestamp : null
}

function fitPercent(value: number | undefined): number | null {
  return value != null && Number.isFinite(value) && value > 0 && value <= 100
    ? roundedFit(value, 1)
    : null
}

function fitPhase(value: number[] | undefined): [number, number] | null {
  if (!value || value.length < 2) return null
  const start = value[0]
  const end = value[1]
  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start < 0 ||
    start > 360 ||
    end <= 0 ||
    end > 360
  )
    return null
  return [roundedFit(start, 1), roundedFit(end, 1)]
}

function evenlySampledIndices(length: number, limit: number): number[] {
  if (length <= limit) return Array.from({ length }, (_, index) => index)
  return Array.from(
    new Set(
      Array.from({ length: limit }, (_, index) =>
        Math.round((index / Math.max(1, limit - 1)) * (length - 1)),
      ),
    ),
  )
}

function fitDistanceAt(records: readonly RecordMesg[], timestampMs: number): number {
  if (records.length === 0) return 0
  let low = 0
  let high = records.length - 1
  while (low + 1 < high) {
    const middle = low + Math.floor((high - low) / 2)
    const timestamp = fitTimestamp(records[middle].timestamp)
    if (timestamp == null || timestamp > timestampMs) high = middle
    else low = middle
  }
  const leftTime = fitTimestamp(records[low].timestamp)
  const rightTime = fitTimestamp(records[high].timestamp)
  const leftDistance = records[low].distance
  const rightDistance = records[high].distance
  if (
    leftTime == null ||
    rightTime == null ||
    leftDistance == null ||
    rightDistance == null ||
    !Number.isFinite(leftDistance) ||
    !Number.isFinite(rightDistance) ||
    rightTime <= leftTime
  )
    return Number.isFinite(leftDistance) ? Math.max(0, leftDistance ?? 0) : 0
  const fraction = Math.min(1, Math.max(0, (timestampMs - leftTime) / (rightTime - leftTime)))
  return Math.max(0, leftDistance + (rightDistance - leftDistance) * fraction)
}

function riderPosition(value: unknown): GarminRiderPosition | null {
  return value === 'seated' || value === 'standing' ? value : null
}

function emptyCyclingDynamics(): GarminCyclingDynamics {
  return {
    time: [],
    distance: [],
    leftPedalSmoothness: [],
    rightPedalSmoothness: [],
    leftTorqueEffectiveness: [],
    rightTorqueEffectiveness: [],
    leftPowerPhaseStart: [],
    leftPowerPhaseEnd: [],
    rightPowerPhaseStart: [],
    rightPowerPhaseEnd: [],
    positionChanges: [],
    seatedTimeS: null,
    standingTimeS: null,
  }
}

export function fitCyclingDynamics(messages: FitMessages): GarminCyclingDynamics {
  const records = (messages.recordMesgs ?? []).filter(record => {
    const timestamp = fitTimestamp(record.timestamp)
    return timestamp != null && record.distance != null && Number.isFinite(record.distance)
  })
  const startMs =
    fitTimestamp(messages.sessionMesgs?.[0]?.startTime) ??
    fitTimestamp(records[0]?.timestamp) ??
    null
  if (startMs == null) return emptyCyclingDynamics()

  const candidates = records.flatMap(record => {
    const timestampMs = fitTimestamp(record.timestamp)
    if (timestampMs == null || record.distance == null) return []
    const leftPowerPhase = fitPhase(record.leftPowerPhase)
    const rightPowerPhase = fitPhase(record.rightPowerPhase)
    const sample = {
      time: roundedFit(Math.max(0, (timestampMs - startMs) / 1000), 3),
      distance: roundedFit(Math.max(0, record.distance), 2),
      leftPedalSmoothness: fitPercent(record.leftPedalSmoothness),
      rightPedalSmoothness: fitPercent(record.rightPedalSmoothness),
      leftTorqueEffectiveness: fitPercent(record.leftTorqueEffectiveness),
      rightTorqueEffectiveness: fitPercent(record.rightTorqueEffectiveness),
      leftPowerPhaseStart: leftPowerPhase?.[0] ?? null,
      leftPowerPhaseEnd: leftPowerPhase?.[1] ?? null,
      rightPowerPhaseStart: rightPowerPhase?.[0] ?? null,
      rightPowerPhaseEnd: rightPowerPhase?.[1] ?? null,
    }
    return [
      sample.leftPedalSmoothness,
      sample.rightPedalSmoothness,
      sample.leftTorqueEffectiveness,
      sample.rightTorqueEffectiveness,
      sample.leftPowerPhaseStart,
      sample.leftPowerPhaseEnd,
      sample.rightPowerPhaseStart,
      sample.rightPowerPhaseEnd,
    ].some(value => value != null)
      ? [sample]
      : []
  })
  const sampled = evenlySampledIndices(candidates.length, MAX_CYCLING_DYNAMICS_SAMPLES).map(
    index => candidates[index],
  )

  const positionChanges: GarminRiderPositionChange[] = []
  for (const event of messages.eventMesgs ?? []) {
    if (event.event !== 'riderPositionChange') continue
    const position = riderPosition(event.riderPosition)
    const timestampMs = fitTimestamp(event.timestamp)
    if (position == null || timestampMs == null) continue
    const previous = positionChanges[positionChanges.length - 1]
    if (previous?.position === position) continue
    positionChanges.push({
      elapsedS: roundedFit(Math.max(0, (timestampMs - startMs) / 1000), 3),
      distanceM: roundedFit(fitDistanceAt(records, timestampMs), 2),
      position,
    })
  }

  const standingTimeS = roundedFit(
    (messages.lapMesgs ?? []).reduce(
      (sum, lap) =>
        sum +
        (lap.timeStanding != null && Number.isFinite(lap.timeStanding) ? lap.timeStanding : 0),
      0,
    ),
    3,
  )
  const totalTimerTime = messages.sessionMesgs?.reduce(
    (sum, session) =>
      sum +
      (session.totalTimerTime != null && Number.isFinite(session.totalTimerTime)
        ? session.totalTimerTime
        : 0),
    0,
  )
  const seatedTimeS =
    totalTimerTime != null && totalTimerTime > 0
      ? roundedFit(Math.max(0, totalTimerTime - standingTimeS), 3)
      : null

  return {
    time: sampled.map(sample => sample.time),
    distance: sampled.map(sample => sample.distance),
    leftPedalSmoothness: sampled.map(sample => sample.leftPedalSmoothness),
    rightPedalSmoothness: sampled.map(sample => sample.rightPedalSmoothness),
    leftTorqueEffectiveness: sampled.map(sample => sample.leftTorqueEffectiveness),
    rightTorqueEffectiveness: sampled.map(sample => sample.rightTorqueEffectiveness),
    leftPowerPhaseStart: sampled.map(sample => sample.leftPowerPhaseStart),
    leftPowerPhaseEnd: sampled.map(sample => sample.leftPowerPhaseEnd),
    rightPowerPhaseStart: sampled.map(sample => sample.rightPowerPhaseStart),
    rightPowerPhaseEnd: sampled.map(sample => sample.rightPowerPhaseEnd),
    positionChanges,
    seatedTimeS,
    standingTimeS: standingTimeS > 0 ? standingTimeS : null,
  }
}

export function garminFitTrainingEffect(messages: FitMessages): GarminFitTrainingEffect {
  const session = messages.sessionMesgs?.find(
    candidate =>
      candidate.totalTrainingEffect != null || candidate.totalAnaerobicTrainingEffect != null,
  )
  const metric = (value: number | null | undefined): number | null =>
    value != null && Number.isFinite(value) && value >= 0 ? roundedFit(value, 1) : null
  return {
    aerobic: metric(session?.totalTrainingEffect),
    anaerobic: metric(session?.totalAnaerobicTrainingEffect),
  }
}

function positiveFitMetric(value: number | null | undefined, dp: number): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? roundedFit(value, dp) : null
}

function decodedSwimStroke(value: unknown): SwimStroke | null {
  if (value === 'drill') return 'kickboard'
  if (value === 'im') return 'mixed'
  if (
    value === 'freestyle' ||
    value === 'backstroke' ||
    value === 'breaststroke' ||
    value === 'butterfly' ||
    value === 'mixed'
  )
    return value
  return null
}

function decodedSwimStrokeMetrics(
  stroke: SwimStroke | null,
  durationS: number,
  count: number | null | undefined,
  rate: number | null | undefined,
): { strokeCount: number | null; strokeRateSpm: number | null } {
  const nativeStrokeRate = positiveFitMetric(rate, 1)
  const nativeStrokeCount = stroke === 'kickboard' ? null : positiveFitMetric(count, 1)
  const strokeCount =
    nativeStrokeCount ??
    (nativeStrokeRate == null ? null : roundedFit((nativeStrokeRate * durationS) / 60, 1))
  return {
    strokeCount,
    strokeRateSpm:
      nativeStrokeRate ??
      (strokeCount == null ? null : roundedFit((strokeCount / durationS) * 60, 1)),
  }
}

export function garminFitSwim(messages: FitMessages): GarminSwimData | null {
  const session = messages.sessionMesgs?.find(candidate => candidate.sport === 'swimming')
  if (!session) return null
  const sessionStartMs = fitTimestamp(session.startTime)
  const poolLengthM = positiveFitMetric(session.poolLength, 2)
  const location: SwimLocation =
    session.subSport === 'lapSwimming' || poolLengthM != null ? 'pool' : 'openWater'
  const laps: GarminSwimData['laps'] = []
  let previousLapEndElapsedS = 0
  for (const lap of messages.lapMesgs ?? []) {
    const elapsedTimeS =
      positiveFitMetric(lap.totalElapsedTime, 3) ?? positiveFitMetric(lap.totalTimerTime, 3)
    const durationS =
      positiveFitMetric(lap.totalTimerTime, 3) ?? positiveFitMetric(lap.totalElapsedTime, 3)
    if (elapsedTimeS == null || durationS == null) continue
    const distanceM =
      positiveFitMetric(lap.totalDistance, 2) ??
      (positiveFitMetric(lap.avgSpeed, 3) != null
        ? roundedFit((lap.avgSpeed ?? 0) * durationS, 2)
        : null)
    if (distanceM == null) continue
    const startMs = fitTimestamp(lap.startTime)
    const startElapsedS =
      startMs != null && sessionStartMs != null
        ? roundedFit(Math.max(0, (startMs - sessionStartMs) / 1000), 3)
        : previousLapEndElapsedS
    const endElapsedS = roundedFit(startElapsedS + elapsedTimeS, 3)
    const stroke = decodedSwimStroke(lap.swimStroke)
    const strokes = decodedSwimStrokeMetrics(
      stroke,
      durationS,
      lap.totalStrokes ?? lap.totalCycles,
      lap.avgCadence,
    )
    laps.push({
      startElapsedS,
      endElapsedS,
      distanceM,
      durationS,
      strokeCount: strokes.strokeCount,
      strokeTimeS: strokes.strokeCount == null ? null : durationS,
      strokeRateSpm: strokes.strokeRateSpm,
      stroke,
      averageHeartRate: positiveFitMetric(lap.avgHeartRate, 1),
      elevationGainM: location === 'openWater' ? positiveFitMetric(lap.totalAscent, 2) : null,
    })
    previousLapEndElapsedS = endElapsedS
  }
  const lengths: GarminSwimData['lengths'] = []
  let cumulativeDistanceM = 0
  let previousEndElapsedS = 0
  for (const length of messages.lengthMesgs ?? []) {
    if (length.lengthType !== 'active') continue
    const durationS =
      positiveFitMetric(length.totalTimerTime, 3) ?? positiveFitMetric(length.totalElapsedTime, 3)
    if (durationS == null) continue
    const startMs = fitTimestamp(length.startTime)
    const startElapsedS =
      startMs != null && sessionStartMs != null
        ? roundedFit(Math.max(0, (startMs - sessionStartMs) / 1000), 3)
        : previousEndElapsedS
    const endElapsedS = roundedFit(startElapsedS + durationS, 3)
    const distanceM =
      poolLengthM ??
      (positiveFitMetric(length.avgSpeed, 3) != null
        ? roundedFit((length.avgSpeed ?? 0) * durationS, 2)
        : null)
    if (distanceM == null) continue
    const stroke = decodedSwimStroke(length.swimStroke)
    const strokes = decodedSwimStrokeMetrics(
      stroke,
      durationS,
      length.totalStrokes,
      length.avgSwimmingCadence,
    )
    cumulativeDistanceM = roundedFit(cumulativeDistanceM + distanceM, 2)
    lengths.push({
      startElapsedS,
      endElapsedS,
      distanceM,
      durationS,
      strokeCount: strokes.strokeCount,
      strokeTimeS: strokes.strokeCount == null ? null : durationS,
      strokeRateSpm: strokes.strokeRateSpm,
      stroke,
    })
    previousEndElapsedS = endElapsedS
  }
  const lengthStrokeCount = lengths.reduce((sum, length) => sum + (length.strokeCount ?? 0), 0)
  const activeTimeS =
    positiveFitMetric(session.activeTime, 3) ?? positiveFitMetric(session.totalTimerTime, 3)
  const strokeCount =
    positiveFitMetric(session.totalStrokes, 1) ??
    (lengthStrokeCount > 0 ? roundedFit(lengthStrokeCount, 1) : null)
  const strokeRateSpm =
    positiveFitMetric(session.avgCadence, 1) ??
    (strokeCount != null && activeTimeS != null
      ? roundedFit((strokeCount / activeTimeS) * 60, 1)
      : null)
  const lengthDistanceM = lengths.reduce((sum, length) => sum + length.distanceM, 0)
  return {
    location,
    elapsedTimeS: positiveFitMetric(session.totalElapsedTime, 3),
    activeTimeS,
    distanceM:
      positiveFitMetric(session.totalDistance, 2) ??
      (lengthDistanceM > 0 ? roundedFit(lengthDistanceM, 2) : null),
    strokeCount,
    strokeRateSpm,
    poolLengthM,
    laps,
    lengths,
  }
}
