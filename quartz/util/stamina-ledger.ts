import type { GarminHealthDay } from '../plugins/stores/garmin-health'
import type { ActivityKind, StravaActivityDetail } from '../plugins/stores/strava'
import { staminaDepletionPerHour } from './heart-rate-physiology'

export const STAMINA_LEDGER_METHOD = 'garden-stamina-ledger-v1'
export const STAMINA_LEDGER_OFFSET_METHOD = 'garden-stamina-native-offset-v1'

// Ledger state when a session starts. Deficits are stamina points below 100.
export interface StaminaLedgerStart {
  method: typeof STAMINA_LEDGER_METHOD
  // Garmin computed load or stamina for this session, so its own ledger includes it.
  garminVisible: boolean
  fatigueDeficit: number
  recoveryHours: number
  recoveryFloor: number
  // Garden deficit minus Garmin's: drain from sessions Garmin never recorded.
  garminOffset: number
}

export interface RecoveryTimeReading {
  timeMs: number
  hours: number
}

// Garmin carries stamina across every session of a day, and current snaps to potential when a
// session starts. Fast potential deficit recovery, refit jointly with the recovery floor on 107
// consecutive native Garmin starts.
const FATIGUE_RECOVERY_TIME_S = 5.6 * 60 * 60
// Since Garmin started reporting recovery time (2026-09-03), native starts never sit above
// 100 - 0.1 per remaining recovery hour (held-out start MAE 1.75 -> 1.39 on 36 starts).
// Earlier natives show no floor, so the floor starts with the first reading.
const RECOVERY_FLOOR_PER_HOUR = 0.1
// Garmin recovery time counts down in real time and caps near 85 h. A session adds
// a·(load/100)^b·e^(c·anaerobic TE), doubled for runs, scaled by the room under the cap
// (2.8 h MAE on 88 single-session reading pairs; no increment: 7.8 h).
const RECOVERY_CAP_H = 85
const RECOVERY_SCALE_H = 13.8
const RECOVERY_LOAD_EXPONENT = 1.27
const RECOVERY_ANAEROBIC_GAIN = 0.2
const RECOVERY_RUN_GAIN = 0.97
// Scalar Kalman filter on recovery hours, fit by reading-innovation likelihood: the increment
// model is loose (sd 2.8× the increment), readings are Garmin's own state (sd 1.4 h).
const RECOVERY_MODEL_SD_RATIO = 2.81
const RECOVERY_READING_VARIANCE_H2 = 1.39 ** 2
const RECOVERY_INITIAL_VARIANCE_H2 = 25
// Garmin learns fatigue resistance from recent same-sport history (monthly drift 0.79–1.36×),
// so estimates rescale to the native drops of the latest same-sport sessions.
const CALIBRATION_WINDOW = 8
const MIN_CALIBRATION_SESSIONS = 3
const MIN_SCALE = 0.5
const MAX_SCALE = 2
const MAX_CALIBRATION_GAP_S = 900
const MIN_NATIVE_OFFSET = 0.5

interface StaminaPoint {
  elapsedS: number
  stamina: number | null
  potentialStamina: number | null
}

interface Calibration {
  observed: number
  modeled: number
}

const valid = (point: StaminaPoint): boolean =>
  point.stamina != null && point.potentialStamina != null

// Matches the card: route traces first, then the swim model, then the HR model.
function sessionPoints(detail: StravaActivityDetail): StaminaPoint[] | null {
  for (const points of [
    detail.route,
    detail.swimPhysiology?.points,
    detail.heartRatePhysiology?.points,
  ])
    if (points && points.filter(valid).length >= 2) return points
  return null
}

const calibrationSport = (sport: ActivityKind): ActivityKind | null =>
  sport === 'run' || sport === 'bike' ? sport : null

function modeledRouteDrop(detail: StravaActivityDetail, maxHeartRateBpm: number): number {
  let drop = 0
  for (let index = 1; index < detail.route.length; index++) {
    const previous = detail.route[index - 1]
    const point = detail.route[index]
    const durationS = point.elapsedS - previous.elapsedS
    if (durationS <= 0 || durationS > MAX_CALIBRATION_GAP_S || previous.hr <= 0 || point.hr <= 0)
      continue
    const heartRate = (previous.hr + point.hr) / 2
    drop += (staminaDepletionPerHour(detail.sport, heartRate, maxHeartRateBpm) * durationS) / 3600
  }
  return drop
}

function scale(history: readonly Calibration[] | undefined): number {
  const recent = history?.slice(-CALIBRATION_WINDOW) ?? []
  const modeled = recent.reduce((sum, item) => sum + item.modeled, 0)
  if (recent.length < MIN_CALIBRATION_SESSIONS || modeled <= 0) return 1
  const observed = recent.reduce((sum, item) => sum + item.observed, 0)
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, observed / modeled))
}

// Garmin's load engine only processes sessions it recorded; Apple Watch race legs have no load.
const garminVisible = (detail: StravaActivityDetail): boolean =>
  detail.staminaTrace?.source === 'garmin' || detail.garmin?.exerciseLoad != null

function recoveryIncrementHours(detail: StravaActivityDetail): number {
  const load = detail.garmin?.exerciseLoad ?? detail.calculatedExerciseLoad?.value ?? null
  if (load == null || !Number.isFinite(load) || load <= 0) return 0
  const anaerobic =
    detail.garmin?.anaerobicTrainingEffect ?? detail.calculatedTrainingEffect?.anaerobic ?? 0
  return (
    RECOVERY_SCALE_H *
    (load / 100) ** RECOVERY_LOAD_EXPONENT *
    Math.exp(RECOVERY_ANAEROBIC_GAIN * anaerobic) *
    (detail.sport === 'run' ? 1 + RECOVERY_RUN_GAIN : 1)
  )
}

const addRecovery = (hours: number, increment: number): number =>
  Math.min(RECOVERY_CAP_H, hours + increment * Math.max(0, 1 - hours / RECOVERY_CAP_H))

export function garminRecoveryReadings(
  health: Record<string, GarminHealthDay> | undefined,
): RecoveryTimeReading[] {
  const readings: RecoveryTimeReading[] = []
  for (const day of Object.values(health ?? {}))
    for (const row of day.trainingReadiness.value ?? [])
      if (row.timestamp != null && row.recoveryTimeMinutes != null)
        readings.push({ timeMs: row.timestamp, hours: row.recoveryTimeMinutes / 60 })
  return readings.sort((left, right) => left.timeMs - right.timeMs)
}

// Two ledgers run side by side: Garmin's view (sessions Garmin recorded) and the garden view
// (every session). Garmin readings and native traces anchor Garmin's view; the garden view keeps
// the same corrections plus the drain of sessions Garmin never saw.
export function applyStaminaLedger(
  details: Record<string, StravaActivityDetail>,
  maxHeartRateBpm: number | null,
  readings: readonly RecoveryTimeReading[] = [],
): void {
  const sessions = Object.values(details)
    .map(detail => ({ detail, startMs: Date.parse(detail.start) }))
    .filter(session => Number.isFinite(session.startMs))
    .sort((left, right) => left.startMs - right.startMs)
  const calibration = new Map<ActivityKind, Calibration[]>()
  let timeMs: number | null = null
  let fatigue = 0
  let garminFatigue = 0
  let recovery = 0
  let garminRecovery = 0
  let variance = RECOVERY_INITIAL_VARIANCE_H2
  let readingIndex = 0
  const advance = (toMs: number) => {
    if (timeMs != null && toMs > timeMs) {
      const elapsedS = (toMs - timeMs) / 1000
      const decay = Math.exp(-elapsedS / FATIGUE_RECOVERY_TIME_S)
      fatigue *= decay
      garminFatigue *= decay
      recovery = Math.max(0, recovery - elapsedS / 3600)
      garminRecovery = Math.max(0, garminRecovery - elapsedS / 3600)
    }
    timeMs = timeMs == null ? toMs : Math.max(timeMs, toMs)
  }
  const observeUntil = (untilMs: number) => {
    for (
      ;
      readingIndex < readings.length && readings[readingIndex].timeMs <= untilMs;
      readingIndex++
    ) {
      const reading = readings[readingIndex]
      advance(reading.timeMs)
      const gain = variance / (variance + RECOVERY_READING_VARIANCE_H2)
      const innovation = reading.hours - garminRecovery
      garminRecovery = Math.max(0, garminRecovery + gain * innovation)
      recovery = Math.max(0, recovery + gain * innovation)
      variance *= 1 - gain
    }
  }
  for (const { detail, startMs } of sessions) {
    observeUntil(startMs)
    advance(startMs)
    const visible = garminVisible(detail)
    const floorPerHour =
      readings.length > 0 && startMs >= readings[0].timeMs ? RECOVERY_FLOOR_PER_HOUR : 0
    const deficit = Math.max(fatigue, floorPerHour * recovery)
    const garminDeficit = Math.max(garminFatigue, floorPerHour * garminRecovery)
    const offset = Math.max(0, deficit - garminDeficit)
    detail.staminaLedger = {
      method: STAMINA_LEDGER_METHOD,
      garminVisible: visible,
      fatigueDeficit: fatigue,
      recoveryHours: recovery,
      recoveryFloor: floorPerHour * recovery,
      garminOffset: offset,
    }
    const points = sessionPoints(detail)
    const first = points?.find(valid)
    const last = points?.findLast(valid)
    let endMs = startMs + Math.max(0, detail.elapsedTimeS) * 1000
    if (points && first?.potentialStamina != null && last?.potentialStamina != null) {
      endMs = startMs + last.elapsedS * 1000
      const sport = calibrationSport(detail.sport)
      const sessionStart = first.potentialStamina
      const sessionDrop = Math.max(0, sessionStart - last.potentialStamina)
      if (detail.staminaTrace?.source === 'garmin') {
        if (sport && maxHeartRateBpm != null) {
          const modeled = modeledRouteDrop(detail, maxHeartRateBpm)
          if (modeled > 0)
            calibration.set(sport, [
              ...(calibration.get(sport) ?? []),
              { observed: sessionDrop, modeled },
            ])
        }
        garminFatigue = 100 - last.potentialStamina
        if (offset >= MIN_NATIVE_OFFSET) {
          for (const point of points) {
            if (point.potentialStamina == null || point.stamina == null) continue
            point.potentialStamina = Math.max(0, point.potentialStamina - offset)
            point.stamina = Math.max(0, point.stamina - offset)
          }
          detail.staminaTrace = {
            source: 'garden-estimate',
            method: STAMINA_LEDGER_OFFSET_METHOD,
            ftpWatts: null,
            maxHeartRateBpm: null,
            garminOffset: offset,
          }
        }
        fatigue = 100 - last.potentialStamina
      } else {
        const factor = sport ? scale(calibration.get(sport)) : 1
        const startPotential = 100 - deficit
        for (const point of points) {
          if (point.potentialStamina == null || point.stamina == null) continue
          const currentDeficit = point.potentialStamina - point.stamina
          const potentialStamina = Math.max(
            0,
            startPotential - factor * (sessionStart - point.potentialStamina),
          )
          point.potentialStamina = potentialStamina
          point.stamina = Math.max(0, potentialStamina - currentDeficit)
        }
        fatigue = 100 - last.potentialStamina
        if (visible) garminFatigue = garminDeficit + factor * sessionDrop
      }
    }
    // The session's end state already holds its drain; time inside it does not recover.
    timeMs = Math.max(timeMs ?? endMs, endMs)
    const increment = recoveryIncrementHours(detail)
    recovery = addRecovery(recovery, increment)
    if (visible) {
      garminRecovery = addRecovery(garminRecovery, increment)
      variance += (RECOVERY_MODEL_SD_RATIO * increment) ** 2
    }
    // Garmin's post-exercise reading can land before the last trace sample.
    observeUntil(endMs)
  }
  observeUntil(Number.POSITIVE_INFINITY)
}
