import type { ActivityKind, StravaActivityDetail } from '../plugins/stores/strava'
import { staminaDepletionPerHour } from './heart-rate-physiology'

// Garmin carries stamina across every session of a day: a swim without a stamina trace still
// lowers the next run's start, and current snaps to potential when a session starts.
// Potential deficit recovery fit to 98 consecutive native Garmin pairs: within-day gaps
// fit 3–4 h, overnight gaps 5–6 h; the within-day value wins because days are the unit shown.
const RECOVERY_TIME_S = 4 * 60 * 60
// Garmin learns fatigue resistance from recent same-sport history (monthly drift 0.79–1.36×),
// so estimates rescale to the native drops of the latest same-sport sessions.
const CALIBRATION_WINDOW = 8
const MIN_CALIBRATION_SESSIONS = 3
const MIN_SCALE = 0.5
const MAX_SCALE = 2
const MAX_CALIBRATION_GAP_S = 900

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

export function applyStaminaLedger(
  details: Record<string, StravaActivityDetail>,
  maxHeartRateBpm: number | null,
): void {
  const sessions = Object.values(details)
    .map(detail => ({ detail, startMs: Date.parse(detail.start) }))
    .filter(session => Number.isFinite(session.startMs))
    .sort((left, right) => left.startMs - right.startMs)
  const calibration = new Map<ActivityKind, Calibration[]>()
  let state: { timeMs: number; potentialStamina: number } | null = null
  for (const { detail, startMs } of sessions) {
    const points = sessionPoints(detail)
    if (!points) continue
    const first = points.find(valid)
    const last = points.findLast(valid)
    if (first?.potentialStamina == null || last?.potentialStamina == null) continue
    const endMs = startMs + last.elapsedS * 1000
    const sport = calibrationSport(detail.sport)
    if (detail.staminaTrace?.source === 'garmin') {
      if (sport && maxHeartRateBpm != null) {
        const modeled = modeledRouteDrop(detail, maxHeartRateBpm)
        if (modeled > 0)
          calibration.set(sport, [
            ...(calibration.get(sport) ?? []),
            { observed: first.potentialStamina - last.potentialStamina, modeled },
          ])
      }
      state = { timeMs: endMs, potentialStamina: last.potentialStamina }
      continue
    }
    const recoveryS = state ? Math.max(0, (startMs - state.timeMs) / 1000) : 0
    const startPotential = state
      ? 100 - (100 - state.potentialStamina) * Math.exp(-recoveryS / RECOVERY_TIME_S)
      : 100
    const factor = sport ? scale(calibration.get(sport)) : 1
    const sessionStart = first.potentialStamina
    for (const point of points) {
      if (point.potentialStamina == null || point.stamina == null) continue
      const deficit = point.potentialStamina - point.stamina
      const potentialStamina = Math.max(
        0,
        startPotential - factor * (sessionStart - point.potentialStamina),
      )
      point.potentialStamina = potentialStamina
      point.stamina = Math.max(0, potentialStamina - deficit)
    }
    state = { timeMs: endMs, potentialStamina: last.potentialStamina }
  }
}
