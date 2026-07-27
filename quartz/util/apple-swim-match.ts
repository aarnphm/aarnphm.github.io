import type { AppleSwim } from '../plugins/stores/apple'

export interface SwimActivityCandidate {
  id: number
  date: string
  start: string
  distanceM: number
}

const SIX_HOURS_MS = 6 * 60 * 60 * 1000
const SWIM_TELEMETRY_WINDOW_MS = 15 * 60 * 1000
const MISSING_WATER_TEMPERATURE_PENALTY_MS = 60 * 1000
const MISSING_STROKE_RATE_PENALTY_MS = 60 * 1000
const MISSING_SWIM_LOCATION_PENALTY_MS = 10 * 1000

const parsedTime = (value: string | null): number | null => {
  if (!value) return null
  const time = Date.parse(value)
  return Number.isFinite(time) ? time : null
}

export function matchAppleSwims(
  swims: Iterable<AppleSwim>,
  activities: Iterable<SwimActivityCandidate>,
): Map<number, AppleSwim> {
  const candidates = [...activities].filter(
    activity =>
      /^\d{4}-\d{2}-\d{2}$/.test(activity.date) &&
      Number.isFinite(activity.distanceM) &&
      activity.distanceM > 0,
  )
  const available = [...swims]
  const sessionDates = new Set(available.filter(swim => swim.start != null).map(swim => swim.date))
  const ordered = available
    .filter(swim => swim.start != null || !sessionDates.has(swim.date))
    .sort((a, b) => (a.start ?? a.date).localeCompare(b.start ?? b.date))
  const used = new Set<number>()
  const matches = new Map<number, AppleSwim>()

  for (const swim of ordered) {
    const sameDay = candidates.filter(
      activity => activity.date === swim.date && !used.has(activity.id),
    )
    if (sameDay.length === 0) continue

    const swimTime = parsedTime(swim.start)
    let selected: SwimActivityCandidate | null = null
    if (swimTime != null) {
      let bestScore = Infinity
      const distanceLimit = Math.max(100, swim.totalM * 0.35)
      for (const activity of sameDay) {
        const activityTime = parsedTime(activity.start)
        if (activityTime == null) continue
        const timeDelta = Math.abs(activityTime - swimTime)
        const distanceDelta = Math.abs(activity.distanceM - swim.totalM)
        if (timeDelta > SIX_HOURS_MS || distanceDelta > distanceLimit) continue
        const score = timeDelta / 60_000 + distanceDelta / 10
        if (score < bestScore) {
          bestScore = score
          selected = activity
        }
      }
    } else {
      selected = sameDay.reduce((best, activity) =>
        Math.abs(activity.distanceM - swim.totalM) < Math.abs(best.distanceM - swim.totalM)
          ? activity
          : best,
      )
    }

    if (!selected) continue
    used.add(selected.id)
    matches.set(selected.id, swim)
  }

  return matches
}

export function matchAppleSwimTelemetry(
  swims: Iterable<AppleSwim>,
  activities: Iterable<SwimActivityCandidate>,
): Map<number, AppleSwim> {
  const available = [...swims].filter(
    swim =>
      parsedTime(swim.start) != null &&
      (swim.location != null ||
        swim.waterTemperatureC != null ||
        swim.strokeCount != null ||
        Object.keys(swim.strokes).length > 0),
  )
  const orderedActivities = [...activities]
    .filter(activity => parsedTime(activity.start) != null)
    .sort((left, right) => left.start.localeCompare(right.start) || left.id - right.id)
  const used = new Set<AppleSwim>()
  const matches = new Map<number, AppleSwim>()

  for (const activity of orderedActivities) {
    const activityTime = parsedTime(activity.start)
    if (activityTime == null) continue
    let selected: AppleSwim | null = null
    let selectedScore = Infinity
    for (const swim of available) {
      if (used.has(swim) || swim.date !== activity.date) continue
      const swimTime = parsedTime(swim.start)
      if (swimTime == null) continue
      const timeDelta = Math.abs(activityTime - swimTime)
      if (timeDelta > SWIM_TELEMETRY_WINDOW_MS) continue
      const score =
        timeDelta +
        (swim.waterTemperatureC == null ? MISSING_WATER_TEMPERATURE_PENALTY_MS : 0) +
        (swimStrokeRateAvailable(swim) ? 0 : MISSING_STROKE_RATE_PENALTY_MS) +
        (swim.location == null ? MISSING_SWIM_LOCATION_PENALTY_MS : 0)
      if (
        score < selectedScore ||
        (score === selectedScore && (swim.id ?? '').localeCompare(selected?.id ?? '') < 0)
      ) {
        selected = swim
        selectedScore = score
      }
    }
    if (!selected) continue
    used.add(selected)
    matches.set(activity.id, selected)
  }

  return matches
}

const swimStrokeRateAvailable = (swim: AppleSwim): boolean =>
  swim.strokeCount != null &&
  swim.strokeCount > 0 &&
  swim.strokeTimeS != null &&
  swim.strokeTimeS > 0
