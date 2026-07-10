import type { AppleSwim } from '../plugins/stores/apple'

export interface SwimActivityCandidate {
  id: number
  date: string
  start: string
  distanceM: number
}

const SIX_HOURS_MS = 6 * 60 * 60 * 1000

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
