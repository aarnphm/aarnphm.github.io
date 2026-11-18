import type { RawStravaActivity } from '../../plugins/stores/strava'
import { requireStravaId } from '../strava-id'
import { isRecord } from '../type-guards'
import { intersection, lastDate, localDate, type DistanceRecord, type UsageRange } from './records'

export interface MaintenanceDistanceResult {
  label: string
  bikeId: string
  previousMiles: unknown
  distanceMiles: number | null
  ranges: UsageRange[]
  excludedChangeoverDates: string[]
  reason: string | null
  meters: number
  miles: number
  activityIds: string[]
  virtualMiles: number
  virtualActivityIds: string[]
  defaultBikeActivityIds: string[]
}

export function calculateDistances(
  records: DistanceRecord[],
  activities: Record<string, RawStravaActivity>,
  defaultVirtualRideBikeId?: string,
): MaintenanceDistanceResult[] {
  if (!isRecord(activities)) throw new Error('Strava cache has no activity mapping')
  const trackedBikes = new Set(records.map(record => record.bikeId))
  const rides = new Map<
    string,
    {
      id: string
      bikeId: string
      date: string
      meters: number
      virtual: boolean
      defaultBike: boolean
    }
  >()
  for (const activity of Object.values(activities)) {
    if (!isRecord(activity)) throw new Error('Invalid Strava activity in maintenance cache')
    const virtual = activity.sportType === 'VirtualRide'
    const bikeId = activity.gearId || (virtual ? defaultVirtualRideBikeId : undefined)
    if ((activity.sportType !== 'Ride' && !virtual) || !bikeId || !trackedBikes.has(bikeId))
      continue
    if (
      typeof activity.startDateLocal !== 'string' ||
      typeof activity.distance !== 'number' ||
      !Number.isFinite(activity.distance) ||
      activity.distance < 0
    )
      throw new Error('Invalid Strava Ride date, distance or ID')
    const id = requireStravaId(activity.id)
    rides.set(id, {
      id,
      bikeId,
      date: localDate(activity.startDateLocal.slice(0, 10)),
      meters: activity.distance,
      virtual,
      defaultBike: !activity.gearId,
    })
  }
  const matches = (date: string, range: UsageRange) =>
    date >= range.start && date <= (range.end ?? lastDate)
  const total = (record: DistanceRecord, ranges: UsageRange[], excluded = new Set<string>()) => {
    const selected = [...rides.values()].filter(
      ride =>
        ride.bikeId === record.bikeId &&
        (!ride.virtual || record.includeVirtual) &&
        !excluded.has(ride.date) &&
        ranges.some(range => matches(ride.date, range)),
    )
    const meters = Math.round(selected.reduce((sum, ride) => sum + ride.meters, 0) * 10) / 10
    const virtual = selected.filter(ride => ride.virtual)
    const virtualMeters = Math.round(virtual.reduce((sum, ride) => sum + ride.meters, 0) * 10) / 10
    return {
      meters,
      miles: Number((meters / 1609.344).toFixed(2)),
      activityIds: selected.map(ride => ride.id),
      virtualMiles: Number((virtualMeters / 1609.344).toFixed(2)),
      virtualActivityIds: virtual.map(ride => ride.id),
      defaultBikeActivityIds: selected.filter(ride => ride.defaultBike).map(ride => ride.id),
    }
  }
  return records.map((record, index) => {
    const excluded = new Set<string>()
    const conflicts = new Set<string>()
    if (record.group)
      for (const [otherIndex, other] of records.entries()) {
        if (index === otherIndex || record.group !== other.group || record.bikeId !== other.bikeId)
          continue
        for (const period of record.ranges)
          for (const otherPeriod of other.ranges) {
            const overlap = intersection(period, otherPeriod)
            if (!overlap || !total(record, [overlap]).activityIds.length) continue
            const start = record.replacementDate ?? period.start
            const otherStart = other.replacementDate ?? otherPeriod.start
            if (overlap.start !== overlap.end || (record.replacementWins && start === otherStart))
              conflicts.add(`${other.label}: ${overlap.start} to ${overlap.end ?? 'current'}`)
            else if (record.replacementWins && otherStart > start) excluded.add(overlap.start)
          }
      }
    const calculated = total(record, record.ranges, excluded)
    const reason = conflicts.size ? `overlapping usage: ${[...conflicts].join('; ')}` : null
    return {
      label: record.label,
      bikeId: record.bikeId,
      previousMiles: record.target.distance,
      distanceMiles: reason ? null : calculated.miles,
      ranges: record.ranges,
      excludedChangeoverDates: [...excluded],
      reason,
      ...calculated,
    }
  })
}
