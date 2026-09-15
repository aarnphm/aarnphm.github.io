import type { StravaRawCache } from '../plugins/stores/strava'
import type { DistanceSystem } from './triathlon-presentation'

export const formatEquipmentDistance = (metres: number, system: DistanceSystem): string => {
  const value = system === 'imperial' ? metres / 1_609.344 : metres / 1_000
  return `${value.toLocaleString('en-US', { maximumFractionDigits: 2 })} ${system === 'imperial' ? 'mi' : 'km'}`
}

export interface TriathlonEquipmentUsage {
  id: string
  name: string | null
  lifetimeDistanceM: number | null
  activityCount: number
  firstRecorded: string | null
  lastRecorded: string | null
  source: 'strava'
}

export const buildTriathlonEquipment = (
  cache: Pick<StravaRawCache, 'gear' | 'activities'> | null,
): Record<string, TriathlonEquipmentUsage> => {
  const equipment: Record<string, TriathlonEquipmentUsage> = {}
  for (const gear of Object.values(cache?.gear ?? {})) {
    equipment[gear.id] = {
      id: gear.id,
      name: gear.name,
      lifetimeDistanceM:
        gear.distanceM != null && Number.isFinite(gear.distanceM) && gear.distanceM >= 0
          ? gear.distanceM
          : null,
      activityCount: 0,
      firstRecorded: null,
      lastRecorded: null,
      source: 'strava',
    }
  }
  for (const activity of Object.values(cache?.activities ?? {})) {
    const usage = activity.gearId ? equipment[activity.gearId] : undefined
    if (!usage) continue
    usage.activityCount += 1
    const date = activity.startDateLocal.slice(0, 10)
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    if (usage.firstRecorded == null || date < usage.firstRecorded) usage.firstRecorded = date
    if (usage.lastRecorded == null || date > usage.lastRecorded) usage.lastRecorded = date
  }
  return equipment
}
