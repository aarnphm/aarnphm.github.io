import { activityId } from './garmin-connect/activity'
import { isRecord, type UnknownRecord } from './type-guards'
export { garminConnectActivity, garminConnectActivityStartDate } from './garmin-connect/activity'
export { garminConnectClimbSegments, garminConnectRunWalk } from './garmin-connect/splits'
export { garminConnectStreams } from './garmin-connect/streams'
export {
  garminConnectLactateThresholdHistory,
  garminConnectRunningLactateThreshold,
  garminConnectSleep,
  garminConnectVo2,
} from './garmin-connect/wellness'
export { garminConnectWeightSamples } from './garmin-connect/weight'

export interface GarminConnectActivityListItem {
  id: string
  record: UnknownRecord
}

function recordsFromJson(raw: unknown): UnknownRecord[] {
  if (Array.isArray(raw)) return raw.filter(isRecord)
  if (!isRecord(raw)) return []
  if (Array.isArray(raw.activities)) return raw.activities.filter(isRecord)
  if (Array.isArray(raw.data)) return raw.data.filter(isRecord)
  if (isRecord(raw.data) && raw.data.searchActivitiesScalar != null)
    return recordsFromJson(graphqlScalar(raw.data.searchActivitiesScalar))
  if (isRecord(raw.data) && Array.isArray(raw.data.activities))
    return raw.data.activities.filter(isRecord)
  return []
}

function graphqlScalar(raw: unknown): unknown {
  if (typeof raw !== 'string') return raw
  try {
    return JSON.parse(raw) as unknown
  } catch {
    return null
  }
}

export function garminConnectActivities(raw: unknown): GarminConnectActivityListItem[] {
  const out: GarminConnectActivityListItem[] = []
  const seen = new Set<string>()
  for (const record of recordsFromJson(raw)) {
    const id = activityId(record)
    if (!id || seen.has(id)) continue
    seen.add(id)
    out.push({ id, record })
  }
  return out
}
