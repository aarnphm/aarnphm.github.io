import {
  assertGarminResponseAuthorized,
  garminConnectRequestHeaders,
  garminResponseSummary,
  garminUrlFor,
  type GarminConnectSession,
} from './garmin-session'

export interface GarminActivityTypeDto {
  typeId: number
  typeKey: string
  parentTypeId: number
  isHidden: boolean
  restricted: boolean
  trimmable: boolean
}

interface GarminEventTypeDto {
  typeId: number
  typeKey: string
  sortOrder: number
}

export const GARMIN_POOL_SWIM_ACTIVITY_TYPE: GarminActivityTypeDto = {
  typeId: 27,
  typeKey: 'lap_swimming',
  parentTypeId: 26,
  isHidden: false,
  restricted: false,
  trimmable: false,
}

const GARMIN_UNCATEGORIZED_EVENT_TYPE: GarminEventTypeDto = {
  typeId: 9,
  typeKey: 'uncategorized',
  sortOrder: 10,
}

function normalizedTitle(activityId: string, title: string): string {
  const normalized = title.trim().replace(/\s+/g, ' ')
  if (!normalized) throw new Error(`Garmin activity update for ${activityId} needs a title`)
  return normalized
}

async function assertUpdateAccepted(
  res: Response,
  activityId: string,
  label: string,
): Promise<void> {
  assertGarminResponseAuthorized(res)
  const text = await res.text()
  if (!res.ok)
    throw new Error(
      `Garmin ${label} update failed for ${activityId}: ${res.status} ${garminResponseSummary(res, text)}`,
    )
  const contentType = res.headers.get('content-type') ?? ''
  if (contentType.includes('text/html') || text.trimStart().startsWith('<'))
    throw new Error(
      `Garmin ${label} update returned non-API response for ${activityId}: ${garminResponseSummary(res, text)}`,
    )
}

export async function updateGarminActivityTitle(
  session: GarminConnectSession,
  base: string,
  activityId: string,
  title: string,
): Promise<void> {
  const titleText = normalizedTitle(activityId, title)
  const res = await fetch(
    garminUrlFor(base, `/activity-service/activity/${encodeURIComponent(activityId)}`),
    {
      method: 'PUT',
      headers: garminConnectRequestHeaders(session, 'application/json'),
      body: JSON.stringify({ activityName: titleText, activityId }),
    },
  )
  await assertUpdateAccepted(res, activityId, 'title')
}

export async function updateGarminActivityType(
  session: GarminConnectSession,
  base: string,
  activityId: string,
  title: string,
  activityType: GarminActivityTypeDto,
): Promise<void> {
  const titleText = normalizedTitle(activityId, title)
  const res = await fetch(
    garminUrlFor(base, `/activity-service/activity/${encodeURIComponent(activityId)}`),
    {
      method: 'PUT',
      headers: garminConnectRequestHeaders(session, 'application/json'),
      body: JSON.stringify({
        activityId,
        activityName: titleText,
        activityTypeDTO: activityType,
        eventTypeDTO: GARMIN_UNCATEGORIZED_EVENT_TYPE,
      }),
    },
  )
  await assertUpdateAccepted(res, activityId, 'type')
}
