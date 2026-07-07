import {
  applyGarminSetCookies,
  garminConnectRequestHeaders,
  garminResponseSummary,
  garminUrlFor,
  type GarminConnectSession,
} from './garmin-session'

function titleHeaders(session: GarminConnectSession): HeadersInit {
  const headers = new Headers(garminConnectRequestHeaders(session, 'application/json'))
  headers.set('NK', 'NT')
  headers.set('X-HTTP-Method-Override', 'PUT')
  return headers
}

export async function updateGarminActivityTitle(
  session: GarminConnectSession,
  base: string,
  activityId: string,
  title: string,
): Promise<void> {
  const normalizedTitle = title.trim().replace(/\s+/g, ' ')
  if (!normalizedTitle) throw new Error(`Garmin title update for ${activityId} needs a title`)
  const res = await fetch(
    garminUrlFor(base, `/activity-service/activity/${encodeURIComponent(activityId)}`),
    {
      method: 'POST',
      headers: titleHeaders(session),
      body: JSON.stringify({ activityName: normalizedTitle, activityId: Number(activityId) }),
    },
  )
  const text = await res.text()
  applyGarminSetCookies(session, res.headers)
  if (!res.ok)
    throw new Error(
      `Garmin title update failed for ${activityId}: ${res.status} ${garminResponseSummary(res, text)}`,
    )
  const contentType = res.headers.get('content-type') ?? ''
  if (contentType.includes('text/html') || text.trimStart().startsWith('<'))
    throw new Error(
      `Garmin title update returned non-API response for ${activityId}: ${garminResponseSummary(res, text)}`,
    )
}
