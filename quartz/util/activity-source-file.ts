import { createHash } from 'node:crypto'
import type { WahooCloudClient } from './wahoo-cloud'
import { garminActivityFileFromArchive } from './garmin-fit'
import { fetchGarminBytes, type GarminConnectSession } from './garmin-session'
import { encodeGarminTcxActivityFit } from './garmin-tcx-fit'

export interface ActivitySourceFile {
  bytes: Uint8Array
  filename: string
  // The hash and kind describe the returned bytes; sourceKind survives conversion.
  sha256: string
  kind: 'fit' | 'tcx'
  sourceKind: 'fit' | 'tcx'
}

interface GarminActivitySource {
  id: string
}

interface WahooActivitySource {
  id: string
  workoutId: number
  fitUrl: string
  fitSha256: string
}

function garminActivityId(value: string): string {
  const id = value.startsWith('connect:') ? value.slice('connect:'.length) : value
  if (!id || !/^\d+$/.test(id)) throw new Error(`Garmin activity id is invalid: ${value}`)
  return id
}

function fileSha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

export async function fetchWahooActivityFit(
  activity: WahooActivitySource,
  client: WahooCloudClient,
): Promise<ActivitySourceFile> {
  const bytes = await client.downloadFit(activity.fitUrl)
  const sha256 = fileSha256(bytes)
  if (sha256 !== activity.fitSha256.toLowerCase())
    throw new Error(`Wahoo FIT SHA-256 changed for ${activity.id}`)
  return {
    bytes,
    filename: `wahoo-${activity.workoutId}.fit`,
    sha256,
    kind: 'fit',
    sourceKind: 'fit',
  }
}

export async function fetchGarminActivityFile(
  activity: GarminActivitySource,
  session: GarminConnectSession,
  base: string,
): Promise<ActivitySourceFile> {
  const sourceId = garminActivityId(activity.id)
  const archive = await fetchGarminBytes(
    session,
    base,
    `/download-service/files/activity/${encodeURIComponent(sourceId)}`,
  )
  const sourceFile = garminActivityFileFromArchive(archive)
  return {
    bytes: sourceFile.bytes,
    filename: `garmin-${sourceId}.${sourceFile.kind}`,
    sha256: fileSha256(sourceFile.bytes),
    kind: sourceFile.kind,
    sourceKind: sourceFile.kind,
  }
}

export async function fetchGarminActivityFit(
  activity: GarminActivitySource,
  session: GarminConnectSession,
  base: string,
): Promise<ActivitySourceFile> {
  const source = await fetchGarminActivityFile(activity, session, base)
  if (source.kind === 'fit') return source
  const sourceId = garminActivityId(activity.id)
  const bytes = encodeGarminTcxActivityFit(source.bytes, sourceId).bytes
  return {
    bytes,
    filename: `garmin-${sourceId}.fit`,
    sha256: fileSha256(bytes),
    kind: 'fit',
    sourceKind: source.kind,
  }
}
