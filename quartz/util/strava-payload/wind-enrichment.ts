import { readdirSync, readFileSync, statSync } from 'node:fs'
import { basename, resolve } from 'node:path'
import type { StravaPayload } from '../../plugins/stores/strava'
import type { MyWindsockArchive } from '../mywindsock-archive'
import { releasedOnReload } from '../module-caches'
import { parseMyWindsockGraphs, type MyWindsockArchiveReference } from '../mywindsock-graphs'
import { projectMyWindsockRoute, type MyWindsockRoute } from '../mywindsock-route'
import { isRecord } from '../type-guards'

export const readJson = <T>(path: string): T | null => {
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as T
  } catch {
    return null
  }
}

const windArchiveDirectory = (contentDirectory: string): string =>
  resolve(contentDirectory, 'triathlon', 'wind')

const windArchiveFiles = (contentDirectory: string): { path: string; key: string }[] => {
  const directory = windArchiveDirectory(contentDirectory)
  try {
    return readdirSync(directory)
      .filter(name => /^[1-9]\d*\.json$/.test(name))
      .flatMap(name => {
        const path = resolve(directory, name)
        try {
          const info = statSync(path)
          return [{ path, key: `${path}:${info.mtimeMs}:${info.size}` }]
        } catch {
          return []
        }
      })
  } catch {
    return []
  }
}

export const windArchiveStamp = (contentDirectory: string): string =>
  windArchiveFiles(contentDirectory)
    .map(file => file.key)
    .join('|')

const routeMemo = releasedOnReload(
  new Map<string, { route: MyWindsockRoute | null; archive: MyWindsockArchiveReference | null }>(),
)

// Full schema validation runs in the ingest and validate scripts; this read checks the identity
// fields and lets the projection reject non-numeric series.
const readWindRoute = (
  path: string,
): { route: MyWindsockRoute | null; archive: MyWindsockArchiveReference | null } => {
  const empty = { route: null, archive: null }
  const id = basename(path, '.json')
  const value = readJson<unknown>(path)
  if (!isRecord(value) || value.provider !== 'mywindsock' || value.schemaVersion !== 1) return empty
  const activity = value.activity
  const capture = value.browserCapture
  if (
    !isRecord(activity) ||
    activity.stravaId !== id ||
    !isRecord(capture) ||
    capture.stravaId !== id ||
    !Array.isArray(value.mappings)
  )
    return empty
  const graphs = parseMyWindsockGraphs(value, Number(id))
  return {
    route: projectMyWindsockRoute(value as unknown as MyWindsockArchive),
    archive: graphs
      ? {
          activityId: Number(id),
          capturedAt: graphs.capturedAt,
          path: `/triathlon/wind/${id}.json`,
        }
      : null,
  }
}

export function enrichMyWindsockRoutes(payload: StravaPayload, contentDirectory: string): void {
  const files = windArchiveFiles(contentDirectory)
  const live = new Set(files.map(file => file.key))
  for (const key of routeMemo.keys()) if (!live.has(key)) routeMemo.delete(key)
  for (const file of files) {
    if (!routeMemo.has(file.key)) routeMemo.set(file.key, readWindRoute(file.path))
    const source = routeMemo.get(file.key)
    const id = source?.archive?.activityId ?? source?.route?.activityId
    const detail = id ? payload.details[String(id)] : undefined
    if (source && detail && (detail.sport === 'run' || detail.sport === 'bike')) {
      if (source.route) detail.analyses.native.myWindsockRoute = source.route
      if (source.archive) detail.analyses.native.myWindsockArchive = source.archive
    }
  }
}
