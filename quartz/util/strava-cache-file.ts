import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { dirname } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import type { StravaRawCache } from '../plugins/stores/strava'
import { joinSegments, QUARTZ } from './path'
import { requireStravaId, stravaObjectId } from './strava-id'
import { isRecord } from './type-guards'

export const stravaCachePath = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')

function isMissingFile(error: unknown): boolean {
  return isRecord(error) && error.code === 'ENOENT'
}

function parseStravaCache(source: string, path: string): StravaRawCache {
  try {
    const cache = JSON.parse(source)
    if (!isRecord(cache) || !isRecord(cache.activities) || !isRecord(cache.auth))
      throw new Error('Missing cache records')
    cache.athleteId =
      cache.athleteId === 0 || cache.athleteId === null ? null : requireStravaId(cache.athleteId)
    for (const [key, activity] of Object.entries(cache.activities)) {
      if (!isRecord(activity)) throw new Error('Invalid activity record')
      const id = stravaObjectId(activity)
      if (key !== id) throw new Error('Activity key does not match its exact ID')
      activity.id = id
      delete activity.id_str
    }
    for (const field of ['streams', 'activityDetails', 'geo']) {
      const records = cache[field]
      if (records === undefined) continue
      if (!isRecord(records)) throw new Error(`Invalid ${field} records`)
      for (const key of Object.keys(records)) requireStravaId(key)
    }
    return cache as unknown as StravaRawCache
  } catch (error) {
    throw new Error(`Strava cache at ${path} is invalid`, { cause: error })
  }
}

export async function readStravaCacheFile(path: string): Promise<StravaRawCache | null> {
  let source: string
  try {
    source = await fsp.readFile(path, 'utf8')
  } catch (error) {
    if (isMissingFile(error)) return null
    throw error
  }
  return parseStravaCache(source, path)
}

export function readStravaCacheFileSync(path: string): StravaRawCache | null {
  let source: string
  try {
    source = fs.readFileSync(path, 'utf8')
  } catch (error) {
    if (isMissingFile(error)) return null
    throw error
  }
  return parseStravaCache(source, path)
}

export async function writeStravaCacheFile(path: string, cache: StravaRawCache): Promise<void> {
  const validated = parseStravaCache(JSON.stringify(cache), path)
  const target = await fsp.realpath(path).catch(() => path)
  const temporary = `${target}.tmp-${process.pid}-${randomUUID()}`
  await fsp.mkdir(dirname(target), { recursive: true })
  try {
    await fsp.writeFile(temporary, JSON.stringify(validated, null, 2))
    await fsp.rename(temporary, target)
  } finally {
    await fsp.rm(temporary, { force: true })
  }
}

// Serialize read/modify/replace across CLI processes, including symlinked cache paths.
export async function updateStravaCacheFile(
  path: string,
  update: (current: StravaRawCache | null) => StravaRawCache,
): Promise<void> {
  const target = await fsp.realpath(path).catch(() => path)
  await fsp.mkdir(dirname(target), { recursive: true })
  const lockPath = `${target}.lock`
  let lock: Awaited<ReturnType<typeof fsp.open>> | undefined
  for (let attempt = 0; !lock; attempt++) {
    try {
      lock = await fsp.open(lockPath, 'wx')
    } catch (error) {
      if (!isRecord(error) || error.code !== 'EEXIST' || attempt >= 500) throw error
      await delay(20)
    }
  }
  try {
    await writeStravaCacheFile(target, update(await readStravaCacheFile(target)))
  } finally {
    await lock.close()
    await fsp.unlink(lockPath)
  }
}
