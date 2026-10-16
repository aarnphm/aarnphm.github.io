import fs from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseWahooCache } from '../plugins/stores/wahoo'
import { planRouvyWahooPairs } from '../util/rouvy-wahoo'
import {
  readRouvyWahooPairs,
  rouvyWahooPairsPath,
  type RouvyWahooPair,
} from '../util/rouvy-wahoo-pairs'
import { readStravaCacheFile } from '../util/strava-cache-file'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord } from '../util/type-guards'
import { wahooCloudClientFromEnv } from '../util/wahoo-cloud'
import { decodeWahooFit, wahooFitSha256 } from '../util/wahoo-fit'

// Apply the requested policy prospectively; older authored links remain unchanged.
const SINCE = '2026-10-16'

async function readOptional(path: string): Promise<Buffer | null> {
  try {
    return await fs.readFile(path)
  } catch (error) {
    if (isRecord(error) && error.code === 'ENOENT') return null
    throw error
  }
}

export async function syncRouvyWahooPairs(): Promise<void> {
  const strava = await readStravaCacheFile('quartz/.quartz-cache/strava.json')
  const wahooBytes = await readOptional('quartz/.quartz-cache/wahoo.json')
  if (!strava || !wahooBytes) return
  const wahoo = parseWahooCache(JSON.parse(wahooBytes.toString('utf8')))
  if (!wahoo) return
  const plans = planRouvyWahooPairs(
    Object.values(strava.activities),
    Object.values(wahoo.activities),
    SINCE,
  )
  const path = rouvyWahooPairsPath()
  const lockPath = resolve('quartz/.quartz-cache/rouvy-wahoo-sync.lock')
  await fs.mkdir(dirname(path), { recursive: true })
  // Both provider commands invoke this step; never race their pairing writes.
  const lock = await fs.open(lockPath, 'wx')
  try {
    const pairs = readRouvyWahooPairs()
    const before = JSON.stringify(pairs)
    for (const plan of plans) {
      const route = strava.streams?.[String(plan.canonical.id)]
      if (!route?.time || route.time.length < 2 || route.latlng.length !== route.time.length) {
        console.warn(`[rouvy-wahoo] waiting for complete ROUVY streams: ${plan.canonical.id}`)
        continue
      }
      const previous = pairs.find(pair => pair.canonicalId === plan.canonical.id)
      if (previous && previous.workoutId !== plan.wahoo.workoutId)
        throw new Error(
          `ROUVY ${plan.canonical.id} already has a different archived Wahoo recording`,
        )
      if (
        pairs.some(
          pair => pair.workoutId === plan.wahoo.workoutId && pair.canonicalId !== plan.canonical.id,
        )
      )
        throw new Error(`Wahoo ${plan.wahoo.workoutId} already belongs to another ROUVY activity`)
      const date = plan.canonical.startDateLocal.slice(2, 10)
      let fitPath = previous?.fitPath ?? `triathlon/wahoo/${date}.fit`
      let bytes = await readOptional(resolve('content', fitPath))
      if (!previous && bytes && wahooFitSha256(bytes) !== plan.wahoo.sourceFile.sha256) {
        fitPath = `triathlon/wahoo/${date}-${plan.wahoo.workoutId}.fit`
        bytes = await readOptional(resolve('content', fitPath))
      }
      if (!bytes) {
        if (previous) throw new Error(`Archived Wahoo FIT is missing: ${fitPath}`)
        const client = await wahooCloudClientFromEnv()
        bytes = Buffer.from(await client.downloadFit(plan.wahoo.sourceFile.url))
        if (wahooFitSha256(bytes) !== plan.wahoo.sourceFile.sha256)
          throw new Error(`Wahoo ${plan.wahoo.workoutId} changed since provider sync`)
        decodeWahooFit(bytes)
        await fs.writeFile(resolve('content', fitPath), bytes, { flag: 'wx' })
      }
      const sha256 = wahooFitSha256(bytes)
      if (sha256 !== (previous?.sha256 ?? plan.wahoo.sourceFile.sha256))
        throw new Error(`Archived Wahoo FIT checksum mismatch: ${fitPath}`)
      const fit = decodeWahooFit(bytes)
      if (
        Date.parse(fit.startDate) !== Date.parse(plan.wahoo.startDate) ||
        fit.elapsedTimeS == null ||
        fit.distanceM == null ||
        fit.elapsedTimeS !== plan.wahoo.elapsedTimeS ||
        Math.abs(fit.distanceM - (plan.wahoo.distanceM ?? Number.NaN)) > 50 ||
        !fit.streams.watts.some(value => value != null && Number.isFinite(value))
      )
        throw new Error(`Archived Wahoo FIT session mismatch: ${fitPath}`)
      const pair: RouvyWahooPair = {
        canonicalId: plan.canonical.id,
        canonicalStartDate: plan.canonical.startDate,
        workoutId: plan.wahoo.workoutId,
        duplicateStravaId: plan.duplicate?.id ?? previous?.duplicateStravaId ?? null,
        fitPath,
        sha256,
        startDate: fit.startDate,
        elapsedTimeS: fit.elapsedTimeS,
        distanceM: fit.distanceM,
      }
      if (previous) pairs[pairs.indexOf(previous)] = pair
      else pairs.push(pair)
      if (plan.duplicate)
        console.log(
          `[rouvy-wahoo] archived ${fitPath}; keep Strava ${pair.canonicalId}; pending confirmed Strava deletion: ${plan.duplicate.id}`,
        )
    }
    if (JSON.stringify(pairs) !== before) {
      const temporary = `${path}.tmp-${process.pid}`
      await fs.writeFile(temporary, JSON.stringify({ version: 1, pairs }, null, 2) + '\n')
      await fs.rename(temporary, path)
      await refreshTriathlonRouteSource()
    }
    console.log(
      `[rouvy-wahoo] ${plans.length} unambiguous pairs checked; ${pairs.length} archived; no remote activities deleted`,
    )
  } finally {
    await lock.close()
    await fs.unlink(lockPath)
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  syncRouvyWahooPairs().catch(error => {
    console.error(`[rouvy-wahoo] ${error instanceof Error ? error.message : error}`)
    process.exitCode = 1
  })
