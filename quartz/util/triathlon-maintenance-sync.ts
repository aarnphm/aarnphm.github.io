import { randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import { dirname } from 'node:path'
import { setTimeout } from 'node:timers/promises'
import type { RawStravaActivity } from '../plugins/stores/strava'
import { readStravaCacheFile, stravaCachePath } from './strava-cache-file'
import { calculateDistances } from './triathlon-maintenance-sync/calculate'
import { maintenanceDocument } from './triathlon-maintenance-sync/document'
import { distanceRecords } from './triathlon-maintenance-sync/records'
import { isRecord } from './type-guards'

export function reconcileMaintenanceDistances(
  source: string,
  activities: Record<string, RawStravaActivity>,
) {
  const document = maintenanceDocument(source)
  if (!document) return { source, entries: [] }
  const records = distanceRecords(document.maintenance)
  const entries = calculateDistances(records, activities)
  for (const [index, record] of records.entries())
    document.setDistance(record.target, entries[index].distanceMiles)
  return { source: document.finish(), entries }
}

async function acquireLock(path: string) {
  const deadline = Date.now() + 5_000
  for (;;) {
    try {
      return await fs.open(path, 'wx')
    } catch (error) {
      if (!isRecord(error) || error.code !== 'EEXIST' || Date.now() >= deadline) throw error
      await setTimeout(50)
    }
  }
}

export async function syncMaintenanceDistances(notePath = 'content/triathlon.md'): Promise<void> {
  await fs.mkdir(dirname(stravaCachePath), { recursive: true })
  const lockPath = `${dirname(stravaCachePath)}/maintenance-sync.lock`
  const lock = await acquireLock(lockPath)
  try {
    const cache = await readStravaCacheFile(stravaCachePath)
    if (!cache) return
    const target = await fs.realpath(notePath)
    const source = await fs.readFile(target, 'utf8')
    const result = reconcileMaintenanceDistances(source, cache.activities)
    const changed = result.source !== source
    if (changed) {
      const temporary = `${target}.tmp-${randomUUID()}`
      try {
        const { mode } = await fs.stat(target)
        await fs.writeFile(temporary, result.source, { mode })
        if ((await fs.readFile(target, 'utf8')) !== source)
          throw new Error(
            'Triathlon source changed during maintenance sync; retry with the latest edits',
          )
        await fs.rename(temporary, target)
      } finally {
        await fs.rm(temporary, { force: true })
      }
    }
    const report = {
      cacheSyncedAt: new Date(cache.lastSync).toISOString(),
      method:
        'Ride matched by record bikeId; unassigned rides excluded; inclusive local dates; union of recorded usage ranges; cycles inherit chain bikeId and intersect chain usage; new tire or cycle owns shared changeover dates; numeric miles to two decimals',
      entries: result.entries,
    }
    await fs.writeFile(
      `${dirname(stravaCachePath)}/maintenance-sync.json`,
      JSON.stringify(report, null, 2) + '\n',
    )
    for (const entry of result.entries)
      if (entry.reason) console.warn(`[maintenance] ${entry.label}: ${entry.reason}`)
    console.log(
      `[maintenance] ${result.entries.length} distances reconciled${changed ? '; updated triathlon.md' : '; unchanged'}`,
    )
  } finally {
    await lock.close()
    await fs.rm(lockPath, { force: true })
  }
}
