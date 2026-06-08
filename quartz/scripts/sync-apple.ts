import { createReadStream } from 'node:fs'
import fs from 'node:fs/promises'
import readline from 'node:readline'
import {
  aggregateAppleRecords,
  AppleCache,
  AppleDaily,
  AppleRecord,
  matchAppleRecord,
  mergeAppleDay,
  parseAppleJson,
} from '../plugins/stores/apple'
import { joinSegments, QUARTZ } from '../util/path'

const CACHE_VERSION = 1
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'apple-health.json')

async function readCache(): Promise<AppleCache | null> {
  try {
    return JSON.parse(await fs.readFile(cacheFile, 'utf8')) as AppleCache
  } catch {
    return null
  }
}

async function parseXmlFile(path: string): Promise<AppleDaily[]> {
  const records: AppleRecord[] = []
  const rl = readline.createInterface({ input: createReadStream(path), crlfDelay: Infinity })
  for await (const line of rl) {
    const r = matchAppleRecord(line)
    if (r) records.push(r)
  }
  return aggregateAppleRecords(records)
}

async function loadEntries(path: string): Promise<AppleDaily[]> {
  if (path.endsWith('.json')) return parseAppleJson(JSON.parse(await fs.readFile(path, 'utf8')))
  return parseXmlFile(path)
}

async function main(): Promise<void> {
  const prev = await readCache()
  const days: Record<string, AppleDaily> = { ...prev?.days }

  const envFile = process.env.APPLE_HEALTH_FILE
  const candidates = envFile
    ? [envFile]
    : [
        joinSegments(QUARTZ, '.quartz-cache', 'apple-health-import.json'),
        joinSegments(QUARTZ, '.quartz-cache', 'apple-health-import.xml'),
      ]

  let touched = 0
  let read = 0
  for (const path of candidates) {
    try {
      await fs.stat(path)
    } catch {
      continue
    }
    read += 1
    const entries = await loadEntries(path)
    for (const e of entries) {
      days[e.date] = mergeAppleDay(days[e.date], e)
      touched += 1
    }
    console.log(`[apple] read ${entries.length} days from ${path}`)
  }

  if (read === 0) {
    console.log(
      '[apple] no import found. set APPLE_HEALTH_FILE=<export.xml|day.json>, or drop one at quartz/.quartz-cache/apple-health-import.{json,xml}',
    )
    if (!prev) return
  }

  const cache: AppleCache = { version: CACHE_VERSION, lastSync: Date.now(), days }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  console.log(
    `[apple] merged ${touched} day-entries → ${Object.keys(days).length} days → ${cacheFile}`,
  )
}

main().catch(err => {
  console.error(`[apple] sync failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
