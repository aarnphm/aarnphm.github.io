import assert from 'node:assert/strict'
import {
  lstatSync,
  mkdirSync,
  mkdtempSync,
  renameSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { date, fetchedAt, health } from './fixtures/garmin-health'
import { serializeStravaDetails } from './strava-detail'
import { loadStravaDataSync, loadStravaPayloadSync } from './strava-payload'
import { isRecord } from './type-guards'

test('a source edit before the first payload read invalidates the next bundle', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'quartz-strava-source-'))
  const originalDirectory = process.cwd()
  const originalInputs = globalThis.__quartzStravaPayloadSourceInputs
  const cachePath = join(directory, 'quartz/.quartz-cache/strava.json')
  const modelPath = join(directory, 'model.ts')
  mkdirSync(join(directory, 'quartz/.quartz-cache'), { recursive: true })
  writeFileSync(
    cachePath,
    JSON.stringify({
      athleteId: 1,
      auth: { refreshToken: '', obtainedAt: 0 },
      lastSync: 0,
      lastActivityStart: 0,
      activities: {},
    }),
  )
  writeFileSync(modelPath, 'export const version = 1\n')
  const earlier = new Date(Date.now() - 10_000)
  utimesSync(modelPath, earlier, earlier)
  try {
    process.chdir(directory)
    globalThis.__quartzStravaPayloadSourceInputs = [modelPath]
    const firstUrl = new URL('./strava-payload.ts', import.meta.url)
    firstUrl.searchParams.set('bundle', `first-${Date.now()}`)
    const firstBundle: typeof import('./strava-payload') = await import(firstUrl.href)

    // The watched edit happens during the build, before its first activity embed is rendered.
    writeFileSync(modelPath, 'export const version = 2\n')
    const later = new Date(Date.now() + 1_000)
    utimesSync(modelPath, later, later)
    const first = firstBundle.loadStravaDataSync('2026-10-07', null)
    assert.equal(firstBundle.loadStravaDataSync('2026-10-07', null), first)

    const secondUrl = new URL('./strava-payload.ts', import.meta.url)
    secondUrl.searchParams.set('bundle', `second-${Date.now()}`)
    const secondBundle: typeof import('./strava-payload') = await import(secondUrl.href)
    const second = secondBundle.loadStravaDataSync('2026-10-07', null)
    assert.notEqual(second, first)
    assert.equal(secondBundle.loadStravaDataSync('2026-10-07', null), second)
  } finally {
    process.chdir(originalDirectory)
    globalThis.__quartzStravaPayloadSourceInputs = originalInputs
    rmSync(directory, { recursive: true, force: true })
  }
})

test('embeds reuse emitter data and invalidate it when tracking or provider files change', () => {
  const directory = mkdtempSync(join(tmpdir(), 'quartz-strava-data-'))
  const originalDirectory = process.cwd()
  const cachePath = join(directory, 'quartz/.quartz-cache/strava.json')
  mkdirSync(join(directory, 'quartz/.quartz-cache'), { recursive: true })
  const cache = {
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: 0 },
    lastSync: Date.parse('2026-09-09T12:00:00Z'),
    lastActivityStart: 0,
    activities: {},
  }
  writeFileSync(cachePath, JSON.stringify(cache))
  const tracking = { activities: [], fueling: [], strength: [], moves: [], sauna: [] }
  try {
    process.chdir(directory)
    const first = loadStravaDataSync('2026-09-09', tracking)
    assert.equal(loadStravaDataSync('2026-09-09', tracking), first)
    assert.equal(loadStravaPayloadSync('2026-09-09', tracking), first.payload)

    const inputs = {
      weights: [
        {
          date: '2026-09-09',
          weightLbs: null,
          weightKg: null,
          windKph: 5,
          windDir: 'NW',
          race: false,
          event: null,
        },
      ],
    }
    const revised = loadStravaDataSync('2026-09-09', tracking, inputs)
    assert.notEqual(revised, first)
    assert.equal(revised.payload.health['2026-09-09'].windKph, 5)
    assert.equal(loadStravaPayloadSync('2026-09-09', tracking, inputs), revised.payload)

    writeFileSync(cachePath, JSON.stringify({ ...cache, lastSync: cache.lastSync + 1000 }))
    const later = new Date(Date.now() + 1000)
    utimesSync(cachePath, later, later)
    const refreshed = loadStravaDataSync('2026-09-09', tracking, inputs)
    assert.notEqual(refreshed, revised)
    assert.equal(refreshed.generatedAt, cache.lastSync + 1000)
    assert.equal(loadStravaPayloadSync('2026-09-09', tracking, inputs), refreshed.payload)
  } finally {
    process.chdir(originalDirectory)
    rmSync(directory, { recursive: true, force: true })
  }
})

test('emitter data follows health-cache symlinks and reloads replaced targets', () => {
  const directory = mkdtempSync(join(tmpdir(), 'quartz-health-symlink-'))
  const originalDirectory = process.cwd()
  const sourceDirectory = join(directory, 'garden/quartz/.quartz-cache')
  const publishDirectory = join(directory, 'publish')
  const publishCache = join(publishDirectory, 'quartz/.quartz-cache')
  mkdirSync(sourceDirectory, { recursive: true })
  mkdirSync(publishCache, { recursive: true })
  const garminPath = join(sourceDirectory, 'garmin.json')
  const cache = {
    version: 14,
    lastSync: 0,
    activities: {},
    health: { [date]: health },
    healthLastSync: fetchedAt,
  }
  writeFileSync(garminPath, JSON.stringify(cache))
  writeFileSync(
    join(sourceDirectory, 'strava.json'),
    JSON.stringify({
      athleteId: 1,
      auth: { refreshToken: '', obtainedAt: 0 },
      lastSync: 0,
      lastActivityStart: 0,
      activities: {
        '1': {
          id: 1,
          name: 'Fixture run',
          sportType: 'Run',
          distance: 5000,
          movingTime: 1800,
          elapsedTime: 1800,
          totalElevationGain: 0,
          startDate: `${date}T12:00:00Z`,
          startDateLocal: `${date}T08:00:00`,
          averageSpeed: 5000 / 1800,
        },
      },
    }),
  )
  for (const name of ['garmin.json', 'strava.json'])
    symlinkSync(join(sourceDirectory, name), join(publishCache, name))
  try {
    process.chdir(publishDirectory)
    const first = loadStravaDataSync(date, null)
    assert.deepEqual(first.sources.garmin?.health?.[date], health)
    assert.deepEqual(first.payload.dailyAnalytics[date]?.garminHealth, health)
    const manifest: unknown = JSON.parse(serializeStravaDetails(first.payload).manifest)
    assert.ok(isRecord(manifest) && isRecord(manifest.dailyAnalytics))
    const serializedDay = manifest.dailyAnalytics[date]
    assert.ok(isRecord(serializedDay))
    assert.deepEqual(serializedDay.garminHealth, health)
    assert.equal(first.generatedAt, fetchedAt)

    const updatedDay = {
      ...health,
      hillScore: {
        status: 'unavailable',
        fetchedAt: fetchedAt + 1000,
        attemptedAt: fetchedAt + 1000,
        value: null,
      },
    }
    const replacementPath = `${garminPath}.tmp`
    writeFileSync(
      replacementPath,
      JSON.stringify({
        ...cache,
        health: { [date]: updatedDay },
        healthLastSync: fetchedAt + 1000,
      }),
    )
    const later = new Date(Date.now() + 1000)
    utimesSync(replacementPath, later, later)
    renameSync(replacementPath, garminPath)

    const refreshed = loadStravaDataSync(date, null)
    assert.notEqual(refreshed, first)
    assert.deepEqual(refreshed.payload.dailyAnalytics[date]?.garminHealth, updatedDay)
    assert.equal(refreshed.generatedAt, fetchedAt + 1000)
    assert.equal(lstatSync(join(publishCache, 'garmin.json')).isSymbolicLink(), true)
  } finally {
    process.chdir(originalDirectory)
    rmSync(directory, { recursive: true, force: true })
  }
})
