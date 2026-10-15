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
