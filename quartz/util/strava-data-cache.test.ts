import assert from 'node:assert/strict'
import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { loadStravaDataSync, loadStravaPayloadSync } from './strava-payload'

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
