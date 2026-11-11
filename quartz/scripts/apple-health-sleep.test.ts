import { toHtml } from 'hast-util-to-html'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { AppleCache } from '../plugins/stores/apple'
import type { OuraCache } from '../plugins/stores/oura'
import { triathlonCardFactory } from '../components/triathlon-day-card'
import { garminConnectSleep } from '../util/garmin-connect'
import { isSleepMetrics, resolveSleepMetrics } from '../util/sleep-metrics'
import { buildStravaData } from '../util/strava-payload'
import { buildDaySleepAnalytics } from '../util/triathlon-card/day-analytics'
import { isTriathlonDailyAnalytics } from '../util/triathlon-day-analytics'
import { resolveSleepSummary, napSummaries, sleepIsValid } from '../util/triathlon-day-sleep'
import { applyOuraSleepRows } from './sync-oura'

// Failures: overlapping sources/stages inflate sleep, in-bed becomes sleep, naps replace
// nights, offset dates drift, repeat imports lose history, legacy exports clear sleep.
test('Apple export imports through CLI and resolves sleep and naps with provenance', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'apple-health-sleep-'))
  const input = path.join(root, 'export.json')
  const sample = (
    id: string,
    start: string,
    end: string,
    value: string,
    source = 'Apple Watch',
  ) => ({
    id,
    start,
    end,
    value,
    sourceName: source,
    sourceBundleIdentifier: source === 'Apple Watch' ? 'com.apple.health' : 'com.ouraring.oura',
  })
  const samples = [
    sample('bed', '2026-11-08T22:00:00-04:00', '2026-11-09T06:00:00-04:00', 'inBed'),
    sample('core', '2026-11-08T23:00:00-04:00', '2026-11-09T05:00:00-04:00', 'asleepCore'),
    sample('deep', '2026-11-09T05:00:00-04:00', '2026-11-09T06:00:00-04:00', 'asleepDeep'),
    sample(
      'duplicate',
      '2026-11-08T23:00:00-04:00',
      '2026-11-09T06:00:00-04:00',
      'asleepUnspecified',
      'Oura',
    ),
    sample('nap', '2026-11-09T13:00:00-04:00', '2026-11-09T13:30:00-04:00', 'asleepUnspecified'),
    sample(
      'three-hour-nap',
      '2026-11-10T10:00:00-04:00',
      '2026-11-10T13:00:00-04:00',
      'asleepUnspecified',
    ),
    sample('bad', 'invalid', 'invalid', 'asleepCore'),
  ]
  const run = (source = input) =>
    execFileSync(
      path.resolve('node_modules/.bin/tsx'),
      [path.resolve('quartz/scripts/sync-apple.ts')],
      { cwd: root, env: { ...process.env, APPLE_HEALTH_FILE: source }, encoding: 'utf8' },
    )
  try {
    await fs.mkdir(path.join(root, 'content'))
    await fs.writeFile(path.join(root, 'content/triathlon.md'), '# Triathlon\n')
    await fs.writeFile(input, JSON.stringify({ days: [], sleepSamples: samples }))
    const first = run()
    const cachePath = path.join(root, 'quartz/.quartz-cache/apple-health.json')
    const cache: AppleCache = JSON.parse(await fs.readFile(cachePath, 'utf8'))
    assert.equal(cache.sleep?.['2026-11-10']?.sleep, null)
    assert.equal(napSummaries(null, null, cache.sleep?.['2026-11-10'])?.[0].totalSleepS, 10800)
    const apple = cache.sleep?.['2026-11-09']
    assert.ok(apple)
    const sleep = resolveSleepSummary(null, null, undefined, apple)
    assert.equal(sleep?.totalSleepS, 7 * 3600)
    assert.equal(sleep?.nightSource, 'apple')
    assert.equal(sleep?.deepS, 3600)
    assert.equal(sleep?.stages?.length, 2)
    assert.equal(sleep?.fieldSources?.stages, 'apple')
    assert.equal(sleep?.sleepScore, null)
    assert.ok(sleepIsValid(sleep))
    const loaded = buildStravaData(
      {
        apple: cache,
        oura: null,
        garmin: null,
        wahoo: null,
        core: null,
        weather: null,
        strava: {
          athleteId: 1,
          auth: { refreshToken: '', obtainedAt: 0 },
          lastSync: Date.parse('2026-11-10T00:00:00Z'),
          lastActivityStart: 0,
          activities: {
            '1': {
              id: 1,
              name: 'Run',
              sportType: 'Run',
              distance: 1000,
              movingTime: 600,
              elapsedTime: 600,
              totalElevationGain: 0,
              startDate: '2026-11-09T12:00:00Z',
              startDateLocal: '2026-11-09T08:00:00Z',
              averageSpeed: 1.67,
            },
          },
        },
      },
      '2026-11-09',
      null,
      {},
      root,
    )
    assert.equal(loaded.analytics.daily.find(d => d.date === '2026-11-09')?.napDurationS, 1800)
    assert.equal(loaded.payload.health['2026-11-09']?.sleepDurationS, 25200)
    assert.equal(loaded.payload.health['2026-11-09']?.sleepSources?.sleepDurationS, 'apple')
    assert.ok(isTriathlonDailyAnalytics(JSON.parse(JSON.stringify(loaded.payload.dailyAnalytics))))
    const rendered = buildDaySleepAnalytics(
      triathlonCardFactory,
      loaded.payload.dailyAnalytics['2026-11-09'],
    )
    assert.ok(rendered)
    const html = toHtml(rendered)
    assert.match(html, /data-day-sleep-source="Apple Health"/)
    assert.match(html, /data-day-sleep-start="2026-11-08T23:00:00/)
    assert.match(html, /Inferred daytime sleep session/)
    const naps = napSummaries(null, null, apple)
    assert.equal(naps?.length, 1)
    assert.equal(naps?.[0].totalSleepS, 1800)
    assert.equal(naps?.[0].nightSource, 'apple')
    assert.equal(naps?.[0].lightS, null)
    const garmin = garminConnectSleep(
      {
        dailySleepDTO: {
          calendarDate: '2026-11-09',
          sleepFromDevice: true,
          sleepStartTimestampGMT: Date.parse('2026-11-09T03:00:00Z'),
          sleepEndTimestampGMT: Date.parse('2026-11-09T10:00:00Z'),
          sleepTimeSeconds: 24000,
        },
      },
      '2026-11-09',
    )
    assert.ok(garmin)
    garmin.naps = [
      {
        id: 'garmin-nap',
        startTime: Date.parse('2026-11-09T15:00:00-04:00'),
        endTime: Date.parse('2026-11-09T15:20:00-04:00'),
        durationS: 1200,
        providerSource: null,
        feedback: null,
      },
    ]
    const oura: OuraCache = { lastSync: 0, days: {}, details: {} }
    const details = oura.details ?? {}
    applyOuraSleepRows(
      [
        {
          id: 'oura-nap',
          type: 'late_nap',
          day: '2026-11-09',
          bedtime_start: '2026-11-09T16:00:00-04:00',
          bedtime_end: '2026-11-09T16:10:00-04:00',
          total_sleep_duration: 600,
        },
      ],
      oura.days,
      details,
      '2026-11-09',
      '2026-11-09',
    )
    assert.deepEqual(
      napSummaries(details['2026-11-09'], garmin, apple)?.map(n => n.id),
      ['oura-nap'],
    )
    assert.deepEqual(
      napSummaries(null, garmin, apple)?.map(n => n.id),
      ['garmin-nap'],
    )
    assert.deepEqual(
      napSummaries(null, { ...garmin, naps: [] }, apple)?.map(n => n.id),
      naps?.map(n => n.id),
    )

    applyOuraSleepRows(
      [
        {
          id: 'oura-night',
          type: 'long_sleep',
          day: '2026-11-09',
          bedtime_start: '2026-11-08T23:00:00-04:00',
          bedtime_end: '2026-11-09T05:00:00-04:00',
          total_sleep_duration: 21600,
        },
      ],
      oura.days,
      details,
      '2026-11-09',
      '2026-11-09',
    )
    assert.equal(
      resolveSleepSummary(details['2026-11-09'], garmin, undefined, apple)?.nightSource,
      'oura',
    )
    assert.equal(
      resolveSleepSummary(details['2026-11-09'], garmin, undefined, apple)?.totalSleepS,
      21600,
    )
    assert.equal(resolveSleepSummary(null, garmin, undefined, apple)?.nightSource, 'garmin')
    assert.equal(resolveSleepSummary(null, garmin, undefined, apple)?.totalSleepS, 24000)
    assert.ok(isSleepMetrics(resolveSleepMetrics(null, null, apple), '2026-11-09'))
    run()
    assert.deepEqual(JSON.parse(await fs.readFile(cachePath, 'utf8')).sleep, cache.sleep)
    await fs.writeFile(input, JSON.stringify({ days: [{ date: '2026-11-09', activeKcal: 300 }] }))
    run()
    assert.deepEqual(JSON.parse(await fs.readFile(cachePath, 'utf8')).sleep, cache.sleep)
    const xmlPath = path.join(root, 'export.xml')
    const xml =
      '<HealthData>\n' +
      samples
        .filter(sample => sample.id !== 'bad')
        .map(
          sample =>
            `<Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="${sample.sourceName}" startDate="${sample.start}" endDate="${sample.end}" value="HKCategoryValueSleepAnalysis${sample.value[0].toUpperCase()}${sample.value.slice(1)}"/>`,
        )
        .join('\n') +
      '\n</HealthData>'
    await fs.writeFile(xmlPath, xml)
    run(xmlPath)
    const xmlCache: AppleCache = JSON.parse(await fs.readFile(cachePath, 'utf8'))
    assert.equal(xmlCache.sleep?.['2026-11-09'].sleep?.totalSleepS, 25200)
    assert.equal(xmlCache.sleep?.['2026-11-09'].naps[0]?.totalSleepS, 1800)
    const artifact = path.resolve('docs/verification/apple-health-sleep/artifacts')
    await fs.mkdir(artifact, { recursive: true })
    await fs.writeFile(path.join(artifact, 'sleep.html'), html)
    await fs.writeFile(
      path.join(artifact, 'cli-replay.json'),
      JSON.stringify({ samples, first, sleep, naps, result: 'passed' }, null, 2),
    )
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
})
