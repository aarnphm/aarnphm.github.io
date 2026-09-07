import assert from 'node:assert/strict'
import test from 'node:test'
import type { ManualSaunaEntry } from './tracking'
import { isRecord } from '../../util/type-guards'
import { buildAnalytics, buildDataFeed } from './analytics'
import {
  applyManualSauna,
  buildPayload,
  type RawStravaActivity,
  type StravaRawCache,
} from './strava'

const date = '2026-05-12'

function fixture(entries: readonly Partial<ManualSaunaEntry>[], raw: RawStravaActivity[] = []) {
  const cache: StravaRawCache = {
    athleteId: 123,
    auth: { refreshToken: 'test-token', obtainedAt: 0 },
    lastSync: Date.parse(`${date}T23:00:00Z`),
    lastActivityStart: 0,
    activities: Object.fromEntries(raw.map(activity => [String(activity.id), activity])),
    streams: Object.fromEntries(
      raw.map(activity => [
        String(activity.id),
        {
          latlng: [
            [43.6, -79.4],
            [43.61, -79.41],
          ],
          altitude: [100, 100],
          distance: [0, activity.distance],
        },
      ]),
    ),
  }
  const payload = buildPayload(cache, null, null)
  applyManualSauna(
    payload,
    entries.map((entry, i) => ({
      id: 8_202_605_121_830 + i,
      stravaActivityId: null,
      garminActivityId: null,
      title: 'Evening sauna',
      date,
      time: '18:30',
      durationS: 3_900,
      temperatureC: 85,
      humidityPct: 11,
      cooldown: 'cold plunge',
      heatTrainingLoad: 7.7,
      ...entry,
    })),
    [],
    'America/Toronto',
  )
  return { cache, payload }
}

function ride(id = 1): RawStravaActivity {
  return {
    id,
    name: 'Warm ride',
    sportType: 'Ride',
    distance: 10_000,
    movingTime: 1_800,
    elapsedTime: 1_800,
    totalElevationGain: 0,
    startDate: `${date}T12:00:00Z`,
    startDateLocal: `${date}T08:00:00`,
    averageSpeed: 10_000 / 1_800,
    averageTemp: 28,
  }
}

test('a standalone sauna supplies HTL dose, metadata, and heat dates with no raw training activity', () => {
  const { cache, payload } = fixture([{}])
  const analytics = buildAnalytics(cache, {
    activityDetails: payload.details,
    timeZone: 'America/Toronto',
  })
  const { heat } = analytics
  assert.equal(heat.currentPct, 5.5)
  assert.equal(heat.series[0].dose, 0.77)
  assert.equal(heat.lastHeatDate, date)
  assert.equal(heat.lastObservedDate, date)
  assert.equal(heat.latestTemperatureC, null)
  assert.equal(heat.heatMinutes14d, 0)
  assert.equal(heat.heatDays14d, 1)
  assert.equal(heat.saunaMinutes14d, 65)
  assert.equal(heat.saunaHtl14d, 7.7)
  assert.equal(heat.saunaSessions14d, 1)
  assert.equal(heat.coveragePct, 100)
  assert.equal(heat.sourceCounts['manual-sauna'], 1)
  assert.equal(heat.activities[0].source, 'manual-sauna')
  assert.equal(heat.activities[0].heatStrainIndex, null)
  assert.equal(heat.activities[0].temperatureC, null)
  assert.deepEqual(heat.activities[0].sauna, {
    ...Object.values(payload.details)[0].sauna,
    durationMinutes: 65,
  })
  assert.equal(analytics.activities[0].load, 0)
  assert.ok(analytics.daily.every(day => day.load === 0))
  assert.deepEqual(JSON.parse(JSON.stringify(heat)), heat)
  const feed = buildDataFeed(cache, analytics, { activityDetails: payload.details })
  const row = feed
    .split('\n')
    .filter(Boolean)
    .map((line): unknown => JSON.parse(line))
    .filter(isRecord)
    .find(record => record.kind === 'activity')
  assert.equal(row?.skipTraining, true)
})

test('sauna and routed workout heat combine under one daily credit without mixing temperatures', () => {
  const { cache, payload } = fixture([{}], [ride()])
  const { heat } = buildAnalytics(cache, { activityDetails: payload.details })
  assert.equal(heat.series[0].dose, 1)
  assert.equal(heat.series[0].source, 'mixed')
  assert.equal(heat.series[0].temperatureC, 28)
  assert.equal(heat.series[0].heatStrainIndex, null)
  assert.equal(heat.series[0].hotMinutes, 30)
  assert.equal(heat.series[0].saunaMinutes, 65)
  assert.equal(heat.series[0].saunaHtl, 7.7)
  assert.equal(heat.heatDays14d, 1)
  assert.equal(
    heat.activities.find(activity => activity.sport === 'run'),
    undefined,
  )
})

test('multiple sauna sessions retain recorded HTL and cap the daily passive dose at 0.8', () => {
  const { cache, payload } = fixture([{}, { time: '20:00' }])
  const { heat } = buildAnalytics(cache, { activityDetails: payload.details })
  assert.equal(heat.series[0].dose, 0.8)
  assert.equal(heat.saunaHtl14d, 15.4)
  assert.equal(heat.saunaSessions14d, 2)
  assert.equal(heat.saunaMinutes14d, 130)
  assert.equal(heat.heatDays14d, 1)
})

test('missing sauna HTL remains visible and reduces dose coverage while zero remains observed', () => {
  for (const heatTrainingLoad of [null, 0, Number.NaN, -1]) {
    const { cache, payload } = fixture([{ heatTrainingLoad }])
    const { heat } = buildAnalytics(cache, { activityDetails: payload.details })
    assert.equal(heat.activities.length, 1)
    assert.equal(heat.saunaMinutes14d, 65)
    assert.equal(heat.series[0].dose, 0)
    assert.equal(heat.lastHeatDate, null)
    assert.equal(heat.saunaHtl14d, heatTrainingLoad === 0 ? 0 : null)
    assert.equal(heat.coveragePct, heatTrainingLoad === 0 ? 100 : 0)
    assert.equal(heat.observedMinutes, heatTrainingLoad === 0 ? 65 : 0)
    assert.equal(heat.confidence, heatTrainingLoad === 0 ? 'low' : 'none')
  }
})

test('a linked sauna contributes once even when its raw recording looks like a routed hot ride', () => {
  const { cache, payload } = fixture([{ stravaActivityId: 1 }], [ride()])
  const analytics = buildAnalytics(cache, { activityDetails: payload.details })
  assert.equal(analytics.heat.activities.length, 1)
  assert.equal(analytics.heat.activities[0].id, 1)
  assert.equal(analytics.heat.activities[0].sauna?.durationMinutes, 30)
  assert.equal(analytics.heat.series[0].dose, 0.77)
  assert.equal(analytics.heat.sourceCounts.strava, 0)
  assert.equal(analytics.heat.heatMinutes14d, 0)
  assert.equal(analytics.activities[0].sport, 'sauna')
  assert.equal(analytics.activities[0].load, 0)
})

test('sauna obeys the since boundary and decays after the existing three-day grace period', () => {
  const { cache, payload } = fixture([{}, { date: '2026-05-11' }])
  cache.lastSync = Date.parse('2026-05-16T23:00:00Z')
  const { heat } = buildAnalytics(cache, { activityDetails: payload.details, since: date })
  assert.equal(heat.activities.length, 1)
  assert.equal(heat.series.length, 5)
  assert.deepEqual(
    heat.series.map(day => day.acclimatisationPct),
    [5.5, 5.5, 5.5, 5.5, 5.4],
  )
  assert.equal(heat.lastHeatDate, date)
})

test('sauna dates extend the analytics window beyond raw activities', () => {
  const { cache, payload } = fixture([{ date: '2026-05-11' }, { date: '2026-05-13' }], [ride()])
  const { heat } = buildAnalytics(cache, { activityDetails: payload.details })
  assert.equal(heat.series[0].date, '2026-05-11')
  assert.equal(heat.series.at(-1)?.date, '2026-05-13')
  assert.equal(heat.lastHeatDate, '2026-05-13')
})
