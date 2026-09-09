import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import type { GarminSleepSummary } from '../../../plugins/stores/garmin'
import {
  buildAnalytics,
  type PowerToWeightDurationS,
  type PowerToWeightEffort,
} from '../../../plugins/stores/analytics'
import { applyManualSauna, buildPayload, type StravaRawCache } from '../../../plugins/stores/strava'
import { resolveSleepMetrics } from '../../../util/sleep-metrics'
import { DEFAULT_TRIATHLON_FORMATTER } from '../runtime/formatter'
import { ANALYTICS_CATALOG, ANALYTICS_PANEL_ORDER } from './catalog'
import { analyticsChartPath, AnalyticsServerPanel } from './render'

test('analytics catalog is complete and preserves the dedicated route order', () => {
  assert.deepEqual(
    ANALYTICS_CATALOG.map(panel => panel.key),
    ANALYTICS_PANEL_ORDER,
  )
  assert.equal(new Set(ANALYTICS_PANEL_ORDER).size, ANALYTICS_PANEL_ORDER.length)
})

test('server lactate threshold charts contain native running pace and heart-rate history', () => {
  const speedMps = Array.from({ length: 31 }, (_, index) => ({
    date: new Date(Date.parse('2026-09-08') - (30 - index) * 86_400_000).toISOString().slice(0, 10),
    value: 3.78,
  }))
  const analytics = buildAnalytics(null, {
    garmin: {
      lastSync: Date.parse('2026-09-08T20:00:00Z'),
      activities: {},
      runningLactateThreshold: {
        speedMps: { value: 3.75, date: '2026-09-08' },
        heartRateBpm: { value: 174, date: '2026-09-08' },
        history: { speedMps, heartRateBpm: [{ date: '2026-05-31', value: 166 }] },
      },
    },
  })
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'lactate')
  assert.ok(definition)
  const content = definition.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.deepEqual(
    content.series?.find(series => series.label === 'run pace · Garmin')?.dates,
    speedMps.map(point => point.date),
  )
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /data-series="run pace · Garmin"/)
  assert.match(html, /data-series="run heart rate · Garmin"/)
  assert.equal(analytics.engine.lactateThreshold.sports[0].projected, null)
  assert.equal(
    analyticsChartPath([10, 10, 10], undefined, ['2026-09-01', '2026-09-02', '2026-09-11']),
    'M0.00 29.00 L10.00 29.00 L100.00 29.00',
  )
})

test('server lactate threshold keeps collected Garmin history hidden before 31 pace readings', () => {
  const analytics = buildAnalytics(null, {
    garmin: {
      lastSync: Date.parse('2026-09-08T20:00:00Z'),
      activities: {},
      runningLactateThreshold: {
        speedMps: { value: 3.75, date: '2026-09-08' },
        heartRateBpm: { value: 174, date: '2026-09-08' },
        history: {
          speedMps: [{ date: '2026-09-03', value: 3.78 }],
          heartRateBpm: [{ date: '2026-05-31', value: 166 }],
        },
      },
    },
  })
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'lactate')
  assert.ok(definition)
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /heart rate · declared/)
  assert.doesNotMatch(html, /Garmin|data-series="run/)
  assert.equal(analytics.engine.lactateThreshold.runningHistory.pace.length, 2)
})

test('every analytics panel produces meaningful server markup from the real analytics model', () => {
  const analytics = buildAnalytics(null)
  for (const definition of ANALYTICS_CATALOG) {
    const content = definition.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
    assert.ok(content.title.length > 0, definition.key)
    assert.ok(content.values.length > 0, definition.key)
    const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
    assert.match(html, /data-tri-ssr="true"/)
    assert.match(html, new RegExp(`data-tri-server-panel="${definition.key}"`))
    assert.match(html, /<dl/)
  }
})

test('heat server markup includes passive HTL and sauna minutes from the shared analytics model', () => {
  const date = '2026-09-07'
  const cache: StravaRawCache = {
    athleteId: 123,
    auth: { refreshToken: 'test-token', obtainedAt: 0 },
    lastSync: Date.parse(`${date}T23:00:00Z`),
    lastActivityStart: 0,
    activities: {},
  }
  const payload = buildPayload(cache, null, null)
  applyManualSauna(
    payload,
    [
      {
        id: 8_202_609_071_830,
        stravaActivityId: null,
        garminActivityId: null,
        title: 'Sauna',
        date,
        time: '18:30',
        durationS: 3_900,
        temperatureC: 85,
        humidityPct: 11,
        cooldown: 'cold plunge',
        heatTrainingLoad: 7.7,
      },
    ],
    [],
    'America/Toronto',
  )
  const analytics = buildAnalytics(cache, { activityDetails: payload.details })
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'heat')
  assert.ok(definition)
  const content = definition.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.ok(
    content.values.some(metric => metric.label === 'sauna min' && metric.value === '65 min'),
  )
  assert.ok(
    content.values.some(metric => metric.label === 'recorded sauna HTL' && metric.value === '7.7'),
  )
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /data-series="sauna HTL"/)
  assert.match(html, /recorded sauna HTL/)
  assert.match(html, /65 min/)
})

test('server analytics markup draws source-backed series when observations exist', () => {
  const analytics = buildAnalytics(null)
  analytics.body.latestKg = 70
  analytics.body.series = [
    { date: '2026-08-01', ts: 1, kg: 71 },
    { date: '2026-08-08', ts: 2, kg: 70 },
  ]
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'body')
  assert.ok(definition)
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /class="tri-ana-ssr-chart"/)
  assert.match(html, /data-series="weight"/)
  assert.match(html, /M0\.00 3\.00 L100\.00 29\.00/)
})

test('sleep server markup includes Oura respiration and Garmin overnight metrics without stages', () => {
  const date = '2026-09-08'
  const analytics = buildAnalytics(null)
  const garmin: GarminSleepSummary = {
    source: 'garmin',
    date,
    startTime: '2026-09-08T00:50:00-04:00',
    endTime: '2026-09-08T08:15:00-04:00',
    averageBreathsPerMinute: 15,
    lowestBreathsPerMinute: 11,
    highestBreathsPerMinute: 19,
    averageSpO2: 97,
    lowestSpO2: 91,
    bodyBatteryStart: 64,
    bodyBatteryEnd: 100,
    bodyBatteryChange: 36,
    averageStress: 11,
    restlessMoments: 39,
  }
  const day = {
    date,
    load: 0,
    effort: 0,
    swimLoad: 0,
    bikeLoad: 0,
    runLoad: 0,
    ctl: 0,
    atl: 0,
    tsb: 0,
    swimCtl: 0,
    bikeCtl: 0,
    runCtl: 0,
    readiness: null,
    hrv: null,
    rhr: null,
    sleepScore: null,
    sleepDurationS: null,
    sleepMetrics: resolveSleepMetrics({ avgBreath: 14.25 }, garmin),
    tempDevC: null,
    weightKg: null,
    totalCalories: null,
    intakeKcal: null,
    warmup: false,
  }
  analytics.daily = [day]
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'sleep')
  assert.ok(definition)
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /respiration<\/dt><dd>14\.3 brpm<span/)
  assert.match(html, /Pulse Ox<\/dt><dd>97\.0%<span/)
  assert.match(html, /Body Battery change<\/dt><dd>\+36<span/)
  assert.match(html, /sleep stress<\/dt><dd>11\.0<span/)
  assert.match(html, /role="tooltip">Garmin\nlowest Pulse Ox 91%<\/span>/)
  assert.doesNotMatch(html, / · (Oura|Garmin)|<dt>lowest Pulse Ox<\/dt>/)
  assert.match(html, /sleep details<\/dt><dd>Sep 8/)
  assert.doesNotMatch(html, /sleep stages|hypnogram/)

  analytics.daily = [{ ...day, sleepMetrics: resolveSleepMetrics(null, garmin) }]
  const garminOnly = renderToString(
    <AnalyticsServerPanel definition={definition} data={analytics} />,
  )
  assert.match(garminOnly, /respiration<\/dt><dd>15\.0 brpm<span/)
  assert.match(garminOnly, /role="tooltip">Garmin\nlowest Pulse Ox 91%<\/span>/)
  assert.doesNotMatch(garminOnly, /Oura|data-series="sleep duration"|data-series="sleep score"/)
})

test('power-to-weight server series share one zero-based scale', () => {
  const analytics = buildAnalytics(null)
  const effort = (
    durationS: PowerToWeightDurationS,
    wattsPerKg: number,
    date: string,
  ): PowerToWeightEffort => ({
    durationS,
    watts: Math.round(wattsPerKg * 80),
    wattsPerKg,
    massKg: 80,
    massDate: date,
    massSource: 'tracking',
    activityId: durationS,
    activityDate: date,
  })
  analytics.powerCurve.powerToWeight.points = [
    {
      date: '2026-08-01',
      efforts: {
        5: effort(5, 10, '2026-08-01'),
        60: effort(60, 6, '2026-08-01'),
        300: effort(300, 4, '2026-08-01'),
        1200: effort(1200, 2, '2026-08-01'),
      },
    },
    {
      date: '2026-08-02',
      efforts: {
        5: effort(5, 12, '2026-08-02'),
        60: effort(60, 7, '2026-08-02'),
        300: effort(300, 5, '2026-08-02'),
        1200: effort(1200, 3, '2026-08-02'),
      },
    },
  ]
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'power')
  assert.ok(definition)
  const content = definition.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.equal(content.seriesDomain, 'shared-zero')
  assert.deepEqual(
    content.series?.map(series => series.label),
    ['5s', '1m', '5m', '20m'],
  )
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /data-tri-series-count="4"/)
  assert.match(html, /data-tri-series-domain="shared-zero"/)
  assert.match(html, /data-series="20m"/)
  assert.equal(analyticsChartPath([10, 12], { minimum: 0, maximum: 12 }), 'M0.00 7.33 L100.00 3.00')
  assert.equal(analyticsChartPath([2, 3], { minimum: 0, maximum: 12 }), 'M0.00 24.67 L100.00 22.50')
})
