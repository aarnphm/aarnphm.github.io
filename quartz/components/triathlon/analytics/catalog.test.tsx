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
import { estimateFtpFromPowerCurve } from '../../../util/cycling-ftp'
import { health as garminHealthFixture } from '../../../util/fixtures/garmin-health'
import { resolveSleepMetrics } from '../../../util/sleep-metrics'
import { buildSwimPowerEstimate, buildSwimPowerCurveBlock } from '../../../util/swim-power'
import { DEFAULT_TRIATHLON_FORMATTER } from '../runtime/formatter'
import { ANALYTICS_CATALOG } from './catalog'
import { analyticsChartPath, AnalyticsServerPanel } from './render'

test('server swim power presents compact relative units and duration series', () => {
  const analytics = buildAnalytics(null)
  const estimate = buildSwimPowerEstimate(
    'garmin',
    [0, 40, 80, 120].map(startElapsedS => ({
      startElapsedS,
      endElapsedS: startElapsedS + 40,
      durationS: 40,
      distanceM: 25,
      stroke: 'freestyle',
    })),
  )
  assert.ok(estimate)
  analytics.swimPowerCurve = buildSwimPowerCurveBlock(
    [{ id: 42, date: '2026-09-29', swimPower: estimate }],
    '2026-10-02',
  )
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'swim-power')
  assert.ok(definition)
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /swim.*drag power/)
  assert.match(html, /idx/)
  assert.doesNotMatch(html, /2:30|reference/)
  assert.match(html, /data-series="last 6 weeks"/)
  assert.doesNotMatch(html, /W\/kg|FTP/)
})

test('server lactate threshold lists values without charts', () => {
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
  const content = definition.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.equal(content.series, undefined)
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /running heart rate · Garmin · 2026-09-08/)
  assert.match(html, /run · Garmin running estimate · 2026-09-08/)
  assert.doesNotMatch(html, /data-series=/)
})

test('dated server chart paths space points by calendar date', () => {
  assert.equal(
    analyticsChartPath([10, 10, 10], undefined, ['2026-09-01', '2026-09-02', '2026-09-11']),
    'M0.00 29.00 L10.00 29.00 L100.00 29.00',
  )
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
  assert.doesNotMatch(html, /sleep details<\/dt><dd>Sep 8/)
  assert.doesNotMatch(html, /sleep stages|hypnogram/)

  analytics.daily = [{ ...day, sleepMetrics: resolveSleepMetrics(null, garmin) }]
  const garminOnly = renderToString(
    <AnalyticsServerPanel definition={definition} data={analytics} />,
  )
  assert.match(garminOnly, /respiration<\/dt><dd>15\.0 brpm<span/)
  assert.match(garminOnly, /role="tooltip">Garmin\nlowest Pulse Ox 91%<\/span>/)
  assert.doesNotMatch(garminOnly, /Oura|data-series="sleep duration"|data-series="sleep score"/)

  analytics.meta.today = garminHealthFixture.date
  analytics.daily = [{ ...day, date: garminHealthFixture.date, garminHealth: garminHealthFixture }]
  const health = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(health, /training readiness<\/dt><dd>78/)
  assert.match(health, /Endurance Score<\/dt><dd>7918/)
  assert.match(health, /Recovery Time<\/dt><dd>6h 12m/)
  assert.doesNotMatch(health, /sleep details<\/dt>/)
})

test('server power summary exposes each estimated FTP separately from configured FTP', () => {
  const analytics = buildAnalytics(null)
  analytics.powerCurve.ftp = 287
  analytics.powerCurve.yearLabel = 2026
  analytics.powerCurve.estimatedFtp = estimateFtpFromPowerCurve([{ s: 1200, w: 243 }])
  analytics.powerCurve.estimatedFtpYear = estimateFtpFromPowerCurve([{ s: 1200, w: 300 }])
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'power')
  assert.ok(definition)
  const content = definition.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.deepEqual(
    content.values
      .filter(item => item.label.includes('FTP'))
      .map(({ label, value }) => ({ label, value })),
    [
      { label: 'FTP', value: '287 W' },
      { label: 'eFTP · last 6 weeks', value: '231 W' },
      { label: 'eFTP · all of 2026', value: '285 W' },
    ],
  )
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /eFTP · last 6 weeks<\/dt><dd>231 W/)
  assert.match(html, /eFTP · all of 2026<\/dt><dd>285 W/)
  assert.ok(html.includes('Provisional: effort may be submaximal.'))
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
        180: effort(180, 5, '2026-08-01'),
        300: effort(300, 4, '2026-08-01'),
        360: effort(360, 3.8, '2026-08-01'),
        720: effort(720, 3.5, '2026-08-01'),
        1200: effort(1200, 2, '2026-08-01'),
      },
    },
    {
      date: '2026-08-02',
      efforts: {
        5: effort(5, 12, '2026-08-02'),
        60: effort(60, 7, '2026-08-02'),
        180: effort(180, 6, '2026-08-02'),
        300: effort(300, 5, '2026-08-02'),
        360: effort(360, 4.8, '2026-08-02'),
        720: effort(720, 4.5, '2026-08-02'),
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
    ['5s', '1m', '3m', '5m', '6m', '12m', '20m'],
  )
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /data-tri-series-count="7"/)
  assert.match(html, /data-tri-series-domain="shared-zero"/)
  assert.match(html, /data-series="20m"/)
  for (const label of ['3m', '6m', '12m']) assert.ok(html.includes(`data-series="${label}"`))
  assert.equal(analyticsChartPath([10, 12], { minimum: 0, maximum: 12 }), 'M0.00 7.33 L100.00 3.00')
  assert.equal(analyticsChartPath([2, 3], { minimum: 0, maximum: 12 }), 'M0.00 24.67 L100.00 22.50')
})
