import assert from 'node:assert/strict'
import test from 'node:test'
import {
  parseActivityProviderReports,
  parseMyWindsockReport,
  parsePelotanReport,
} from './activity-provider-reports'

const activityId = 19_943_165_126
const retrievedAt = Date.parse('2026-09-02T12:00:00Z')

test('parses the final Pelotan report and preserves its native band', () => {
  const report = parsePelotanReport(
    `Original description
── pelotan.cc/uv UV Load™ Analysis ──
UV Load 18 — Light
Avg UV 1.5
🌡️ 20°C ☁️ 42%

── pelotan.cc/uv UV Load™ Analysis ──
UV Load 83 — High
Avg UV 2.3
🌡️ 20°C ☁️ 42%`,
    activityId,
    retrievedAt,
  )

  assert.deepEqual(report, {
    source: 'provider-native',
    transport: 'strava-description',
    schemaVersion: 1,
    activityId,
    retrievedAt,
    provider: 'pelotan',
    score: 83,
    rawBand: 'High',
    severity: 'high',
    averageUvIndex: 2.3,
    averageTemperatureC: 20,
    averageCloudCoverPct: 42,
  })
})

test('normalizes Pelotan Light and Low to the same severity', () => {
  for (const rawBand of ['Light', 'Low']) {
    const report = parsePelotanReport(
      `-- pelotan.cc/uv UV Load Analysis --\nUV Load 18 — ${rawBand}`,
      activityId,
      retrievedAt,
    )
    assert.equal(report?.rawBand, rawBand)
    assert.equal(report?.severity, 'low')
  }
})

test('keeps valid zero values and rejects values outside provider bounds', () => {
  const report = parsePelotanReport(
    '-- pelotan.cc/uv UV Load Analysis --\nUV Load 0 — Negligible\nAvg UV 0\nTemperature 0 C\nCloud 0%',
    activityId,
    retrievedAt,
  )
  assert.equal(report?.score, 0)
  assert.equal(report?.averageUvIndex, 0)
  assert.equal(report?.averageTemperatureC, 0)
  assert.equal(report?.averageCloudCoverPct, 0)
  assert.equal(
    parsePelotanReport(
      '-- pelotan UV Load Analysis --\nUV Load 101 — Extreme',
      activityId,
      retrievedAt,
    ),
    null,
  )
})

test('parses and normalizes the MyWindsock report fields', () => {
  const report = parseMyWindsockReport(
    `-- myWindsock Report --
CdA: 0.324
Feels Like Elevation: +74 ft
Weather Impact: -0.6%
Headwind: 56% @ 4.2–12 mph
Longest Headwind: 03h32m41s
Air Speed: 16.8 mph
Temperature: 68°F
Precipitation: 25% @ 0.02 Inch/hr
-- END --`,
    activityId,
    retrievedAt,
  )

  assert.equal(report?.weatherImpactPct, -0.6)
  assert.equal(report?.cdaM2, 0.324)
  assert.equal(report?.feelsLikeElevationM, 22.555)
  assert.equal(report?.headwindPct, 56)
  assert.equal(report?.headwindMinKph, 6.759)
  assert.equal(report?.headwindMaxKph, 19.312)
  assert.equal(report?.longestHeadwindS, 12_761)
  assert.equal(report?.airSpeedKph, 27.037)
  assert.equal(report?.averageTemperatureC, 20)
  assert.equal(report?.precipitationProbabilityPct, 25)
  assert.equal(report?.precipitationRateMmPerHour, 0.508)
})

test('requires a recognized provider header and never returns the raw description', () => {
  assert.deepEqual(parseActivityProviderReports('UV Load 83 — High', activityId, retrievedAt), {
    myWindsock: null,
    pelotan: null,
  })
  const reports = parseActivityProviderReports(
    '-- myWindsock Report --\nWeather Impact: 0%\n-- END --\n-- pelotan UV Load Analysis --\nUV Load 3 — Negligible',
    activityId,
    retrievedAt,
  )
  assert.equal(reports.myWindsock?.weatherImpactPct, 0)
  assert.equal(reports.pelotan?.score, 3)
  assert.equal(JSON.stringify(reports).includes('Weather Impact:'), false)
})

test('parses the documented MyWindsock field spellings and units', () => {
  const report = parseMyWindsockReport(
    `-- myWindsock Report --
Weather Impact: 1.7%
Headwind: 49% @ 3.6-6.1m/s
Longest Headwind: 01h 37m 33s
Feels Like Elev: 1308.9m
Air Speed: 18.4km/h
Temp: 13.3-17°C
Precip: 0% @ 0 Inch/hr
-- END --`,
    activityId,
    retrievedAt,
  )

  assert.equal(report?.weatherImpactPct, 1.7)
  assert.equal(report?.headwindPct, 49)
  assert.equal(report?.headwindMinKph, 12.96)
  assert.equal(report?.headwindMaxKph, 21.96)
  assert.equal(report?.longestHeadwindS, 5_853)
  assert.equal(report?.feelsLikeElevationM, 1_308.9)
  assert.equal(report?.airSpeedKph, 18.4)
  assert.equal(report?.averageTemperatureC, 15.2)
  assert.equal(report?.precipitationProbabilityPct, 0)
  assert.equal(report?.precipitationRateMmPerHour, 0)
})

test('parses sanitized live provider writeback spellings', () => {
  const reports = parseActivityProviderReports(
    `-- myWindsock Report --
Aerodynamics: 0.324 CdA
Feels Like Elev™: 74ft
Weather Impact™: 0.6%
Headwind: 56% @ 4.2 - 12 mph
Longest Headwind: 3h 32m 41s
Air Speed: 16.8 mph
Precip: 0% @ 0 Inch/hr
-- END --
Pelotan UV Load Analysis, via pelotan.cc/uv
🔴 UV Load 83 — High
Serious UV stress — plan recovery and protect skin.
Avg UV 2.3 · 🌡️ 20 °C · ☁️ 42% cloud`,
    activityId,
    retrievedAt,
  )

  assert.equal(reports.pelotan?.score, 83)
  assert.equal(reports.pelotan?.severity, 'high')
  assert.equal(reports.pelotan?.averageUvIndex, 2.3)
  assert.equal(reports.pelotan?.averageTemperatureC, 20)
  assert.equal(reports.pelotan?.averageCloudCoverPct, 42)
  assert.equal(reports.myWindsock?.weatherImpactPct, 0.6)
  assert.equal(reports.myWindsock?.cdaM2, 0.324)
  assert.equal(reports.myWindsock?.longestHeadwindS, 12_761)
  assert.equal(reports.myWindsock?.airSpeedKph, 27.037)
})

test('uses the final valid block and keeps provider fields inside their delimiters', () => {
  const reports = parseActivityProviderReports(
    `-- pelotan.cc/uv UV Load Analysis --
UV Load 18 — Light
Avg UV 1.5

-- pelotan.cc/uv UV Load Analysis --
UV Load nope — High

-- myWindsock Report --
Weather Impact: -2%
-- END --
Temperature: 68°F

-- myWindsock Report --
malformed
-- END --`,
    activityId,
    retrievedAt,
  )

  assert.equal(reports.pelotan?.score, 18)
  assert.equal(reports.pelotan?.rawBand, 'Light')
  assert.equal(reports.myWindsock?.weatherImpactPct, -2)
  assert.equal(reports.myWindsock?.averageTemperatureC, null)
})

test('rejects undelimited provider-shaped prose', () => {
  assert.deepEqual(
    parseActivityProviderReports(
      'myWindsock Report\nWeather Impact: 2%\npelotan UV Load Analysis\nUV Load 83 — High',
      activityId,
      retrievedAt,
    ),
    { myWindsock: null, pelotan: null },
  )
})
