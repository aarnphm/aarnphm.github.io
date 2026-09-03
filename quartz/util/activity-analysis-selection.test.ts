import assert from 'node:assert/strict'
import test from 'node:test'
import type { ActivityAnalyses } from '../plugins/stores/strava'
import { selectActivityAnalysisSummary } from './activity-analysis-selection'

const analyses = (): ActivityAnalyses => ({
  native: { myWindsock: null, pelotan: null },
  derived: { environment: null, uvScore: null, apparentWind: null },
})

test('selects exact provider values ahead of Garden estimates', () => {
  const value = analyses()
  value.native.myWindsock = {
    source: 'provider-native',
    provider: 'mywindsock',
    transport: 'strava-description',
    schemaVersion: 1,
    activityId: 101,
    retrievedAt: 1,
    weatherImpactPct: 0.6,
    cdaM2: null,
    feelsLikeElevationM: null,
    headwindPct: null,
    headwindMinKph: null,
    headwindMaxKph: null,
    longestHeadwindS: null,
    airSpeedKph: null,
    averageTemperatureC: null,
    precipitationProbabilityPct: null,
    precipitationRateMmPerHour: null,
  }
  value.native.pelotan = {
    source: 'provider-native',
    provider: 'pelotan',
    transport: 'strava-description',
    schemaVersion: 1,
    activityId: 101,
    retrievedAt: 1,
    score: 83,
    rawBand: 'High',
    severity: 'high',
    averageUvIndex: 2.3,
    averageTemperatureC: 20,
    averageCloudCoverPct: 42,
  }

  assert.deepEqual(selectActivityAnalysisSummary(value), {
    weatherImpact: { valuePct: 0.6 },
    uvExposure: {
      kind: 'pelotan-score',
      label: 'uv load™',
      score: 83,
      severity: 'high',
      rawBand: 'High',
    },
  })
})

test('falls through Garden score to complete ambient SED', () => {
  const value = analyses()
  value.derived.environment = {
    source: 'garden-estimate',
    formulaId: 'garden-environment-v1',
    formulaVersion: 1,
    inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1',
    normalizationVersion: 1,
    computedAt: 2,
    inputAsOf: 1,
    temporalSamplingModel: 'weatherkit-hourly-piecewise-constant',
    spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint',
    summary: {
      averageUvIndex: 2,
      peakUvIndex: 4,
      uviHours: 2,
      ambientSed: 1.8,
      averageAmbientTemperatureC: 20,
      averageCloudCoverPct: 42,
      daylightCoveragePct: 100,
      weatherCoveragePct: 100,
      coveredDurationS: 3_600,
      elapsedDurationS: 3_600,
    },
    doseClocks: { elapsedSed: 1.8, movingTelemetrySed: 1.6 },
    coverage: { weatherPct: 100, uvPct: 100, temperaturePct: 100, cloudPct: 100, daylightPct: 100 },
    samples: [],
    attribution: null,
  }
  assert.equal(selectActivityAnalysisSummary(value).uvExposure?.kind, 'garden-sed')

  value.derived.uvScore = {
    ...value.derived.environment,
    formulaId: 'garden-uv-score-v1',
    score: 17,
    severity: 'low',
    doseClock: 'elapsed',
    doseSed: 1.8,
    coefficientSed: 10,
    calibrationVersion: 1,
  }
  assert.equal(selectActivityAnalysisSummary(value).uvExposure?.kind, 'garden-score')
})
