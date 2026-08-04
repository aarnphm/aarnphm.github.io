import assert from 'node:assert/strict'
import test from 'node:test'
import { buildMatchedRides } from './matched-rides'
import { haversineMeters, type RawStravaActivity, type StravaStreams } from './strava'

const DAY_MS = 86_400_000

interface RideMetrics {
  distanceM: number
  elevationM: number
  averageWatts: number
  normalizedWatts: number | null
  deviceWatts: boolean
}

const ride = (
  id: number,
  dayOffset: number,
  name: string,
  metrics: RideMetrics,
): RawStravaActivity => {
  const startDate = new Date(Date.parse('2026-07-01T12:00:00Z') + dayOffset * DAY_MS)
  return {
    id,
    name,
    sportType: 'Ride',
    distance: metrics.distanceM,
    movingTime: Math.round(metrics.distanceM / 7),
    elapsedTime: Math.round(metrics.distanceM / 6.8),
    totalElevationGain: metrics.elevationM,
    startDate: startDate.toISOString(),
    startDateLocal: startDate.toISOString(),
    averageSpeed: 7,
    averageWatts: metrics.averageWatts,
    weightedAverageWatts: metrics.normalizedWatts ?? undefined,
    maxWatts: metrics.averageWatts * 3,
    deviceWatts: metrics.deviceWatts,
    sufferScore: 20 + dayOffset,
  }
}

const route = (offset: number, reverse = false, diverging = false): StravaStreams => {
  const latlng = Array.from({ length: 240 }, (_, index): [number, number] => {
    const fraction = index / 239
    const divergence = diverging ? Math.sin(Math.PI * fraction) * 0.012 : 0
    return [
      43.65 + fraction * 0.09 + Math.sin(index * 1.7) * offset,
      -79.4 + divergence + Math.cos(index * 1.3) * offset,
    ]
  })
  if (reverse) latlng.reverse()
  const distance: number[] = [0]
  for (let index = 1; index < latlng.length; index++) {
    const previous = latlng[index - 1]
    const current = latlng[index]
    distance.push(
      distance[index - 1] + haversineMeters(previous[0], previous[1], current[0], current[1]),
    )
  }
  return { latlng, distance, altitude: latlng.map((_, index) => 100 + Math.sin(index / 12) * 20) }
}

test('prioritizes repeated routes and falls back to complete ride characteristics', () => {
  const routeMetrics: RideMetrics = {
    distanceM: 30_000,
    elevationM: 220,
    averageWatts: 175,
    normalizedWatts: 190,
    deviceWatts: true,
  }
  const fallbackMetrics: RideMetrics = {
    distanceM: 42_000,
    elevationM: 410,
    averageWatts: 205,
    normalizedWatts: 218,
    deviceWatts: true,
  }
  const activities = [
    ride(1, 0, 'route one', routeMetrics),
    ride(2, 2, 'route one again', { ...routeMetrics, averageWatts: 185 }),
    ride(3, 4, 'climbing ride', fallbackMetrics),
    ride(4, 6, 'similar climbing ride', {
      ...fallbackMetrics,
      distanceM: 44_000,
      elevationM: 430,
      averageWatts: 212,
      normalizedWatts: 224,
    }),
    ride(5, 8, 'estimated power lookalike', {
      ...fallbackMetrics,
      deviceWatts: false,
      normalizedWatts: null,
    }),
  ]
  const run = ride(6, 10, 'run on the same route', routeMetrics)
  run.sportType = 'Run'
  activities.push(run)

  const matched = buildMatchedRides(activities, {
    '1': route(0),
    '2': route(0.000006),
    '3': route(0, false, true),
    '4': route(0, true),
    '6': route(0),
  })

  assert.equal(matched.candidateRideCount, 5)
  assert.equal(matched.matchedActivityCount, 4)
  assert.equal(matched.routeMatchedActivityCount, 2)
  assert.equal(matched.characteristicMatchedActivityCount, 2)
  assert.equal(matched.groups.length, 2)
  assert.deepEqual(
    matched.groups.find(group => group.match === 'route')?.efforts.map(effort => effort.id),
    [1, 2],
  )
  const characteristicGroup = matched.groups.find(group => group.match === 'characteristics')
  assert.deepEqual(
    characteristicGroup?.efforts.map(effort => effort.id),
    [3, 4],
  )
  assert.equal(characteristicGroup?.powerMetric, 'normalized')
  assert.equal(characteristicGroup?.averagePowerWatts, 221)
  assert.deepEqual(
    characteristicGroup?.efforts.map(effort => effort.powerSource),
    ['device', 'device'],
  )
  assert.deepEqual(matched.method, {
    route: {
      source: 'gps',
      sampleSpacingM: 200,
      maximumSampleDistanceM: 100,
      minimumDistanceRatio: 0.8,
      minimumOrderedCoverage: 0.82,
    },
    characteristics: {
      source: 'activity-summary',
      minimumDistanceRatio: 0.75,
      minimumElevationGainRatio: 0.65,
      minimumClimbingDensityRatio: 0.7,
      minimumAveragePowerRatio: 0.85,
      minimumNormalizedPowerRatio: 0.85,
      powerSourceMustMatch: true,
    },
  })
  assert.doesNotMatch(JSON.stringify(matched), /"(?:lat|lng|latlng|streams)"/)
})

test('uses complete-link fallback so a marginal ride cannot bridge characteristic groups', () => {
  const activities = [
    ride(1, 0, 'lower power', {
      distanceM: 20_000,
      elevationM: 100,
      averageWatts: 180,
      normalizedWatts: 190,
      deviceWatts: true,
    }),
    ride(2, 1, 'middle power', {
      distanceM: 22_000,
      elevationM: 110,
      averageWatts: 195,
      normalizedWatts: 205,
      deviceWatts: true,
    }),
    ride(3, 2, 'higher power', {
      distanceM: 24_000,
      elevationM: 125,
      averageWatts: 215,
      normalizedWatts: 225,
      deviceWatts: true,
    }),
  ]

  const matched = buildMatchedRides(activities, {})

  assert.equal(matched.groups.length, 1)
  assert.equal(matched.groups[0].match, 'characteristics')
  assert.deepEqual(
    matched.groups[0].efforts.map(effort => effort.id),
    [1, 2],
  )
  assert.equal(matched.characteristicMatchedActivityCount, 2)
})
