import assert from 'node:assert/strict'
import test from 'node:test'
import { buildMatchedRuns } from './matched-runs'
import { haversineMeters, type RawStravaActivity, type StravaStreams } from './strava'

const DAY_MS = 86_400_000

const activity = (
  id: number,
  dayOffset: number,
  distanceM: number,
  name: string,
): RawStravaActivity => {
  const startDate = new Date(Date.parse('2026-07-01T12:00:00Z') + dayOffset * DAY_MS)
  return {
    id,
    name,
    sportType: 'Run',
    distance: distanceM,
    movingTime: Math.round((distanceM / 1_000) * 360),
    elapsedTime: Math.round((distanceM / 1_000) * 370),
    totalElevationGain: 10,
    startDate: startDate.toISOString(),
    startDateLocal: startDate.toISOString(),
    averageSpeed: distanceM / Math.round((distanceM / 1_000) * 360),
    sufferScore: 12 + dayOffset,
  }
}

const route = (offset: number, reverse = false, diverging = false): StravaStreams => {
  const latlng = Array.from({ length: 160 }, (_, index): [number, number] => {
    const fraction = index / 159
    const divergence = diverging ? Math.sin(Math.PI * fraction) * 0.004 : 0
    return [
      43.65 + fraction * 0.018 + Math.sin(index * 1.7) * offset,
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
  return { latlng, distance, altitude: latlng.map(() => 100) }
}

test('groups repeated same-direction GPS routes and rejects route lookalikes', () => {
  const activities = [
    activity(1, 0, 4_800, 'first route effort'),
    activity(2, 2, 5_250, 'second route effort'),
    activity(3, 4, 5_450, 'third route effort'),
    activity(4, 6, 4_630, 'latest route effort'),
    activity(5, 8, 4_900, 'reverse direction'),
    activity(6, 10, 4_900, 'different middle'),
    activity(7, 12, 4_900, 'missing GPS'),
  ]
  const ride = activity(8, 14, 4_900, 'same route on a bike')
  ride.sportType = 'Ride'
  activities.push(ride)
  const matched = buildMatchedRuns(activities, {
    '1': route(0),
    '2': route(0.000006),
    '3': route(0.000009),
    '4': route(0.000012),
    '5': route(0, true),
    '6': route(0, false, true),
    '8': route(0),
  })

  assert.equal(matched.candidateRunCount, 6)
  assert.equal(matched.matchedActivityCount, 4)
  assert.equal(matched.groups.length, 1)
  assert.deepEqual(
    matched.groups[0].efforts.map(effort => effort.id),
    [1, 2, 3, 4],
  )
  assert.equal(matched.groups[0].id, '1')
  assert.equal(matched.groups[0].routeDistanceKm, 5.03)
  assert.equal(matched.groups[0].averagePaceSPerKm, 360.011)
  assert.equal(matched.groups[0].fastestPaceSPerKm, 360)
  assert.equal(matched.groups[0].slowestPaceSPerKm, 360)
  assert.deepEqual(
    matched.groups[0].efforts.map(effort => effort.relativeEffort),
    [12, 14, 16, 18],
  )
  assert.deepEqual(matched.method, {
    source: 'gps',
    sampleSpacingM: 50,
    maximumSampleDistanceM: 50,
    minimumDistanceRatio: 0.8,
    minimumOrderedCoverage: 0.85,
  })
  assert.doesNotMatch(JSON.stringify(matched), /"(?:lat|lng|latlng|streams)"/)
})

test('uses complete-link clustering so a marginal route cannot bridge two groups', () => {
  const activities = [
    activity(1, 0, 5_000, 'west route'),
    activity(2, 1, 5_000, 'west route again'),
    activity(3, 2, 5_000, 'east route'),
    activity(4, 3, 5_000, 'east route again'),
  ]
  const matched = buildMatchedRuns(activities, {
    '1': route(0),
    '2': route(0.00001),
    '3': route(0, false, true),
    '4': route(0.00001, false, true),
  })

  assert.equal(matched.groups.length, 2)
  assert.deepEqual(
    matched.groups.map(group => group.efforts.map(effort => effort.id)),
    [
      [3, 4],
      [1, 2],
    ],
  )
})
