import assert from 'node:assert/strict'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { emptyWahooMetrics, type WahooActivity, type WahooCache } from '../plugins/stores/wahoo'
import { WahooApiError, WahooCloudClient, type WahooWorkoutDto } from '../util/wahoo-cloud'
import { fetchWahooCache, resolveWahooWorkoutSummary } from './sync-wahoo'

function workout(): WahooWorkoutDto {
  return {
    id: 55,
    starts: '2026-08-27T12:00:00.000Z',
    minutes: 60,
    name: 'Ride',
    workoutToken: '55',
    workoutTypeId: 15,
    summary: null,
    createdAt: '2026-08-27T12:00:00.000Z',
    updatedAt: '2026-08-27T13:00:00.000Z',
  }
}

function summary(): object {
  return {
    id: 66,
    name: 'Ride',
    ascent_accum: '400',
    cadence_avg: '90',
    calories_accum: '500',
    distance_accum: '48200',
    duration_active_accum: '7200',
    duration_paused_accum: '300',
    duration_total_accum: '7500',
    heart_rate_avg: '140',
    power_bike_np_last: '240',
    power_bike_tss_last: '120',
    power_avg: '210',
    speed_avg: '6.69',
    work_accum: '1500000',
    time_zone: 'America/Toronto',
    manual: false,
    edited: false,
    fitness_app_id: 1,
    file: { url: 'https://cdn.wahoofitness.com/ride.fit' },
    created_at: '2026-08-27T12:00:00.000Z',
    updated_at: '2026-08-27T13:00:00.000Z',
  }
}

function cachedActivity(): WahooActivity {
  return {
    id: 'wahoo:55',
    workoutId: 55,
    workoutTypeId: 15,
    workoutUpdatedAt: '2026-08-27T13:00:00.000Z',
    name: 'Ride',
    sport: 'bike',
    startDate: '2026-08-27T12:00:00.000Z',
    startDateLocal: '2026-08-27T08:00:00',
    distanceM: 48_200,
    movingTimeS: 7_200,
    elapsedTimeS: 7_500,
    sourceDevice: 'ELEMNT BOLT',
    sourceFile: {
      url: 'https://cdn.wahoofitness.com/ride.fit',
      sha256: 'a'.repeat(64),
      byteLength: 4_000,
      profileVersion: '21.208',
    },
    sweatLoss: { fluidMl: null, sodiumMg: null },
    metrics: emptyWahooMetrics(),
    summary: {
      id: 66,
      name: 'Ride',
      timeZone: 'America/Toronto',
      manual: false,
      edited: false,
      fitnessAppId: 1,
      durationPausedS: 300,
      createdAt: '2026-08-27T12:00:00.000Z',
      updatedAt: '2026-08-27T13:00:00.000Z',
    },
  }
}

function previousCache(): WahooCache {
  const activity = cachedActivity()
  const id = activity.id
  return {
    version: 4,
    lastSync: Date.parse('2026-08-27T14:00:00.000Z'),
    activities: { [id]: activity },
    streams: {
      [id]: {
        timestamps: [],
        time: [],
        latlng: [],
        altitude: [],
        distance: [],
        watts: [],
        rightBalance: [],
        heartrate: [],
        cadence: [],
        speed: [],
        temperature: [],
        respiration: [],
        muscleOxygenPercent: [],
        totalHemoglobinConcentration: [],
        heatStrainIndex: [],
        coreTemperatureC: [],
        skinTemperatureC: [],
        minuteVentilation: [],
        tidalVolume: [],
        fluidLossMl: [],
        sodiumLossMg: [],
      },
    },
    gearShifts: { [id]: [] },
    cyclingDynamics: {
      [id]: {
        time: [],
        distance: [],
        leftPedalSmoothness: [],
        rightPedalSmoothness: [],
        leftTorqueEffectiveness: [],
        rightTorqueEffectiveness: [],
        leftPowerPhaseStart: [],
        leftPowerPhaseEnd: [],
        rightPowerPhaseStart: [],
        rightPowerPhaseEnd: [],
        positionChanges: [],
        seatedTimeS: null,
        standingTimeS: null,
      },
    },
    summitSegments: {
      [id]: [
        {
          feature: 'summit-freeride',
          uuid: 'WAHOO_OFF_ROUTE_CLIMB-1',
          name: '1',
          startDate: '2026-08-27T12:30:00.000Z',
          endDate: '2026-08-27T12:35:00.000Z',
          distanceM: 1_500,
          durationS: 300,
          elevationGainM: 90,
          avgGradePct: 6,
          avgSpeedMps: 5,
          avgHeartRate: 155,
          avgPower: 280,
          avgCadence: 82,
        },
      ],
    },
  }
}

test('fetches the summary show endpoint when workout pages omit the embedded summary', async () => {
  const calls: string[] = []
  const request: typeof fetch = async input => {
    const url = input instanceof Request ? input.url : input.toString()
    calls.push(url)
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    return Response.json(summary())
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    { refreshTokenFile: join(tmpdir(), `wahoo-sync-${process.pid}.token`), request },
  )

  const resolved = await resolveWahooWorkoutSummary(client, workout())
  assert.equal(resolved.kind, 'available')
  if (resolved.kind !== 'available') assert.fail('expected an available Wahoo summary')
  assert.equal(resolved.summary.id, 66)
  assert.ok(calls.some(url => url.endsWith('/v1/workouts/55/workout_summary')))
})

test('treats absent third-party or planned summaries as an explicit skip', async () => {
  const request: typeof fetch = async input => {
    const url = input instanceof Request ? input.url : input.toString()
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    return new Response('missing', { status: 404 })
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    { refreshTokenFile: join(tmpdir(), `wahoo-sync-missing-${process.pid}.token`), request },
  )

  assert.deepEqual(await resolveWahooWorkoutSummary(client, workout()), { kind: 'missing' })
})

test('treats Wahoo empty deleted-workout summaries as an explicit skip', async () => {
  const request: typeof fetch = async input => {
    const url = input instanceof Request ? input.url : input.toString()
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    return Response.json({})
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    { refreshTokenFile: join(tmpdir(), `wahoo-sync-deleted-${process.pid}.token`), request },
  )

  assert.deepEqual(await resolveWahooWorkoutSummary(client, workout()), { kind: 'missing' })
})

test('rejects nonempty malformed Wahoo workout summaries', async () => {
  const request: typeof fetch = async input => {
    const url = input instanceof Request ? input.url : input.toString()
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    return Response.json({ name: 'Malformed summary' })
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    { refreshTokenFile: join(tmpdir(), `wahoo-sync-malformed-${process.pid}.token`), request },
  )

  await assert.rejects(
    resolveWahooWorkoutSummary(client, workout()),
    /Wahoo workout summary\.id must be a nonnegative integer/,
  )
})

test('treats Wahoo-restricted workout summaries as an explicit skip', async () => {
  const request: typeof fetch = async input => {
    const url = input instanceof Request ? input.url : input.toString()
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    return Response.json(
      { error: 'You are not authorized to view this workout summary' },
      { status: 401 },
    )
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    { refreshTokenFile: join(tmpdir(), `wahoo-sync-restricted-${process.pid}.token`), request },
  )

  assert.deepEqual(await resolveWahooWorkoutSummary(client, workout()), { kind: 'restricted' })
})

test('preserves unrelated Wahoo authorization failures', async () => {
  const request: typeof fetch = async input => {
    const url = input instanceof Request ? input.url : input.toString()
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    return Response.json({ error: 'Invalid access token' }, { status: 401 })
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    { refreshTokenFile: join(tmpdir(), `wahoo-sync-unauthorized-${process.pid}.token`), request },
  )

  await assert.rejects(
    resolveWahooWorkoutSummary(client, workout()),
    error => error instanceof WahooApiError && error.status === 401,
  )
})

test('reuses Summit segments for unchanged cached workouts', async () => {
  const previous = previousCache()
  const calls: string[] = []
  const request: typeof fetch = async input => {
    const url = input instanceof Request ? input.url : input.toString()
    calls.push(url)
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    if (url.includes('/v1/workouts?'))
      return Response.json({
        workouts: [
          {
            id: 55,
            starts: '2026-08-27T12:00:00.000Z',
            minutes: 60,
            name: 'Ride',
            workout_token: '55',
            workout_type_id: 15,
            workout_summary: null,
            created_at: '2026-08-27T12:00:00.000Z',
            updated_at: '2026-08-27T13:00:00.000Z',
          },
        ],
        total: 1,
        page: 1,
        per_page: 100,
      })
    return new Response('unexpected request', { status: 500 })
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    { refreshTokenFile: join(tmpdir(), `wahoo-sync-reuse-${process.pid}.token`), request },
  )

  const refreshed = await fetchWahooCache(client, previous)

  assert.strictEqual(refreshed.summitSegments['wahoo:55'], previous.summitSegments['wahoo:55'])
  assert.ok(!calls.some(url => url.includes('/workout_summary')))
  assert.ok(!calls.some(url => url.endsWith('.fit')))
})
