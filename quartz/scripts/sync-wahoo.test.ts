import assert from 'node:assert/strict'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { WahooCloudClient, type WahooWorkoutDto } from '../util/wahoo-cloud'
import { resolveWahooWorkoutSummary } from './sync-wahoo'

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
  assert.equal(resolved?.id, 66)
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

  assert.equal(await resolveWahooWorkoutSummary(client, workout()), null)
})
