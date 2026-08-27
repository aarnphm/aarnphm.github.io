import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  parseWahooWorkoutFileUpload,
  safeWahooFileUrl,
  WahooCloudClient,
  wahooRateLimitDelay,
} from './wahoo-cloud'

function workout(id: number): object {
  return {
    id,
    starts: '2026-08-27T12:00:00.000Z',
    minutes: 60,
    name: `Ride ${id}`,
    workout_token: String(id),
    workout_type_id: 15,
    workout_summary: null,
    created_at: '2026-08-27T12:00:00.000Z',
    updated_at: '2026-08-27T13:00:00.000Z',
  }
}

function upload(status: string): object {
  return {
    id: 9,
    token: 'upload-token',
    status,
    time_zone: 'America/Toronto',
    workout_id: status === 'complete' ? 55 : null,
    workout_summary_id: status === 'complete' ? 66 : null,
    workout_file_id: status === 'complete' ? 77 : null,
    workout_name: 'Tempo',
    error: null,
    target_workout_id: null,
    created_at: '2026-08-27T12:00:00.000Z',
    updated_at: '2026-08-27T12:00:01.000Z',
  }
}

test('paginates workouts while caching a non-expiring access token', async () => {
  const directory = await fs.mkdtemp(join(tmpdir(), 'wahoo-cloud-'))
  const calls: string[] = []
  const request: typeof fetch = async (input, init) => {
    const url = input instanceof Request ? input.url : input.toString()
    calls.push(url)
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access-one',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    const headers = new Headers(init?.headers)
    assert.equal(headers.get('authorization'), 'Bearer access-one')
    const page = new URL(url).searchParams.get('page')
    return Response.json({
      workouts: [workout(page === '1' ? 1 : 2)],
      total: 2,
      page: Number(page),
      per_page: 1,
      order: 'descending',
      sort: 'starts',
    })
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    {
      apiBaseUrl: 'https://api.wahooligan.com',
      tokenUrl: 'https://api.wahooligan.com/oauth/token',
      envFile: join(directory, '.env'),
      request,
      now: () => 1_000,
    },
  )

  assert.deepEqual(
    (await client.listWorkouts(1)).map(value => value.id),
    [1, 2],
  )
  assert.equal(calls.filter(url => url.endsWith('/oauth/token')).length, 1)
  assert.match(
    await fs.readFile(join(directory, '.env'), 'utf8'),
    /WAHOO_REFRESH_TOKEN=refresh-two/,
  )
  await fs.rm(directory, { recursive: true })
})

test('encodes FIT upload fields and parses completed upload status', async () => {
  let uploadBody = ''
  const request: typeof fetch = async (input, init) => {
    const url = input instanceof Request ? input.url : input.toString()
    if (url.endsWith('/oauth/token'))
      return Response.json({
        access_token: 'access',
        refresh_token: 'refresh-two',
        expires_in: 3600,
      })
    if (url.endsWith('/v1/workout_file_uploads')) {
      uploadBody = init?.body instanceof URLSearchParams ? init.body.toString() : ''
      return Response.json(upload('pending'))
    }
    return Response.json(upload('complete'))
  }
  const client = new WahooCloudClient(
    { clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh-one' },
    { envFile: join(tmpdir(), `wahoo-cloud-${process.pid}.env`), request },
  )
  const created = await client.createWorkoutFileUpload({
    bytes: Uint8Array.from([1, 2, 3]),
    filename: 'ride.fit',
    timeZone: 'America/Toronto',
    workoutName: 'Tempo',
    targetWorkoutId: 42,
  })
  assert.equal(created.status, 'pending')
  const fields = new URLSearchParams(uploadBody)
  assert.equal(fields.get('workout_file_upload[file]'), 'data:application/vnd.fit;base64,AQID')
  assert.equal(fields.get('workout_file_upload[filename]'), 'ride.fit')
  assert.equal(fields.get('workout_file_upload[time_zone]'), 'America/Toronto')
  assert.equal(fields.get('workout_file_upload[workout_name]'), 'Tempo')
  assert.equal(fields.get('workout_file_upload[target_workout_id]'), '42')
  assert.equal(
    (await client.pollWorkoutFileUpload('upload-token', { intervalMs: 0 })).workoutId,
    55,
  )
})

test('rejects unsafe FIT URLs and unknown upload statuses', () => {
  assert.equal(
    safeWahooFileUrl('https://cdn.wahoofitness.com/ride.fit'),
    'https://cdn.wahoofitness.com/ride.fit',
  )
  assert.throws(() => safeWahooFileUrl('http://127.0.0.1/ride.fit'), /public HTTPS/)
  assert.throws(() => parseWahooWorkoutFileUpload(upload('finished')), /invalid status/)
})

test('uses provider rate-limit reset headers with a bounded fallback', () => {
  assert.equal(wahooRateLimitDelay(new Headers({ 'retry-after': '12' }), 1_000), 12_000)
  assert.equal(wahooRateLimitDelay(new Headers({ 'x-ratelimit-reset': '16' }), 1_000), 15_000)
  assert.equal(wahooRateLimitDelay(new Headers(), 1_000), 60_000)
})
