import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import test from 'node:test'
import { STRAVA_DETAIL_INDEX_KIND } from '../../../util/strava-detail'
import { readDetailPayload } from './data'

test('loads and reconstructs a Strava detail index from valid JSON shards', async () => {
  const requested: string[] = []
  const server = createServer((request, response) => {
    const path = request.url ?? ''
    requested.push(path)
    const values: Record<string, unknown> = {
      '/static/strava-detail.json': {
        kind: STRAVA_DETAIL_INDEX_KIND,
        shards: ['strava-detail/2026-08.json', 'strava-detail/2026-07.json'],
        health: {},
        ftp: 250,
      },
      '/static/strava-detail/2026-08.json': {
        details: {
          '2': { id: 2, date: '2026-08-02', sport: 'bike' },
          '3': { id: 3, date: '2026-08-03', sport: 'walk' },
          '4': { id: 4, date: '2026-08-04', sport: 'yoga' },
          '5': { id: 5, date: '2026-08-05', sport: 'treatment' },
        },
      },
      '/static/strava-detail/2026-07.json': {
        details: { '1': { id: 1, date: '2026-07-31', sport: 'run' } },
      },
    }
    const value = values[path]
    if (!value) {
      response.writeHead(404)
      response.end()
      return
    }
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify(value))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  try {
    const controller = new AbortController()
    const response = await fetch(`http://127.0.0.1:${address.port}/static/strava-detail.json`)
    const payload = await readDetailPayload(response, controller.signal)
    assert.deepEqual(Object.keys(payload.details).sort(), ['1', '2', '3', '4', '5'])
    assert.equal(payload.details['1'].date, '2026-07-31')
    assert.equal(payload.details['2'].date, '2026-08-02')
    assert.equal(payload.details['3'].sport, 'walk')
    assert.equal(payload.details['4'].sport, 'yoga')
    assert.equal(payload.details['5'].sport, 'treatment')
    assert.equal(payload.ftp, 250)
    assert.deepEqual(requested.sort(), [
      '/static/strava-detail.json',
      '/static/strava-detail/2026-07.json',
      '/static/strava-detail/2026-08.json',
    ])
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close(error => (error ? reject(error) : resolve())),
    )
  }
})
