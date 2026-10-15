import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import test from 'node:test'
import { date, health, raw } from '../util/fixtures/garmin-health'
import { fetchGarminHealthRange } from './garmin-health-sync'

test('health download distinguishes no content, native scores, and failed score requests', async t => {
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? '', 'http://localhost')
    const score = url.pathname.endsWith('endurancescore') || url.pathname.endsWith('hillscore')
    const mode = url.pathname.split('/')[1]
    if (score && mode === 'absent') {
      response.writeHead(204).end()
      return
    }
    if (score && mode === 'empty') {
      response.end()
      return
    }
    if (score && mode === 'html') {
      response.setHeader('Content-Type', 'text/html')
      response.end('<html>Sign in</html>')
      return
    }
    response.setHeader('Content-Type', 'application/json')
    if (score && mode === 'failure') {
      response.writeHead(503).end(JSON.stringify({ message: 'unavailable' }))
      return
    }
    if (score && mode === 'unauthorized') {
      response.writeHead(401).end(JSON.stringify({ message: 'unauthorized' }))
      return
    }
    const value = url.pathname.endsWith('/daily')
      ? raw.bodyBattery
      : url.pathname.includes('/trainingreadiness/')
        ? raw.readiness
        : url.pathname.includes('/trainingstatus/')
          ? raw.trainingStatus
          : url.pathname.endsWith('endurancescore')
            ? raw.endurance
            : raw.hill
    response.end(mode === 'malformed' && score ? '{' : JSON.stringify(value))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  t.after(
    () =>
      new Promise<void>((resolve, reject) =>
        server.close(error => (error ? reject(error) : resolve())),
      ),
  )
  const address = server.address()
  assert.ok(address != null && typeof address !== 'string')

  for (const mode of [
    'absent',
    'available',
    'empty',
    'html',
    'failure',
    'unauthorized',
    'malformed',
  ]) {
    await t.test(mode, async () => {
      const result = await fetchGarminHealthRange(
        { headers: { authorization: 'Bearer fixture' } },
        `http://127.0.0.1:${address.port}/${mode}`,
        { [date]: health },
        date,
        date,
        0,
      )
      const day = result.health[date]
      const failed = mode !== 'absent' && mode !== 'available'
      assert.equal(result.responses, failed ? 3 : 5)
      assert.equal(result.failures, failed ? 2 : 0)
      assert.deepEqual(day.bodyBattery.value, health.bodyBattery.value)
      assert.equal(day.bodyBattery.status, 'available')
      for (const key of ['enduranceScore', 'hillScore'] as const) {
        assert.equal(
          day[key].status,
          failed ? 'error' : mode === 'absent' ? 'unavailable' : 'available',
        )
        assert.deepEqual(day[key].value, mode === 'absent' ? null : health[key].value)
        assert.equal(
          failed ? day[key].fetchedAt === health[key].fetchedAt : day[key].fetchedAt != null,
          true,
        )
      }
      assert.equal(health.enduranceScore.status, 'available')
      assert.equal(health.hillScore.status, 'available')
    })
  }
})
