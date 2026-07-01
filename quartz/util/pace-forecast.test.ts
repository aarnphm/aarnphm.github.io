import assert from 'node:assert/strict'
import { test } from 'node:test'
import { type PaceContext, type PaceDayState, type PaceLegSpec } from './pace-features'
import { PaceForecaster, type WorkerLike } from './pace-forecast'

class MockWorker implements WorkerLike {
  onmessage: ((event: MessageEvent) => void) | null = null
  loaded = true

  postMessage(message: unknown): void {
    const msg = message as { type: string; id?: number; raw?: number[] }
    if (msg.type === 'load') {
      this.reply({ type: 'loaded', ok: this.loaded })
    } else if (msg.type === 'predict') {
      const raw = msg.raw ?? []
      const mu = raw[0] ? 1.3 : raw[1] ? 8 : 3.3
      this.reply({ type: 'prediction', id: msg.id, ok: true, mu, sigma: 0.1 })
    }
  }

  private reply(data: unknown): void {
    this.onmessage?.({ data } as MessageEvent)
  }
}

const day: PaceDayState = {
  ctl: 50,
  atl: 40,
  tsb: 10,
  swimCtl: 5,
  bikeCtl: 20,
  runCtl: 25,
  hrv: 60,
  rhr: 48,
  readiness: 80,
  sleepDurationS: 28800,
  tempDeviationC: -0.25,
  weightKg: 88,
}

const ctx: PaceContext = { vThrBySport: { swim: 1.2, bike: 8, run: 3.3 }, hrMax: 182 }

const olympic: PaceLegSpec[] = [
  { sport: 'swim', distanceKm: 1.5, elevationM: 0, tempC: null, windKph: null },
  { sport: 'bike', distanceKm: 40, elevationM: 0, tempC: null, windKph: null },
  { sport: 'run', distanceKm: 10, elevationM: 0, tempC: null, windKph: null },
]

function primed(): PaceForecaster {
  const f = new PaceForecaster(new MockWorker())
  f.ready = true
  f.day = day
  f.ctx = ctx
  return f
}

test('predict returns null when not ready', async () => {
  const f = new PaceForecaster(new MockWorker())
  assert.equal(await f.predict(new Float32Array(19), new Float32Array(19)), null)
})

test('forecastFinish composes per-leg ranges + transitions', async () => {
  const fin = await primed().forecastFinish(olympic, 300)
  assert.ok(fin)
  assert.equal(fin.legs.length, 3)
  const expectedMid = 1500 / 1.3 + 40000 / 8 + 10000 / 3.3 + 300
  assert.ok(Math.abs(fin.midSec - expectedMid) < 1, `mid=${fin.midSec}`)
  assert.ok(fin.fastSec < fin.midSec && fin.midSec < fin.slowSec)
  assert.equal(fin.legs[0].sport, 'swim')
  assert.ok(Math.abs(fin.legs[1].mu - 8) < 1e-9)
  const totalHalf = fin.slowSec - fin.midSec
  const legHalfSum = fin.legs.reduce((s, l) => s + (l.slowSec - l.midSec), 0)
  assert.ok(totalHalf < legHalfSum, 'finish band combines leg variances in quadrature')
})

test('forecastFinish bails to null on a bad leg prediction', async () => {
  const f = primed()
  f.ready = false
  assert.equal(await f.forecastFinish(olympic, 300), null)
})

test('forecastLeg null when state missing', async () => {
  const f = new PaceForecaster(new MockWorker())
  f.ready = true
  assert.equal(await f.forecastLeg(olympic[0]), null)
})

test('concurrent predictions route to the right ids', async () => {
  const f = primed()
  const [swim, bike, run] = await Promise.all(olympic.map(leg => f.forecastLeg(leg)))
  assert.ok(Math.abs((swim?.mu ?? 0) - 1.3) < 1e-9)
  assert.ok(Math.abs((bike?.mu ?? 0) - 8) < 1e-9)
  assert.ok(Math.abs((run?.mu ?? 0) - 3.3) < 1e-9)
})
