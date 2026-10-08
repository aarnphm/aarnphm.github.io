import assert from 'node:assert/strict'
import test from 'node:test'
import {
  emptyWahooMetrics,
  type WahooActivity,
  type WahooCache,
  type WahooStreams,
} from '../plugins/stores/wahoo'
import { estimateWahooCyclingStamina } from './cycling-stamina'

const start = '2026-08-29T12:00:00.000Z'

const activity = (values: Partial<WahooActivity> = {}): WahooActivity => ({
  id: 'wahoo:1',
  workoutId: 1,
  workoutTypeId: 15,
  name: 'Ride',
  sport: 'bike',
  startDate: start,
  startDateLocal: '2026-08-29T08:00:00',
  distanceM: 20_000,
  movingTimeS: 660,
  elapsedTimeS: 660,
  sourceDevice: 'ELEMNT BOLT',
  sourceFile: {
    url: 'https://cdn.wahoofitness.com/ride.fit',
    sha256: 'a'.repeat(64),
    byteLength: 4_000,
    profileVersion: '21.171',
  },
  sweatLoss: { fluidMl: null, sodiumMg: null },
  metrics: emptyWahooMetrics(),
  summary: {
    id: 1,
    name: 'Ride',
    timeZone: 'America/Toronto',
    manual: false,
    edited: false,
    fitnessAppId: 1,
    durationPausedS: 0,
    createdAt: start,
    updatedAt: start,
  },
  ...values,
})

const streams = (watts: (number | null)[], heartrate: (number | null)[]): WahooStreams => {
  const length = watts.length
  const empty = Array<number | null>(length).fill(null)
  return {
    timestamps: Array.from({ length }, (_, index) =>
      new Date(Date.parse(start) + index * 1000).toISOString(),
    ),
    time: Array.from({ length }, (_, index) => index),
    latlng: Array<[number, number] | null>(length).fill(null),
    altitude: empty,
    distance: Array.from({ length }, (_, index) => index * 30),
    watts,
    rightBalance: empty,
    heartrate,
    cadence: empty,
    speed: empty,
    temperature: empty,
    respiration: empty,
    muscleOxygenPercent: empty,
    totalHemoglobinConcentration: empty,
    heatStrainIndex: empty,
    coreTemperatureC: empty,
    skinTemperatureC: empty,
    minuteVentilation: empty,
    tidalVolume: empty,
    fluidLossMl: empty,
    sodiumLossMg: empty,
  }
}

const cache = (stream: WahooStreams, values: Partial<WahooActivity> = {}): WahooCache => {
  const ride = activity(values)
  return {
    version: 4,
    lastSync: Date.parse('2026-08-30T00:00:00Z'),
    activities: { [ride.id]: ride },
    streams: { [ride.id]: stream },
    gearShifts: { [ride.id]: [] },
    cyclingDynamics: {},
    summitSegments: { [ride.id]: [] },
  }
}

test('estimates slower potential loss and a recoverable current stamina deficit', () => {
  const watts = Array.from({ length: 661 }, (_, index) => (index <= 60 ? 400 : 100))
  const heartrate = Array.from({ length: 661 }, (_, index) => (index <= 60 ? 180 : 125))
  const estimate = estimateWahooCyclingStamina(cache(streams(watts, heartrate)), 250, 200).get(
    'wahoo:1',
  )

  assert.ok(estimate)
  assert.equal(estimate.method, 'garden-stamina-v2')
  assert.equal(estimate.samples[0].stamina, 100)
  assert.ok(estimate.samples[60].stamina < estimate.samples[60].potentialStamina)
  const highIntensityDeficit = estimate.samples[60].potentialStamina - estimate.samples[60].stamina
  const ending = estimate.samples.at(-1)
  assert.ok(ending)
  assert.ok(ending.potentialStamina <= estimate.samples[60].potentialStamina)
  assert.ok(ending.potentialStamina - ending.stamina < highIntensityDeficit)
  assert.ok(
    estimate.samples.every(
      sample =>
        sample.stamina >= 0 &&
        sample.stamina <= sample.potentialStamina &&
        sample.potentialStamina <= 100,
    ),
  )
})

test('fails closed when aligned power and heart rate cover less than eighty percent', () => {
  const watts = Array<number | null>(100).fill(null)
  const heartrate = Array<number | null>(100).fill(null)
  for (let index = 0; index < 79; index++) {
    watts[index] = 200
    heartrate[index] = 150
  }
  const estimates = estimateWahooCyclingStamina(cache(streams(watts, heartrate)), 250, 200)

  assert.equal(estimates.size, 0)
})
