import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { lunarPhase, lunarDiscPath, readLunarEvents } from './lunar-phase'

// Published USNO quarter instants, not a simulated API or implementation-derived dates.
const published = {
  phasedata: [
    { year: 2026, month: 10, day: 3, time: '13:25', phase: 'Last Quarter' },
    { year: 2026, month: 10, day: 10, time: '15:50', phase: 'New Moon' },
    { year: 2026, month: 10, day: 18, time: '16:12', phase: 'First Quarter' },
    { year: 2026, month: 10, day: 26, time: '04:12', phase: 'Full Moon' },
    { year: 2026, month: 11, day: 1, time: '20:28', phase: 'Last Quarter' },
    { year: 2026, month: 11, day: 9, time: '07:02', phase: 'New Moon' },
  ],
}

test('published USNO quarter instants anchor new, both halves, and full moon', () => {
  const events = readLunarEvents(published)
  assert.ok(events)
  for (const [instant, phase, name, illumination] of [
    ['2026-10-10T15:50:00Z', 0, 'new moon', 0],
    ['2026-10-18T16:12:00Z', 0.25, 'first quarter', 0.5],
    ['2026-10-26T04:12:00Z', 0.5, 'full moon', 1],
    ['2026-11-01T20:28:00Z', 0.75, 'last quarter', 0.5],
  ] satisfies [string, number, string, number][]) {
    const moon = lunarPhase(Date.parse(instant), events)
    assert.equal(moon.source, 'usno')
    assert.equal(moon.phase, phase)
    assert.equal(moon.name, name)
    assert.ok(Math.abs(moon.illumination - illumination) < 1e-12)
  }
})

test('cycle stays continuous across new moon and uses variable quarter durations', () => {
  const events = readLunarEvents(published)
  assert.ok(events)
  const before = lunarPhase(Date.parse('2026-10-10T15:49:00Z'), events)
  const after = lunarPhase(Date.parse('2026-10-10T15:51:00Z'), events)
  assert.ok(before.phase > 0.99 && after.phase < 0.01)
  assert.equal(before.waxing, false)
  assert.equal(after.waxing, true)
  const current = lunarPhase(Date.parse('2026-10-11T16:00:00Z'), events)
  assert.equal(current.name, 'waxing crescent')
  assert.ok(current.illumination > 0 && current.illumination < 0.03)
})

test('UTC phase events project onto the actual local calendar date', () => {
  const events = readLunarEvents(published)
  assert.ok(events)
  const full = events.find(event => event.name === 'full moon')
  assert.ok(full)
  assert.equal(
    new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Los_Angeles' }).format(full.at),
    '2026-10-25',
  )
  assert.equal(
    new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Toronto' }).format(full.at),
    '2026-10-26',
  )
})

test('reject malformed, impossible, unordered, and interrupted ephemerides', () => {
  for (const changes of [
    { year: 2026, month: 2, day: 30 },
    { time: '25:00' },
    { time: '12:99' },
    { time: '12:00junk' },
    { phase: 'Blood Moon' },
    { year: Number.NaN },
  ]) {
    const data = structuredClone(published)
    Object.assign(data.phasedata[1], changes)
    assert.equal(readLunarEvents(data), null)
  }
  assert.equal(readLunarEvents({ error: 'unavailable', ...published }), null)
  assert.equal(readLunarEvents({ phasedata: published.phasedata.toReversed() }), null)
  assert.equal(readLunarEvents({ phasedata: published.phasedata.slice(0, 1) }), null)
  assert.equal(
    readLunarEvents({ phasedata: [published.phasedata[0], published.phasedata[2]] }),
    null,
  )
  assert.equal(
    readLunarEvents({ phasedata: [published.phasedata[0], published.phasedata[0]] }),
    null,
  )
})

test('out-of-range or unavailable events are visibly an estimate', () => {
  const events = readLunarEvents(published)
  assert.ok(events)
  for (const at of [Date.parse('1960-01-01Z'), Date.parse('2027-01-01Z')]) {
    const moon = lunarPhase(at, events)
    assert.equal(moon.source, 'estimate')
    assert.ok(moon.phase >= 0 && moon.phase < 1)
    assert.ok(moon.illumination >= 0 && moon.illumination <= 1)
  }
})

test('projected lunar disc illuminates the correct side and proportion', () => {
  const area = (path: string) => {
    let result = 0
    for (const match of path.matchAll(/M(-?[\d.]+) (-?[\d.]+)h([\d.]+)v([\d.]+)h-([\d.]+)Z/g))
      result += Number(match[3]) * Number(match[4])
    return result
  }
  const full = lunarDiscPath(0.5, 10, 2)
  const right = lunarDiscPath(0.25, 10, 2)
  const left = lunarDiscPath(0.75, 10, 2)
  assert.equal(lunarDiscPath(0, 10, 2), '')
  assert.ok(area(full) > 250 && area(full) < 360)
  assert.ok(Math.abs(area(right) / area(full) - 0.5) < 0.02)
  assert.ok(Math.abs(area(left) / area(full) - 0.5) < 0.02)
  for (const match of right.matchAll(/M(-?[\d.]+) /g)) assert.ok(Number(match[1]) >= 0)
  for (const match of left.matchAll(/M(-?[\d.]+) -?[\d.]+h([\d.]+)/g))
    assert.ok(Number(match[1]) + Number(match[2]) <= 0)
})

test(
  'captured live USNO response validates through the same boundary',
  { skip: !process.env.LUNAR_EPHEMERIS_FIXTURE },
  () => {
    const fixture = process.env.LUNAR_EPHEMERIS_FIXTURE
    assert.ok(fixture)
    const response: unknown = JSON.parse(readFileSync(fixture, 'utf8'))
    const events = readLunarEvents(response)
    assert.ok(events && events.length > 45)
    assert.equal(lunarPhase(Date.parse('2026-10-26T04:12:00Z'), events).phase, 0.5)
  },
)
