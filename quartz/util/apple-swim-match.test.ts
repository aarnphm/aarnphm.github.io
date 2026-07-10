import assert from 'node:assert/strict'
import test from 'node:test'
import type { AppleSwim } from '../plugins/stores/apple'
import { matchAppleSwims, type SwimActivityCandidate } from './apple-swim-match'

const swim = (values: Partial<AppleSwim> = {}): AppleSwim => ({
  id: 'swim-1',
  date: '2026-07-09',
  start: '2026-07-09T14:00:00Z',
  end: '2026-07-09T14:30:00Z',
  activeTimeS: 1_500,
  totalM: 1_000,
  laps: 40,
  strokes: { freestyle: 1_000 },
  strokeCount: 700,
  strokeTimeS: 1_500,
  ...values,
})

const activity = (id: number, start: string, distanceM: number): SwimActivityCandidate => ({
  id,
  date: start.slice(0, 10),
  start,
  distanceM,
})

test('matches two same-day swims by start time and distance', () => {
  const morning = swim({
    id: 'morning',
    start: '2026-07-09T10:00:00Z',
    end: '2026-07-09T10:20:00Z',
    totalM: 500,
  })
  const evening = swim({ id: 'evening', start: '2026-07-09T18:00:00Z', totalM: 1_000 })
  const matches = matchAppleSwims(
    [evening, morning],
    [activity(1, '2026-07-09T10:01:00Z', 500), activity(2, '2026-07-09T18:02:00Z', 1_000)],
  )

  assert.equal(matches.get(1)?.id, 'morning')
  assert.equal(matches.get(2)?.id, 'evening')
})

test('uses distance for a legacy date-only swim and leaves implausible sessions unmatched', () => {
  const legacy = swim({ id: null, start: null, end: null, totalM: 700 })
  const impossible = swim({
    id: 'far-away',
    date: '2026-07-10',
    start: '2026-07-10T01:00:00Z',
    end: '2026-07-10T01:30:00Z',
    totalM: 4_000,
  })
  const matches = matchAppleSwims(
    [legacy, impossible],
    [activity(1, '2026-07-10T12:00:00Z', 500), activity(2, '2026-07-09T18:00:00Z', 700)],
  )

  assert.equal(matches.get(2)?.totalM, 700)
  assert.equal(matches.has(1), false)
})

test('prefers session rows when a cache still contains a legacy row for the same day', () => {
  const legacy = swim({ id: null, start: null, end: null, totalM: 1_500 })
  const session = swim({ id: 'session', start: '2026-07-09T20:13:30Z', totalM: 1_000 })
  const matches = matchAppleSwims([legacy, session], [activity(1, '2026-07-09T20:13:31Z', 1_000)])

  assert.equal(matches.get(1)?.id, 'session')
})
