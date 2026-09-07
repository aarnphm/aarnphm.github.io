import assert from 'node:assert/strict'
import test from 'node:test'
import type { ActivitySummary } from '../../../plugins/stores/analytics'
import { activityQueryTokens, matchesActivityQuery, parseActivityQuery } from './search'

const query = (text: string) => parseActivityQuery(activityQueryTokens(text))

const activities: Pick<
  ActivitySummary,
  'id' | 'sport' | 'name' | 'date' | 'virtual' | 'treadmill'
>[] = [
  { id: 1, sport: 'bike', name: 'Alpe', date: '2026-09-04', virtual: true },
  { id: 2, sport: 'run', name: 'Easy', date: '2026-09-05', virtual: true },
  { id: 3, sport: 'run', name: 'Warm down', date: '2026-09-06', treadmill: true },
  { id: 4, sport: 'run', name: 'Easy', date: '2026-09-05' },
  { id: 5, sport: 'bike', name: 'Morning', date: '2026-09-06' },
  { id: 6, sport: 'swim', name: 'Pool', date: '2026-09-06', virtual: true },
  { id: 7, sport: 'walk', name: 'Treadmill walk', date: '2026-09-06', treadmill: true },
]

const matchingIds = (text: string) =>
  activities.filter(activity => matchesActivityQuery(activity, query(text))).map(a => a.id)

test('virtual filters include virtual runs and rides, excluding outdoor and other sports', () => {
  assert.deepEqual(matchingIds('filter:virtual'), [1, 2])
  assert.deepEqual(matchingIds('filter: virtual'), [1, 2])
  assert.deepEqual(matchingIds(' FILTER: VIRTUAL '), [1, 2])
})

test('treadmill filters include treadmill and virtual runs', () => {
  assert.deepEqual(matchingIds('filter:treadmill'), [2, 3])
  assert.deepEqual(matchingIds('filter: treadmill'), [2, 3])
})

test('environment filters compose with sport, text, sorting, and inclusive dates', () => {
  assert.deepEqual(matchingIds('filter: virtual filter: run'), [2])
  assert.deepEqual(matchingIds('filter: bike filter:virtual'), [1])
  assert.deepEqual(matchingIds('filter: virtual easy sort: distance'), [2])
  assert.deepEqual(matchingIds('filter: swim filter: virtual'), [])
  const parsed = query('filter: treadmill sort:distance')
  parsed.filterDate = { start: '2026-09-06', end: '2026-09-06' }
  assert.deepEqual(
    activities.filter(a => matchesActivityQuery(a, parsed)).map(a => a.id),
    [3],
  )
  assert.equal(parsed.sortKey, 'distance')
})

test('spaced filter commands preserve existing aliases, date units, and unknown filters', () => {
  assert.equal(query('filter: gym').filterSport, 'strength')
  assert.deepEqual(query('filter: 3 days').filterDate, query('filter:3 days').filterDate)
  assert.deepEqual(
    query('filter: virtual filter: week').filterDate,
    query('filter:week').filterDate,
  )
  assert.equal(parseActivityQuery(['filter:', 'virtual']).filterEnvironment, 'virtual')
  assert.deepEqual(matchingIds('filter:unknown'), [])
  assert.deepEqual(
    matchingIds(''),
    activities.map(a => a.id),
  )
})

test('partial commands normalize into the tokens used for suggestions and insertion', () => {
  assert.deepEqual(activityQueryTokens('filter: vir'), ['filter:vir'])
  assert.deepEqual(activityQueryTokens('filter:run filter: tread'), ['filter:run', 'filter:tread'])
  assert.deepEqual(activityQueryTokens('filter: '), ['filter:'])
  assert.deepEqual(activityQueryTokens('  '), [])
})
