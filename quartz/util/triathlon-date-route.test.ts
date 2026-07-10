import assert from 'node:assert/strict'
import test from 'node:test'
import {
  triathlonActivityDates,
  triathlonActivityFeedRoutes,
  triathlonDateFromSlug,
  triathlonDateRouteFromSlug,
  triathlonDateTree,
  triathlonDaySlug,
  triathlonFeedScopeFromSlug,
  triathlonOnSlugFromShortcutPath,
} from './triathlon-date-route'

test('converts stored local activity dates to nested triathlon slugs', () => {
  assert.equal(triathlonDaySlug('2026-07-09'), 'triathlon/on/2026/07/09')
  assert.equal(triathlonDateFromSlug('triathlon/on/2026/07/09'), '2026-07-09')
})

test('rejects malformed and impossible triathlon dates', () => {
  for (const date of ['2026-7-09', '2026-02-29', '2026-13-01', '2026-04-31']) {
    assert.equal(triathlonDaySlug(date), null)
  }
  assert.equal(triathlonDaySlug('2028-02-29'), 'triathlon/on/2028/02/29')
  assert.equal(triathlonDateFromSlug('triathlon/on/2026/02/29'), null)
  assert.equal(triathlonDateFromSlug('triathlon/feed'), null)
})

test('parses temporal feed scopes from canonical slugs', () => {
  assert.deepEqual(triathlonDateRouteFromSlug('triathlon/on'), { kind: 'index' })
  assert.deepEqual(triathlonFeedScopeFromSlug('triathlon/on/2026'), {
    prefix: '2026',
    title: '2026',
  })
  assert.deepEqual(triathlonFeedScopeFromSlug('triathlon/on/2026/07'), {
    prefix: '2026-07',
    title: '2026 / 07',
  })
  assert.equal(triathlonFeedScopeFromSlug('triathlon/on/2026/07/09'), null)
})

test('maps shortcut date paths into canonical on routes', () => {
  assert.equal(triathlonOnSlugFromShortcutPath('/2026'), 'triathlon/on/2026')
  assert.equal(triathlonOnSlugFromShortcutPath('/2026/07'), 'triathlon/on/2026/07')
  assert.equal(triathlonOnSlugFromShortcutPath('/2026/07/09'), 'triathlon/on/2026/07/09')
  assert.equal(triathlonOnSlugFromShortcutPath('/2026/13'), null)
})

test('returns one sorted route date for each distinct activity day', () => {
  assert.deepEqual(
    triathlonActivityDates({
      bike: { date: '2026-07-09' },
      swim: { date: '2026-07-09' },
      run: { date: '2026-07-08' },
      malformed: { date: '2026-99-99' },
    }),
    ['2026-07-08', '2026-07-09'],
  )
})

test('groups details into a newest-first year/month/day tree with rollups', () => {
  const tree = triathlonDateTree({
    a: { date: '2026-07-09', sport: 'bike', distanceKm: 40, movingTimeS: 5400 },
    b: { date: '2026-07-09', sport: 'swim', distanceKm: 1, movingTimeS: 1800 },
    c: { date: '2026-06-30', sport: 'run', distanceKm: 10, movingTimeS: 3000 },
    d: { date: '2025-12-31', sport: 'run', distanceKm: 5, movingTimeS: 1500 },
    bad: { date: '2026-99-99', sport: 'run', distanceKm: 1, movingTimeS: 60 },
  })
  assert.deepEqual(
    tree.map(y => y.year),
    ['2026', '2025'],
  )
  assert.deepEqual(
    tree[0].months.map(m => m.month),
    ['07', '06'],
  )
  assert.equal(tree[0].count, 3)
  assert.equal(tree[0].km, 51)
  assert.equal(tree[0].slug, 'triathlon/on/2026')
  const july = tree[0].months[0]
  assert.equal(july.slug, 'triathlon/on/2026/07')
  assert.deepEqual(july.days, [
    {
      date: '2026-07-09',
      day: '09',
      slug: 'triathlon/on/2026/07/09',
      count: 2,
      sports: ['swim', 'bike'],
      km: 41,
      timeS: 7200,
    },
  ])
})

test('scopes the tree by year and month prefixes', () => {
  const details: Parameters<typeof triathlonDateTree>[0] = {
    a: { date: '2026-07-09', sport: 'bike', distanceKm: 40, movingTimeS: 5400 },
    b: { date: '2026-06-30', sport: 'run', distanceKm: 10, movingTimeS: 3000 },
    c: { date: '2025-12-31', sport: 'run', distanceKm: 5, movingTimeS: 1500 },
  }
  const year = triathlonDateTree(details, '2026')
  assert.deepEqual(
    year.map(y => y.year),
    ['2026'],
  )
  assert.equal(year[0].months.length, 2)
  const month = triathlonDateTree(details, '2026-06')
  assert.deepEqual(
    month[0].months.map(m => m.month),
    ['06'],
  )
  assert.deepEqual(triathlonDateTree(details, '2024'), [])
})

test('builds one year and month feed route for each represented period', () => {
  assert.deepEqual(
    triathlonActivityFeedRoutes({
      june: { date: '2026-06-30' },
      julyBike: { date: '2026-07-09' },
      julySwim: { date: '2026-07-09' },
      prior: { date: '2025-12-31' },
    }),
    [
      { slug: 'triathlon/on/2025', title: 'triathlon · 2025' },
      { slug: 'triathlon/on/2025/12', title: 'triathlon · 2025 / 12' },
      { slug: 'triathlon/on/2026', title: 'triathlon · 2026' },
      { slug: 'triathlon/on/2026/06', title: 'triathlon · 2026 / 06' },
      { slug: 'triathlon/on/2026/07', title: 'triathlon · 2026 / 07' },
    ],
  )
})
