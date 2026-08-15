import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildStravaActivityIndex,
  isStravaActivityIndex,
  STRAVA_ACTIVITY_INDEX_KIND,
  stravaActivityIdFromShortcutPath,
  triathlonActivityShortcutRedirectUrl,
  triathlonShortcutRedirectUrl,
} from './triathlon-shortcut'

test('builds and validates the generated Strava activity date index', () => {
  const index = buildStravaActivityIndex({
    '19745591953': { id: 19745591953, date: '2026-08-14' },
    '19745591954': { id: 19745591954, date: '2026-08-15' },
  })

  assert.deepEqual(index, {
    kind: STRAVA_ACTIVITY_INDEX_KIND,
    activities: { '19745591953': '2026-08-14', '19745591954': '2026-08-15' },
  })
  assert.equal(isStravaActivityIndex(index), true)
  assert.equal(
    isStravaActivityIndex({
      kind: STRAVA_ACTIVITY_INDEX_KIND,
      activities: { '19745591953': '2026-02-29' },
    }),
    false,
  )
  assert.throws(
    () => buildStravaActivityIndex({ wrong: { id: 19745591953, date: '2026-08-14' } }),
    /invalid ID/,
  )
})

test('redirects Strava activity shortcut paths to their canonical activity day', () => {
  const activityDates = { '19745591953': '2026-08-14' }

  assert.equal(stravaActivityIdFromShortcutPath('/activities/19745591953'), '19745591953')
  assert.equal(stravaActivityIdFromShortcutPath('/activities/19745591953/'), '19745591953')
  assert.equal(stravaActivityIdFromShortcutPath('/activities/unknown'), null)
  assert.equal(
    triathlonActivityShortcutRedirectUrl(
      'https://aarnphm.xyz',
      'https://t.aarnphm.xyz/activities/19745591953',
      activityDates,
    ),
    'https://aarnphm.xyz/triathlon/on/2026/08/14',
  )
  assert.equal(
    triathlonActivityShortcutRedirectUrl(
      'https://t.aarnphm.xyz',
      'https://t.aarnphm.xyz/activities/19745591953?utm_source=strava#effort',
      activityDates,
    ),
    'https://aarnphm.xyz/triathlon/on/2026/08/14?utm_source=strava#effort',
  )
  assert.equal(
    triathlonActivityShortcutRedirectUrl(
      'https://aarnphm.xyz',
      'https://t.aarnphm.xyz/activities/19745591954',
      activityDates,
    ),
    null,
  )
})

test('redirects triathlon shortcut documents to canonical triathlon paths', () => {
  const cases: [string, string][] = [
    ['https://t.aarnphm.xyz', 'https://aarnphm.xyz/triathlon'],
    ['https://t.aarnphm.xyz/', 'https://aarnphm.xyz/triathlon'],
    ['https://t.aarnphm.xyz/analytics', 'https://aarnphm.xyz/triathlon/analytics'],
    ['https://t.aarnphm.xyz/tools', 'https://aarnphm.xyz/triathlon/tools'],
    ['https://t.aarnphm.xyz/maps', 'https://aarnphm.xyz/triathlon/maps'],
    ['https://t.aarnphm.xyz/training', 'https://aarnphm.xyz/triathlon/training'],
    ['https://t.aarnphm.xyz/on', 'https://aarnphm.xyz/triathlon/on'],
    ['https://t.aarnphm.xyz/2026', 'https://aarnphm.xyz/triathlon/on/2026'],
    ['https://t.aarnphm.xyz/2026/07', 'https://aarnphm.xyz/triathlon/on/2026/07'],
    ['https://t.aarnphm.xyz/2026/07/09', 'https://aarnphm.xyz/triathlon/on/2026/07/09'],
    ['https://t.aarnphm.xyz/triathlon', 'https://aarnphm.xyz/triathlon'],
    ['https://t.aarnphm.xyz/triathlon/tools', 'https://aarnphm.xyz/triathlon/tools'],
  ]

  for (const [source, expected] of cases) {
    assert.equal(triathlonShortcutRedirectUrl('https://t.aarnphm.xyz', source, true), expected)
  }
})

test('preserves triathlon shortcut search and hash state', () => {
  assert.equal(
    triathlonShortcutRedirectUrl(
      'https://t.aarnphm.xyz',
      'https://t.aarnphm.xyz/analytics?window=42d#fitness',
      true,
    ),
    'https://aarnphm.xyz/triathlon/analytics?window=42d#fitness',
  )
  assert.equal(
    triathlonShortcutRedirectUrl(
      'https://t.aarnphm.xyz',
      'https://t.aarnphm.xyz/2026/07?sort=distance#bike',
      true,
    ),
    'https://aarnphm.xyz/triathlon/on/2026/07?sort=distance#bike',
  )
})

test('redirects triathlon shortcut assets without inventing nested triathlon paths', () => {
  assert.equal(
    triathlonShortcutRedirectUrl(
      'https://t.aarnphm.xyz',
      'https://t.aarnphm.xyz/static/analytics.json',
      false,
    ),
    'https://aarnphm.xyz/static/analytics.json',
  )
})

test('honors configured canonical base URLs', () => {
  assert.equal(
    triathlonShortcutRedirectUrl(
      'https://preview.aarnphm.xyz',
      'https://t.aarnphm.xyz/tools',
      true,
    ),
    'https://preview.aarnphm.xyz/triathlon/tools',
  )
})
