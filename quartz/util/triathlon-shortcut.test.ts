import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildStravaActivityIndex,
  isStravaActivityIndex,
  STRAVA_ACTIVITY_INDEX_KIND,
  stravaActivityIdFromShortcutPath,
  triathlonActivityShortcutRedirectUrl,
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
      'https://t.aarnphm.xyz/activities/19745591953',
      activityDates,
    ),
    'https://t.aarnphm.xyz/on/2026/08/14',
  )
  assert.equal(
    triathlonActivityShortcutRedirectUrl(
      'https://t.aarnphm.xyz/activities/19745591953?utm_source=strava#effort',
      activityDates,
    ),
    'https://t.aarnphm.xyz/on/2026/08/14?utm_source=strava#effort',
  )
  assert.equal(
    triathlonActivityShortcutRedirectUrl(
      'https://t.aarnphm.xyz/activities/19745591954',
      activityDates,
    ),
    null,
  )
})
