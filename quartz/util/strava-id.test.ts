import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { workoutAnalysisOwner } from '../components/triathlon/activity/planned-tab'
import { parseTrackingBlock, manualSaunaActivityId } from '../plugins/stores/tracking'
import { mapActivity } from '../scripts/sync-strava'
import {
  readStravaCacheFile,
  readStravaCacheFileSync,
  writeStravaCacheFile,
} from './strava-cache-file'
import { reconcileStravaActivities } from './strava-sync-window'
import { triathlonActivityAnchor } from './triathlon-date-route'

// Live sync cannot safely manufacture rounded IDs, corrupt cache keys, malformed provider IDs,
// or conflicting legacy references. Exercise those failure modes at ingestion and persistence.
const ids = ['9007199254740992', '9007199254740993']
const activity = (id: unknown, id_str?: unknown) =>
  mapActivity({ id, id_str, sport_type: 'Ride', start_date: '2026-11-16T12:00:00Z' })

test('activity ingestion preserves exact strings and refuses unrecoverable identities', () => {
  assert.equal(Number(ids[0]), Number(ids[1]))
  assert.deepEqual(
    ids.map(id => activity(Number(id), id).id),
    ids,
  )
  assert.equal(activity(42).id, '42')
  assert.equal(activity('42').id, '42')
  assert.equal(activity(Number.MAX_SAFE_INTEGER).id, String(Number.MAX_SAFE_INTEGER))
  assert.equal(activity(42, '99').id, '99')
  for (const id of [
    undefined,
    null,
    0,
    -1,
    1.5,
    NaN,
    Infinity,
    Number(ids[0]),
    '',
    '0',
    '01',
    ' 42',
    '+42',
    '1e3',
    '1.0',
    'a',
    {},
    true,
  ])
    assert.throws(() => activity(id), /ID/i)
  for (const id_str of [null, '', '01', '1e3', 42, false])
    assert.throws(() => activity(42, id_str), /ID/i)
})

test('cache migration preserves keyed data and exact identities across JSON and reconciliation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'strava-ids-'))
  const path = join(root, 'strava.json')
  const old = {
    version: 8,
    athleteId: 7,
    auth: { refreshToken: '', obtainedAt: 0 },
    lastSync: 0,
    lastActivityStart: 0,
    activities: { '42': { ...activity(42), id: 42 } },
    streams: { '42': { time: [0, 1], latlng: [], altitude: [], distance: [] } },
    activityDetails: { '42': { description: 'keep', fetchedAt: 1 } },
    geo: { '42': 'Toronto' },
  }
  try {
    await writeFile(path, JSON.stringify(old))
    const migrated = await readStravaCacheFile(path)
    assert.ok(migrated)
    assert.equal(migrated.athleteId, '7')
    assert.equal(migrated.activities['42'].id, '42')
    assert.equal(migrated.version, 8, 'ID migration must not trigger a summary backfill')
    assert.deepEqual(migrated.streams, old.streams)
    assert.deepEqual(migrated.activityDetails, old.activityDetails)
    assert.deepEqual(migrated.geo, old.geo)
    const result = reconcileStravaActivities(
      migrated.activities,
      ids.map(id => activity(Number(id), id)),
      0,
    )
    assert.deepEqual(Object.keys(result.activities), ids)
    assert.deepEqual(result.removedIds, ['42'])
    await writeStravaCacheFile(path, { ...migrated, activities: result.activities })
    assert.deepEqual(
      Object.values(readStravaCacheFileSync(path)!.activities).map(a => a.id),
      ids,
    )
    for (const change of [
      { athleteId: Number(ids[0]) },
      { activities: { '42': { ...old.activities['42'], id: Number(ids[0]) } } },
      { activities: { '43': old.activities['42'] } },
      { streams: { '01': {} } },
    ]) {
      await writeFile(path, JSON.stringify({ ...old, ...change }))
      const before = await readFile(path, 'utf8')
      await assert.rejects(readStravaCacheFile(path), /invalid/i)
      assert.throws(() => readStravaCacheFileSync(path), /invalid/i)
      assert.equal(await readFile(path, 'utf8'), before)
    }
    await writeFile(path, JSON.stringify({ ...old, athleteId: 0 }))
    assert.equal((await readStravaCacheFile(path))?.athleteId, null)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('tracking and navigation retain adjacent string IDs while Garmin and manual IDs keep their meaning', () => {
  for (const id of ids) {
    const parsed = parseTrackingBlock(
      null,
      `date: 2026-11-16\nactivity: ${id}\ngarmin: 123\nfueling: 100\nmove: squat 3x10\nskiptraining: true`,
    )
    assert.equal(parsed?.activity?.activityId, id)
    assert.equal(parsed?.activity?.garminActivityId, 123)
    assert.equal(parsed?.fueling?.activityId, id)
    assert.equal(parsed?.trainingExclusion?.activityId, id)
    assert.equal(workoutAnalysisOwner(`tri-workout-analysis-${id}-power-tab`)?.activityId, id)
    const embedded = `embed-2-tri-workout-analysis-${id}`
    assert.deepEqual(workoutAnalysisOwner(`${embedded}-power-tab`), {
      base: embedded,
      activityId: id,
    })
    assert.equal(workoutAnalysisOwner(`embed-2-tri-workout-analysis-0${id}-power-tab`), null)
    assert.equal(triathlonActivityAnchor(id), `tri-activity-${id}`)
  }
  assert.equal(String(manualSaunaActivityId('2026-11-16', '12:30')), '8202611161230')
})
