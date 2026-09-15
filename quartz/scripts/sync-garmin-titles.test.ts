import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { ACTIVITY_KINDS } from '../plugins/stores/strava'
import { parseGarminTitleArgs, persistGarminTitleUpdates } from './sync-garmin-titles'

test('Garmin title CLI defaults to all activities and supports explicit sport filters', () => {
  const defaults = parseGarminTitleArgs([])
  assert.equal(defaults.write, false)
  assert.equal(defaults.kind, undefined)
  assert.equal(parseGarminTitleArgs(['--write', '--kind', 'all']).kind, undefined)
  assert.equal(parseGarminTitleArgs(['--write']).write, true)
  for (const kind of ACTIVITY_KINDS) assert.equal(parseGarminTitleArgs(['--kind', kind]).kind, kind)
  assert.equal(parseGarminTitleArgs(['--sport', 'cardio']).kind, 'strength')
  assert.throws(() => parseGarminTitleArgs(['--kind', 'unknown']), /--kind must be/)
})

test('Garmin title CLI preserves pool swim and selection options', () => {
  const args = parseGarminTitleArgs([
    '--kind',
    'swim',
    '--type',
    'pool-swim',
    '--since',
    '2026-09-01',
    '--limit',
    '2',
    '--ids',
    '1,2',
    '--id',
    '3',
  ])
  assert.equal(args.kind, 'swim')
  assert.equal(args.type, 'pool-swim')
  assert.equal(args.since, '2026-09-01')
  assert.equal(args.limit, 2)
  assert.deepEqual(args.ids, new Set(['1', '2', '3']))
})

test('persists only successful Garmin titles while preserving the latest cache fields', async t => {
  const directory = await fs.mkdtemp(join(tmpdir(), 'garmin-titles-'))
  t.after(() => fs.rm(directory, { recursive: true, force: true }))
  const path = join(directory, 'garmin.json')
  const data = {
    lastSync: 1234,
    activities: {
      one: { id: 'connect:1', name: 'Strength', metrics: { totalCalories: 250 } },
      two: { id: 'connect:2', name: 'Yoga' },
    },
    streams: { 'connect:1': { heartrate: [100, 110] } },
  }
  await fs.writeFile(path, JSON.stringify(data))
  await persistGarminTitleUpdates(
    [
      {
        stravaId: 1,
        garminId: 'connect:1',
        garminActivityId: '1',
        from: 'Strength',
        to: 'full body session',
        startDate: '2026-09-14T12:00:00Z',
        startDateLocal: '2026-09-14T08:00:00',
        score: 0,
        startDiffS: 0,
        distanceDiffM: 0,
        durationDiffS: 0,
      },
    ],
    path,
  )
  data.activities.one.name = 'full body session'
  assert.deepEqual(JSON.parse(await fs.readFile(path, 'utf8')), data)
  assert.deepEqual(await fs.readdir(directory), ['garmin.json'])
  await persistGarminTitleUpdates([], join(directory, 'absent.json'))
})
