import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  emptyActivityBridgeLedger,
  planActivityBridge,
  planTrainingPeaksBackfill,
  upsertActivityBridgeReceipt,
  type ActivityBridgeGarminActivity,
  type ActivityBridgeReceipt,
  type ActivityBridgeWahooActivity,
} from '../plugins/stores/activity-bridge'
import {
  parseActivityBridgeArgs,
  parseActivityBridgeGarminActivities,
  parseActivityBridgeLedger,
  readActivityBridgeLedger,
  selectActivityBridgePlans,
  writeActivityBridgeLedgerAtomic,
} from './sync-activity-providers'

const SHA = 'b'.repeat(64)

function garminCache(sports: readonly unknown[]) {
  return {
    activities: Object.fromEntries(
      sports.map((sport, index) => [
        `connect:${index}`,
        {
          id: `connect:${index}`,
          name: `Garmin activity ${index}`,
          sport,
          startDate: '2026-09-08T12:00:00Z',
          startDateLocal: '2026-09-08T08:00:00',
          distanceM: 1000,
          movingTimeS: 600,
          elapsedTimeS: 600,
        },
      ]),
    ),
  }
}

test('accepts valid Garmin activity kinds and excludes unsupported sports from backfill', () => {
  const garmin = parseActivityBridgeGarminActivities(
    garminCache(['walk', 'strength', 'yoga', 'treatment', 'sauna', 'bike', 'run', 'swim', null]),
  )
  assert.deepEqual(
    garmin.map(activity => activity.sport),
    [null, null, null, null, null, 'bike', 'run', 'swim', null],
  )
  const plans = planTrainingPeaksBackfill(
    { strava: [], garmin, wahoo: [] },
    emptyActivityBridgeLedger(),
    'garmin',
  )
  assert.deepEqual(plans.map(plan => plan.sport).sort(), ['bike', 'run', 'swim'])
})

test('rejects malformed Garmin sports', () => {
  assert.throws(
    () => parseActivityBridgeGarminActivities(garminCache(['invalid'])),
    /Garmin activity connect:0.sport is invalid/,
  )
  assert.throws(
    () => parseActivityBridgeGarminActivities(garminCache([42])),
    /Garmin activity connect:0.sport must be a string or null/,
  )
})

function receipt(): ActivityBridgeReceipt {
  return {
    direction: 'garmin-to-wahoo',
    sourceProvider: 'garmin',
    sourceActivityId: 'connect:1',
    sourceFitSha256: SHA,
    destinationProvider: 'wahoo',
    destinationActivityId: 'wahoo:2',
    stravaActivityId: '3',
    uploadToken: 'upload-token',
    uploadStatus: 'complete',
    createdAt: 100,
    updatedAt: 200,
  }
}

test('parses write mode and rejects unknown bridge arguments', () => {
  assert.deepEqual(parseActivityBridgeArgs([]), { write: false, limit: null, direction: null })
  assert.deepEqual(parseActivityBridgeArgs(['--write']), {
    write: true,
    limit: null,
    direction: null,
  })
  assert.deepEqual(parseActivityBridgeArgs(['--write', '--limit', '8']), {
    write: true,
    limit: 8,
    direction: null,
  })
  assert.deepEqual(parseActivityBridgeArgs(['--from', 'wahoo', '--to', 'garmin']), {
    write: false,
    limit: null,
    direction: 'wahoo-to-garmin',
  })
  assert.deepEqual(parseActivityBridgeArgs(['--to', 'wahoo', '--from', 'garmin']), {
    write: false,
    limit: null,
    direction: 'garmin-to-wahoo',
  })
  assert.throws(() => parseActivityBridgeArgs(['--limit', '0']), /positive integer/)
  assert.throws(() => parseActivityBridgeArgs(['--from', 'wahoo']), /--from and --to/)
  assert.throws(() => parseActivityBridgeArgs(['--to', 'garmin']), /--from and --to/)
  assert.throws(() => parseActivityBridgeArgs(['--from', 'strava', '--to', 'garmin']), /--from/)
  assert.throws(() => parseActivityBridgeArgs(['--from', 'wahoo', '--to']), /--to/)
  assert.throws(
    () => parseActivityBridgeArgs(['--from', 'wahoo', '--to', 'wahoo']),
    /different providers/,
  )
  assert.throws(() => parseActivityBridgeArgs(['--delete']), /unknown activity bridge argument/)
})

test('selects the requested provider direction before applying the upload limit', () => {
  const strava = ['1', '2'].map((id, index) => ({
    id,
    name: `Ride ${id}`,
    sportType: 'Ride',
    startDate: `2026-08-2${index + 7}T12:00:00Z`,
    startDateLocal: `2026-08-2${index + 7}T08:00:00`,
    distanceM: 40_000,
    movingTimeS: 5_000,
    elapsedTimeS: 5_200,
  }))
  const wahoo: ActivityBridgeWahooActivity = {
    id: 'wahoo:1',
    name: 'Wahoo ride',
    workoutId: 1,
    sport: 'bike',
    startDate: strava[0].startDate,
    startDateLocal: strava[0].startDateLocal,
    distanceM: 40_000,
    movingTimeS: 5_000,
    elapsedTimeS: 5_200,
    fitUrl: 'https://cdn.wahooligan.com/1.fit',
    fitSha256: SHA,
  }
  const garmin: ActivityBridgeGarminActivity = {
    id: 'connect:2',
    name: 'Garmin ride',
    sport: 'bike',
    startDate: strava[1].startDate,
    startDateLocal: strava[1].startDateLocal,
    distanceM: 40_000,
    movingTimeS: 5_000,
    elapsedTimeS: 5_200,
  }
  const plans = planActivityBridge(
    { strava, garmin: [garmin], wahoo: [wahoo] },
    emptyActivityBridgeLedger(),
  )
  assert.deepEqual(
    plans.map(plan => plan.direction),
    ['wahoo-to-garmin', 'garmin-to-wahoo'],
  )
  assert.deepEqual(
    selectActivityBridgePlans(
      plans,
      parseActivityBridgeArgs(['--from', 'garmin', '--to', 'wahoo', '--limit', '1']),
    ).map(plan => plan.direction),
    ['garmin-to-wahoo'],
  )
  assert.deepEqual(selectActivityBridgePlans(plans, parseActivityBridgeArgs([])), plans)
})

test('atomically persists and reloads the bridge receipt ledger', async t => {
  const root = await fs.mkdtemp(join(os.tmpdir(), 'activity-bridge-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const path = join(root, 'nested', 'ledger.json')
  const ledger = upsertActivityBridgeReceipt(emptyActivityBridgeLedger(), receipt())

  await writeActivityBridgeLedgerAtomic(ledger, path)

  assert.deepEqual(await readActivityBridgeLedger(path), ledger)
  assert.deepEqual(await fs.readdir(join(root, 'nested')), ['ledger.json'])
})

test('uses an empty ledger only when the receipt file does not exist', async t => {
  const root = await fs.mkdtemp(join(os.tmpdir(), 'activity-bridge-missing-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))

  assert.deepEqual(
    await readActivityBridgeLedger(join(root, 'missing.json')),
    emptyActivityBridgeLedger(),
  )
})

test('rejects receipt keys that do not match their provenance payload', () => {
  assert.throws(
    () => parseActivityBridgeLedger({ version: 1, updatedAt: 200, receipts: { wrong: receipt() } }),
    /does not match payload/,
  )
})
