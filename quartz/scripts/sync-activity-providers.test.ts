import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  emptyActivityBridgeLedger,
  planTrainingPeaksBackfill,
  upsertActivityBridgeReceipt,
  type ActivityBridgeReceipt,
} from '../plugins/stores/activity-bridge'
import {
  parseActivityBridgeArgs,
  parseActivityBridgeGarminActivities,
  parseActivityBridgeLedger,
  readActivityBridgeLedger,
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
  assert.deepEqual(parseActivityBridgeArgs([]), { write: false, limit: null })
  assert.deepEqual(parseActivityBridgeArgs(['--write']), { write: true, limit: null })
  assert.deepEqual(parseActivityBridgeArgs(['--write', '--limit', '8']), { write: true, limit: 8 })
  assert.throws(() => parseActivityBridgeArgs(['--limit', '0']), /positive integer/)
  assert.throws(() => parseActivityBridgeArgs(['--delete']), /unknown activity bridge argument/)
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
