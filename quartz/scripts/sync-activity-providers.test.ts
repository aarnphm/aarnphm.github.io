import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  emptyActivityBridgeLedger,
  upsertActivityBridgeReceipt,
  type ActivityBridgeReceipt,
} from '../plugins/stores/activity-bridge'
import {
  parseActivityBridgeArgs,
  parseActivityBridgeLedger,
  readActivityBridgeLedger,
  writeActivityBridgeLedgerAtomic,
} from './sync-activity-providers'

const SHA = 'b'.repeat(64)

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
  assert.deepEqual(parseActivityBridgeArgs([]), { write: false })
  assert.deepEqual(parseActivityBridgeArgs(['--write']), { write: true })
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
