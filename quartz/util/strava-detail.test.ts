import assert from 'node:assert/strict'
import test from 'node:test'
import {
  serializeStravaDetails,
  STRAVA_DETAIL_INDEX_KIND,
  type StravaDetailPayload,
} from './strava-detail'

type TestDetail = { id: number; date: string; name: string }

const detail = (id: number, date: string): TestDetail => ({ id, date, name: `activity ${id}` })

test('serializes independently valid monthly JSON shards under the byte limit', () => {
  const first = detail(1, '2026-08-01')
  const maxShardBytes = Buffer.byteLength(JSON.stringify({ details: { '1': first } }))
  const payload: StravaDetailPayload<TestDetail> = {
    details: { '1': first, '2': detail(2, '2026-08-02'), '3': detail(3, '2026-07-31') },
    health: {},
    ftp: 250,
  }

  const serialized = serializeStravaDetails(payload, maxShardBytes)
  const manifest: { kind: string; shards: string[]; ftp: number } = JSON.parse(serialized.manifest)
  assert.equal(manifest.kind, STRAVA_DETAIL_INDEX_KIND)
  assert.deepEqual(manifest.shards, [
    'strava-detail/2026-08.json',
    'strava-detail/2026-08-2.json',
    'strava-detail/2026-07.json',
  ])
  assert.equal(manifest.ftp, 250)

  const reconstructed: Record<string, TestDetail> = {}
  for (const shard of serialized.shards) {
    assert(shard.bytes <= maxShardBytes)
    assert.doesNotThrow(() => JSON.parse(shard.content))
    const parsed: { details: Record<string, TestDetail> } = JSON.parse(shard.content)
    Object.assign(reconstructed, parsed.details)
  }
  assert.deepEqual(reconstructed, payload.details)
})

test('rejects invalid activity dates and details larger than one shard', () => {
  const invalid: StravaDetailPayload<TestDetail> = {
    details: { '1': detail(1, 'August 1') },
    health: {},
  }
  assert.throws(() => serializeStravaDetails(invalid), /invalid date/)

  const oversized: StravaDetailPayload<TestDetail> = {
    details: { '1': { ...detail(1, '2026-08-01'), name: 'x'.repeat(200) } },
    health: {},
  }
  assert.throws(() => serializeStravaDetails(oversized, 100), /exceeds the shard byte limit/)
})
