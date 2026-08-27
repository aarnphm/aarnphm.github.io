import assert from 'node:assert/strict'
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import type { StravaRawCache } from '../plugins/stores/strava'
import {
  readStravaCacheFile,
  readStravaCacheFileSync,
  writeStravaCacheFile,
} from './strava-cache-file'

function cache(lastSync: number): StravaRawCache {
  return {
    version: 4,
    athleteId: 1,
    auth: { refreshToken: 'refresh', obtainedAt: lastSync },
    lastSync,
    lastActivityStart: 0,
    activities: {},
  }
}

test('reads a missing Strava cache as absent and rejects malformed cache bytes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'strava-cache-file-'))
  const path = join(root, 'strava.json')
  try {
    assert.equal(await readStravaCacheFile(path), null)
    assert.equal(readStravaCacheFileSync(path), null)
    await writeFile(path, '{')
    await assert.rejects(readStravaCacheFile(path), /Strava cache .* is invalid/)
    assert.throws(() => readStravaCacheFileSync(path), /Strava cache .* is invalid/)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('atomically replaces the Strava cache without leaving temporary files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'strava-cache-file-'))
  const path = join(root, 'strava.json')
  try {
    await writeStravaCacheFile(path, cache(1))
    await writeStravaCacheFile(path, cache(2))
    assert.equal((await readStravaCacheFile(path))?.lastSync, 2)
    assert.equal(readStravaCacheFileSync(path)?.lastSync, 2)
    assert.deepEqual(await readdir(root), ['strava.json'])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
