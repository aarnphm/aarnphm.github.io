import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { syncLatestLocalPaceModels } from './local-pace-models'

function writeModel(
  archiveRoot: string,
  family: 'pace' | 'hr',
  directory: string,
  version: number,
  payload: string,
): void {
  const modelDirectory = path.join(archiveRoot, family, directory)
  mkdirSync(modelDirectory, { recursive: true })
  writeFileSync(path.join(modelDirectory, 'model.safetensors'), payload)
  writeFileSync(
    path.join(modelDirectory, 'manifest.json'),
    JSON.stringify({
      target: family,
      version,
      sha256: createHash('sha256').update(payload).digest('hex'),
      datasetHash: `sha256:${directory}`,
      createdAt: directory,
      val: { mae: version / 100, beatsBaseline: family === 'pace' },
    }),
  )
}

test('local model pointers select the newest archive artifact for each family', async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'local-pace-models-'))
  t.after(async () => rm(root, { recursive: true, force: true }))
  const archiveRoot = path.join(root, '.models-archive')
  const publicRoot = path.join(root, 'public')

  writeModel(archiveRoot, 'pace', '20260715-211534-v10', 12, 'older-pace')
  writeModel(archiveRoot, 'pace', '20260716-172953-v11', 11, 'newest-pace')
  writeModel(archiveRoot, 'hr', '20260716-172947-v9', 9, 'newest-hr')

  assert.deepEqual(syncLatestLocalPaceModels(archiveRoot, publicRoot), [
    { family: 'pace', archive: '20260716-172953-v11', version: 11 },
    { family: 'hr', archive: '20260716-172947-v9', version: 9 },
  ])

  const pacePointer: unknown = JSON.parse(
    readFileSync(path.join(publicRoot, 'models/pace/latest.json'), 'utf8'),
  )
  assert.deepEqual(pacePointer, {
    version: 11,
    manifest: 'models/pace/local-v11/manifest.json',
    weights: 'models/pace/local-v11/model.safetensors',
    sha256: createHash('sha256').update('newest-pace').digest('hex'),
    datasetHash: 'sha256:20260716-172953-v11',
    val: { mae: 0.11, beatsBaseline: true },
    createdAt: '20260716-172953-v11',
  })
  assert.equal(
    readFileSync(path.join(publicRoot, 'models/pace/local-v11/model.safetensors'), 'utf8'),
    'newest-pace',
  )
})

test('local model materialization rejects corrupt latest weights', async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'local-pace-models-'))
  t.after(async () => rm(root, { recursive: true, force: true }))
  const archiveRoot = path.join(root, '.models-archive')
  const publicRoot = path.join(root, 'public')

  writeModel(archiveRoot, 'pace', '20260716-172953-v11', 11, 'pace')
  writeFileSync(path.join(archiveRoot, 'pace/20260716-172953-v11/model.safetensors'), 'corrupt')

  assert.throws(
    () => syncLatestLocalPaceModels(archiveRoot, publicRoot),
    /pace: local safetensors sha256 mismatch/,
  )
})
