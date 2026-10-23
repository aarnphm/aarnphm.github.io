import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import {
  IMAGE_ENCODER_CONTRACT,
  publishImageGeneration,
  sha256,
} from '../../scripts/image-embeddings/store'
import { imageSearchOutput } from './image-search-publication'

// Privacy can change after inference. Exercise source files and real vector shards at publication.
test('publication rechecks public references, course scope, source bytes, and stale generations', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'image-publication-'))
  const contentDir = path.join(root, 'content')
  await mkdir(path.join(contentDir, 'images'), { recursive: true })
  await mkdir(path.join(contentDir, 'courses'), { recursive: true })
  const files: Record<string, string> = {
    'note.md': '![[images/a.png]]\n![[images/b.png]]',
    'secret.md': '---\nprivate: true\ntitle: Secret text\n---\n![[images/secret.png]]',
    'images/a.png': 'image a',
    'images/b.png': 'image b',
    'images/secret.png': 'private pixels',
    'images/course.png': 'course pixels',
    'courses/note.md': '![[images/course.png]]',
  }
  for (const [name, data] of Object.entries(files))
    await writeFile(path.join(contentDir, name), data)
  const ids = ['images/a.png', 'images/b.png', 'images/secret.png', 'images/course.png']
  const sourceDigests = Object.fromEntries(ids.map(id => [id, sha256(files[id])]))
  const vector = new Float32Array(768)
  vector[0] = 1
  const source = await publishImageGeneration({
    outDir: path.join(contentDir, 'embeddings/images'),
    rows: ids.map(id => ({ id, vector })),
    contract: IMAGE_ENCODER_CONTRACT,
  })
  // Supply an intentionally overbroad pregenerated snapshot to test the publication boundary.
  await writeFile(
    path.join(contentDir, 'embeddings/images/manifest.json'),
    JSON.stringify({ ...source, sourceDigests }),
  )
  const options = {
    contentDir,
    publicIndex: {
      note: { title: 'Visible' },
      secret: { title: 'Secret text' },
      'courses/note': { title: 'Lecture' },
    },
    ignorePatterns: [],
  }
  try {
    const first = await imageSearchOutput(options)
    assert.ok(first)
    assert.deepEqual(first.manifest.ids, ['images/a.png', 'images/b.png'])
    assert.ok(!JSON.stringify([...first.files]).includes('Secret text'))
    assert.ok([...first.files.keys()].every(file => !file.includes(source.generation)))
    const payload = first.files.get(first.manifest.vectors.shards[0].path.slice(1))
    assert.equal(payload?.byteLength, 2 * 768 * 4)
    await writeFile(path.join(contentDir, 'images/a.png'), 'new bytes before re-encoding')
    const changed = await imageSearchOutput(options)
    assert.deepEqual(changed?.manifest.ids, ['images/b.png'])
    await writeFile(path.join(contentDir, 'note.md'), '---\nnoindex: true\n---\n![[images/b.png]]')
    const privateNow = await imageSearchOutput(options)
    assert.deepEqual(privateNow?.manifest.ids, [])
    assert.equal(privateNow?.manifest.vectors.shards.length, 0)
    await writeFile(path.join(contentDir, 'note.md'), '![[images/b.png]]')
    const unpublished = await imageSearchOutput({ ...options, publicIndex: {} })
    assert.deepEqual(unpublished?.manifest.ids, [])
    await rm(path.join(contentDir, 'images/b.png'))
    const deleted = await imageSearchOutput(options)
    assert.deepEqual(deleted?.manifest.ids, [])
    const bytes = await readFile(path.join(contentDir, source.vectors.shards[0].path))
    bytes[0] ^= 1
    await writeFile(path.join(contentDir, source.vectors.shards[0].path), bytes)
    await assert.rejects(imageSearchOutput(options), /checksum mismatch/)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
