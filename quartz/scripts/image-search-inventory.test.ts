import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile, rm, rename } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { test } from 'node:test'
import { isCourseMaterialPath } from '../util/image-search-scope'
import { collectImageSearchInventory } from './image-search-inventory'

async function fixture(files: Record<string, string>, published: string[]) {
  const root = await mkdtemp(path.join(tmpdir(), 'image-inventory-'))
  const contentDir = path.join(root, 'content')
  await mkdir(contentDir)
  for (const [name, text] of Object.entries(files)) {
    const target = path.join(contentDir, name)
    await mkdir(path.dirname(target), { recursive: true })
    await writeFile(target, text)
  }
  const publicIndexPath = path.join(root, 'contentIndex.json')
  await writeFile(
    publicIndexPath,
    JSON.stringify(Object.fromEntries(published.map(slug => [slug, { title: slug }]))),
  )
  return { root, contentDir, publicIndexPath }
}

test('a public occurrence permits publication without exposing private occurrence context', async () => {
  const f = await fixture(
    {
      'public.md': '![public caption](assets/photo.png)',
      'private.md':
        '---\nprivate: true\ntitle: Secret name\n---\n![secret caption](assets/photo.png)',
      'assets/photo.png': 'pixels',
      'orphan.png': 'orphan',
    },
    ['public', 'private'],
  )
  try {
    const result = await collectImageSearchInventory(f)
    const asset = result.assets.find(a => a.sourcePath === 'assets/photo.png')!
    assert.equal(asset.public, true)
    assert.equal(asset.contentId.length, 64)
    assert.deepEqual(
      asset.occurrences.filter(o => o.public).map(o => o.caption),
      ['public caption'],
    )
    assert.equal(result.assets.find(a => a.sourcePath === 'orphan.png')?.public, false)
    assert.ok(
      result.assets.find(a => a.sourcePath === 'orphan.png')?.exclusionReasons.includes('orphan'),
    )
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('private draft protected noindex semantic false and unlisted contexts are excluded', async () => {
  const files: Record<string, string> = { 'photo.jpg': 'same pixels' }
  const published: string[] = []
  for (const key of ['private', 'draft', 'protected', 'noindex', 'semantic']) {
    files[`${key}.md`] =
      `---\n${key}: ${key === 'semantic' ? 'false' : 'true'}\n---\n![[photo.jpg]]`
    published.push(key)
  }
  files['unlisted.md'] = '![[photo.jpg]]'
  const f = await fixture(files, published)
  try {
    const result = await collectImageSearchInventory(f)
    assert.equal(result.assets[0].public, false)
    assert.equal(result.assets[0].occurrences.length, 6)
    assert.ok(result.assets[0].occurrences.every(o => !o.public && o.exclusionReasons.length > 0))
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('Markdown references, wikilinks, HTML, Canvas, Base, and posters resolve locally', async () => {
  const f = await fixture(
    {
      'folder/note.md':
        '---\nposters:\n - "[[assets/a.webp]]"\n---\n![A][photo]\n\n[photo]: ../assets/a.webp\n\n![[a.webp|Wiki caption]]\n\n<img src="/assets/a.webp" alt="HTML caption">\n\n`![[ignored.png]]`',
      'map.canvas': JSON.stringify({
        nodes: [
          { type: 'file', file: 'assets/a.webp' },
          { type: 'text', text: '![[assets/a.webp|Canvas caption]]' },
        ],
      }),
      'gallery.base': 'image: "[[assets/a.webp]]"',
      'assets/a.webp': 'pixels',
    },
    ['folder/note', 'map', 'gallery'],
  )
  try {
    const result = await collectImageSearchInventory(f)
    assert.equal(result.assets.length, 1)
    assert.deepEqual(
      new Set(result.assets[0].occurrences.map(o => o.source)),
      new Set(['markdown', 'html', 'canvas', 'base', 'poster']),
    )
    assert.ok(result.assets[0].occurrences.every(o => o.public))
    assert.equal(result.issues.length, 0)
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('ignored and search-excluded assets remain private even with a public reference', async () => {
  const f = await fixture(
    {
      'note.md': '![[private/p.png]]\n![[hinterland/h.jpg]]\n![[.obsidian/editor.png]]',
      'private/p.png': 'p',
      'hinterland/h.jpg': 'h',
      '.obsidian/editor.png': 'e',
    },
    ['note'],
  )
  try {
    const result = await collectImageSearchInventory({
      ...f,
      ignorePatterns: ['private', '.obsidian'],
    })
    assert.equal(result.assets.length, 3)
    assert.ok(result.assets.every(a => !a.public))
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('ambiguous basenames, missing files, remote URLs and path escapes receive explicit diagnostics', async () => {
  const f = await fixture(
    {
      'note.md':
        '![[a.png]]\n![[missing.jpg]]\n![remote](https://example.com/r.png)\n![escape](../outside.png)',
      'one/a.png': 'one',
      'two/a.png': 'two',
    },
    ['note'],
  )
  try {
    const result = await collectImageSearchInventory(f)
    assert.deepEqual(
      new Set(result.issues.map(i => i.reason)),
      new Set(['ambiguous-reference', 'missing-reference', 'remote-reference', 'outside-vault']),
    )
    assert.ok(result.assets.every(a => !a.public))
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('bytes deduplicate and rename deletion and privacy changes refresh inventory', async () => {
  const f = await fixture(
    { 'note.md': '![[one.png]]\n![[two.png]]', 'one.png': 'pixels', 'two.png': 'pixels' },
    ['note'],
  )
  try {
    const initial = await collectImageSearchInventory(f)
    assert.equal(initial.assets[0].contentId, initial.assets[1].contentId)
    await rename(path.join(f.contentDir, 'one.png'), path.join(f.contentDir, 'renamed.png'))
    await rm(path.join(f.contentDir, 'two.png'))
    await writeFile(path.join(f.contentDir, 'note.md'), '---\nnoindex: true\n---\n![[renamed.png]]')
    const changed = await collectImageSearchInventory(f)
    assert.equal(changed.assets.length, 1)
    assert.equal(changed.assets[0].contentId, initial.assets[0].contentId)
    assert.equal(changed.assets[0].public, false)
    assert.notEqual(changed.digest, initial.digest)
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('unresolved Git LFS image pointers are reported and never published', async () => {
  const f = await fixture(
    {
      'note.md': '![[pointer.png]]',
      'pointer.png': 'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 8\n',
    },
    ['note'],
  )
  try {
    const result = await collectImageSearchInventory(f)
    assert.equal(result.assets[0].public, false)
    assert.ok(result.assets[0].exclusionReasons.includes('lfs-pointer'))
    assert.ok(result.issues.some(i => i.reason === 'lfs-pointer'))
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('hidden and gitignored assets remain local even when referenced by published notes', async () => {
  const f = await fixture(
    {
      'note.md': '![[.hidden/a.png]]\n![[ignored.png]]',
      '.hidden/a.png': 'hidden',
      'ignored.png': 'ignored',
      '.gitignore': 'ignored.png\n',
    },
    ['note'],
  )
  try {
    const result = await collectImageSearchInventory(f)
    assert.equal(result.assets.length, 2)
    assert.ok(result.assets.every(a => !a.public && a.exclusionReasons.includes('ignored-path')))
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('notebook markdown images are inventoried and Obsidian comments do not publish images', async () => {
  const f = await fixture(
    {
      'notebook.ipynb': JSON.stringify({
        nbformat: 4,
        nbformat_minor: 5,
        metadata: {},
        cells: [{ cell_type: 'markdown', metadata: {}, source: ['![Notebook](photo.png)'] }],
      }),
      'note.md': '%% ![[comment.png]] %%',
      'photo.png': 'pixels',
      'comment.png': 'comment',
    },
    ['notebook.ipynb', 'note'],
  )
  try {
    const result = await collectImageSearchInventory(f)
    assert.equal(result.assets.find(a => a.sourcePath === 'photo.png')?.public, true)
    assert.equal(result.assets.find(a => a.sourcePath === 'comment.png')?.public, false)
    assert.equal(
      result.assets.find(a => a.sourcePath === 'photo.png')?.occurrences[0].source,
      'notebook',
    )
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('course material paths and course-only image occurrences remain excluded', async () => {
  const courseRoots = ['courses', 'thoughts/university', 'lectures', 'psets', 'statements']
  const files: Record<string, string> = {
    'public.md': '![[shared.png]]\n![[courses/art.png]]',
    'shared.png': 'shared pixels',
    'course-only.png': 'course pixels',
  }
  const published = ['public']
  for (const root of courseRoots) {
    files[`${root}/note.md`] = '![[shared.png]]\n![[course-only.png]]'
    files[`${root}/art.png`] = root
    published.push(`${root}/note`)
  }
  const f = await fixture(files, published)
  try {
    const result = await collectImageSearchInventory(f)
    for (const root of courseRoots) {
      const asset = result.assets.find(entry => entry.sourcePath === `${root}/art.png`)!
      assert.equal(asset.public, false)
      assert.ok(asset.exclusionReasons.includes('course-material'))
    }
    const courseOnly = result.assets.find(entry => entry.sourcePath === 'course-only.png')!
    assert.equal(courseOnly.public, false)
    assert.ok(courseOnly.exclusionReasons.includes('course-material'))
    assert.ok(
      courseOnly.occurrences.every(
        occurrence => !occurrence.public && occurrence.exclusionReasons.includes('course-material'),
      ),
    )
    const shared = result.assets.find(entry => entry.sourcePath === 'shared.png')!
    assert.equal(shared.public, true)
    assert.equal(shared.exclusionReasons.includes('course-material'), false)
    assert.deepEqual(
      shared.occurrences.filter(occurrence => occurrence.public).map(occurrence => occurrence.slug),
      ['public'],
    )
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('current emitted metadata can be supplied without reading a stale index file', async () => {
  const f = await fixture({ 'note.md': '![[photo.png]]', 'photo.png': 'pixels' }, [])
  try {
    await rm(f.publicIndexPath)
    const current = await collectImageSearchInventory({
      contentDir: f.contentDir,
      publicIndex: { note: { title: 'Current title' } },
    })
    assert.equal(current.assets[0].public, true)
    assert.equal(current.assets[0].occurrences[0].title, 'Current title')
    const removed = await collectImageSearchInventory({ contentDir: f.contentDir, publicIndex: {} })
    assert.equal(removed.assets[0].public, false)
    assert.ok(removed.assets[0].occurrences[0].exclusionReasons.includes('unpublished-note'))
    const preferred = await collectImageSearchInventory({
      ...f,
      publicIndex: { note: { title: 'In-memory wins' } },
    })
    assert.equal(preferred.assets[0].public, true)
    assert.equal(preferred.assets[0].occurrences[0].title, 'In-memory wins')
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('browser-safe course scope normalizes vault paths and respects directory boundaries', () => {
  for (const path of [
    'courses/a.png',
    './content/lectures/a.png',
    'content/psets/a.png',
    'statements/a.png',
    'thoughts/university/a.png',
    'content\\thoughts\\university\\a.png',
  ])
    assert.equal(isCourseMaterialPath(path), true)
  for (const path of [
    'thoughts/images/a.png',
    'library/posters/a.png',
    'courses-archive/a.png',
    'thoughts/university-life/a.png',
  ])
    assert.equal(isCourseMaterialPath(path), false)
})

test('publication mode matches full public IDs, byte digests and public context while skipping excluded sources', async () => {
  const f = await fixture(
    {
      'note.md': '![[assets/a.png]]',
      'private.md':
        '---\nprivate: true\ntitle: Secret title\n---\n![[assets/a.png|Secret caption]]',
      'unlisted.md': '---\ntitle: [\n---\n![[assets/a.png]]',
      'courses/broken.md': '---\ntitle: [\n---\n![[assets/a.png]]',
      'assets/a.png': 'same pixels',
      'orphan.png': 'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 8\n',
    },
    ['note', 'private', 'courses/broken'],
  )
  try {
    const options = {
      ...f,
      publicIndex: {
        note: { title: 'Public title' },
        private: { private: true },
        'courses/broken': {},
      },
    }
    const full = await collectImageSearchInventory(options)
    const publication = await collectImageSearchInventory({ ...options, publicationOnly: true })
    const publicRows = (result: typeof full) =>
      result.assets
        .filter(asset => asset.public)
        .map(asset => ({
          assetId: asset.assetId,
          contentId: asset.contentId,
          bytes: asset.bytes,
          occurrences: asset.occurrences.filter(occurrence => occurrence.public),
        }))
    assert.deepEqual(publicRows(publication), publicRows(full))
    assert.ok(full.issues.some(issue => issue.reason === 'lfs-pointer'))
    assert.ok(full.issues.some(issue => issue.reason.startsWith('parse-error')))
    assert.equal(publication.issues.length, 0)
    assert.ok(publication.assets.every(asset => asset.public))
    assert.equal(JSON.stringify(publication).includes('Secret title'), false)
    assert.equal(JSON.stringify(publication).includes('Secret caption'), false)
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('publication mode retains excluded image paths when resolving ambiguous basenames', async () => {
  const f = await fixture(
    {
      'note.md': '![[same.png]]\n![[public/exact.png]]',
      'public/same.png': 'one',
      'private/same.png': 'two',
      'public/exact.png': 'exact',
    },
    ['note'],
  )
  try {
    const full = await collectImageSearchInventory(f)
    const publication = await collectImageSearchInventory({ ...f, publicationOnly: true })
    const projection = (result: typeof full) =>
      result.assets.filter(asset => asset.public).map(asset => [asset.assetId, asset.contentId])
    assert.deepEqual(projection(publication), projection(full))
    assert.deepEqual(publication.issues, full.issues)
    assert.ok(publication.issues.some(issue => issue.reason === 'ambiguous-reference'))
    assert.deepEqual(
      publication.assets.map(asset => asset.assetId),
      ['public/exact.png'],
    )
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})

test('publication mode immediately removes newly private source context and changed or deleted image bytes', async () => {
  const f = await fixture({ 'note.md': '![[a.png]]\n![[b.png]]', 'a.png': 'a', 'b.png': 'b' }, [
    'note',
  ])
  try {
    const options = { ...f, publicationOnly: true }
    const initial = await collectImageSearchInventory(options)
    await writeFile(path.join(f.contentDir, 'a.png'), 'changed pixels')
    const changed = await collectImageSearchInventory(options)
    assert.notEqual(
      changed.assets.find(asset => asset.assetId === 'a.png')?.contentId,
      initial.assets.find(asset => asset.assetId === 'a.png')?.contentId,
    )
    await writeFile(
      path.join(f.contentDir, 'note.md'),
      '---\nnoindex: true\n---\n![[a.png]]\n![[b.png]]',
    )
    assert.deepEqual((await collectImageSearchInventory(options)).assets, [])
    await writeFile(path.join(f.contentDir, 'note.md'), '![[a.png]]\n![[b.png]]')
    await rm(path.join(f.contentDir, 'a.png'))
    const deleted = await collectImageSearchInventory(options)
    assert.deepEqual(
      deleted.assets.map(asset => asset.assetId),
      ['b.png'],
    )
    const unpublished = await collectImageSearchInventory({ ...options, publicIndex: {} })
    assert.deepEqual(unpublished.assets, [])
  } finally {
    await rm(f.root, { recursive: true, force: true })
  }
})
