import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import test, { after } from 'node:test'
import { createReferenceCatalog } from './catalog'
import { missingReferenceRoute, referenceRoute } from './route'
import { interpretPdfFragment, interpretReference } from './target'

// Resolver failures are observable before rendering: ambiguity, escaped source identity,
// stale metadata, protected immutable snapshots, and target-specific selector interpretation.
const observations: unknown[] = []
const catalog = createReferenceCatalog([
  { path: 'Note.md' },
  { path: 'folder/Note.md' },
  { path: 'other/Note.md' },
  { path: 'folder/Twin.md' },
  { path: 'other/Twin.md' },
  { path: 'Café.md' },
  { path: 'Percent%.md' },
  { path: 'Space name.md' },
  { path: 'Alias.md', aliases: ['Canonical alias'] },
  { path: 'picture.avif' },
  { path: 'Notes.fc' },
  { path: 'Library.base', baseViews: [{ name: 'Books' }, { name: 'To read' }] },
])

test('native established root, explicit-relative, nearby and Unicode lookup policies', () => {
  for (const [input, expected] of [
    ['Note', 'Note.md'],
    ['../Note', 'Note.md'],
    ['/Note', 'Note.md'],
    ['other/Note', 'other/Note.md'],
    ['./Note', 'folder/Note.md'],
    ['note', 'Note.md'],
    ['Twin', 'folder/Twin.md'],
    ['Café', 'Café.md'],
  ]) {
    const result = catalog.resolve(input, 'folder/Lookup.md')
    assert.equal(result.kind, 'found', input)
    if (result.kind === 'found') assert.equal(result.entry.path, expected)
    observations.push({ input, result })
  }
})

test('equal-distance and folded-case conflicts are explicit and deterministic', () => {
  const a = catalog.resolve('Twin', 'root.md')
  assert.equal(a.kind, 'ambiguous')
  if (a.kind === 'ambiguous')
    assert.deepEqual(
      a.candidates.map(e => e.path),
      ['folder/Twin.md', 'other/Twin.md'],
    )
  const collisions = createReferenceCatalog([{ path: 'Case.md' }, { path: 'case.md' }])
  const exact = collisions.resolve('Case', 'host.md')
  assert.equal(exact.kind, 'found')
  if (exact.kind === 'found') assert.equal(exact.entry.path, 'Case.md')
  assert.equal(collisions.resolve('CASE', 'host.md').kind, 'ambiguous')
  observations.push(a)
})

test('literal percent and URI decoding are separate host policies', () => {
  assert.equal(catalog.resolve('Space%20name', 'host.md').kind, 'missing')
  const decoded = catalog.resolve('Space%20name', 'host.md', { decodeUri: true })
  assert.equal(decoded.kind, 'found')
  assert.equal(catalog.resolve('Percent%', 'host.md', { decodeUri: true }).kind, 'found')
  assert.equal(catalog.resolve('https://example.test/a#b', 'host.md').kind, 'external')
  assert.equal(catalog.resolve('../../Note', 'folder/host.md').kind, 'unsupported')
})

test('Garden alias policies retain complete-target precedence', () => {
  assert.equal(catalog.resolve('Canonical alias', 'host.md').kind, 'missing')
  assert.equal(catalog.resolve('Canonical alias', 'host.md', { aliases: true }).kind, 'found')
  const full = catalog.resolve('Space name', 'host.md', { implicitAlias: true })
  assert.equal(full.kind, 'found')
  if (full.kind === 'found') assert.equal(full.implicitLabel, undefined)
  const implicit = catalog.resolve('Note display text', 'host.md', { implicitAlias: true })
  assert.equal(implicit.kind, 'found')
  if (implicit.kind === 'found') {
    assert.equal(implicit.entry.path, 'Note.md')
    assert.equal(implicit.implicitLabel, 'display text')
  }
})

test('catalog snapshots retain source identities despite equal published routes', () => {
  const input = [{ path: 'A B.md', aliases: ['old'] }, { path: 'A-B.md' }]
  const snapshot = createReferenceCatalog(input)
  input[0].aliases?.push('new')
  assert.equal(snapshot.entries.length, 2)
  assert.deepEqual(snapshot.entries[0].aliases, ['old'])
  assert.ok(Object.isFrozen(snapshot.entries))
  assert.ok(Object.isFrozen(snapshot.entries[0]))
  assert.equal(snapshot.resolve('A B', 'host.md').kind, 'found')
  assert.equal(snapshot.resolve('new', 'host.md', { aliases: true }).kind, 'missing')
  assert.ok(snapshot.resolve('missing', 'host.md').lookupKeys.length > 0)
})

test('extensionless files and metadata replacement remain explicit catalog inputs', () => {
  const original = createReferenceCatalog([
    { path: 'LICENSE' },
    { path: 'Host.md', blocks: ['old'] },
  ])
  const replacement = createReferenceCatalog([
    { path: 'LICENSE' },
    { path: 'Host.md', blocks: ['new'] },
  ])
  const license = original.resolve('LICENSE', 'Host.md')
  assert.equal(license.kind, 'found')
  if (license.kind === 'found') assert.equal(license.entry.kind, 'unknown')
  assert.deepEqual(original.entries.find(entry => entry.path === 'Host.md')?.blocks, ['old'])
  assert.deepEqual(replacement.entries.find(entry => entry.path === 'Host.md')?.blocks, ['new'])
})

test('missing-reference publication preserves explicit source-relative paths without inventing membership', () => {
  const origin = 'folder/deep/Host.md'
  assert.deepEqual(missingReferenceRoute('./Missing', origin), { slug: 'folder/deep/Missing' })
  assert.deepEqual(missingReferenceRoute('../Missing.md', origin, ['Parent', 'Last heading']), {
    slug: 'folder/Missing',
    anchor: '#last-heading',
  })
  assert.deepEqual(missingReferenceRoute('/Missing', origin, ['^Block-ID']), {
    slug: 'Missing',
    anchor: '#^Block-ID',
  })
  assert.deepEqual(missingReferenceRoute('Missing', origin), { slug: 'Missing' })
  assert.deepEqual(missingReferenceRoute('folder/new/', origin), { slug: 'folder/new/index' })
  assert.equal(catalog.resolve('./Missing', origin).kind, 'missing')
})

test('missing-reference routes retain Base, flashcard, PDF and same-note fragments', () => {
  assert.deepEqual(missingReferenceRoute('../Library.base', 'folder/Host.md', ['To read']), {
    slug: 'Library/To-read',
  })
  assert.deepEqual(missingReferenceRoute('./Deck.fc', 'folder/Host.md', ['Ignored']), {
    slug: 'folder/Deck/flashcards',
  })
  assert.deepEqual(missingReferenceRoute('../paper.pdf', 'folder/Host.md', ['page=3&height=400']), {
    slug: 'paper.pdf',
    anchor: '#page=3&height=400',
  })
  assert.deepEqual(missingReferenceRoute('', 'folder/Host.md', ['Section']), {
    slug: 'folder/Host',
    anchor: '#section',
  })
  assert.deepEqual(
    missingReferenceRoute('./Notebook.ipynb', 'folder/Host.md', ['cell-1', 'cell-2']),
    { slug: 'folder/Notebook', anchor: '#cell-1#cell-2' },
  )
})

test('missing-reference route projection rejects traversal outside the vault and external URLs', () => {
  assert.equal(missingReferenceRoute('../../Missing', 'folder/Host.md'), undefined)
  assert.equal(missingReferenceRoute('/../Missing', 'folder/Host.md'), undefined)
  assert.equal(missingReferenceRoute('https://example.test/path', 'folder/Host.md'), undefined)
})

test('media kinds and published routes retain source-specific ownership', () => {
  const image = catalog.resolve('picture.avif', 'host.md')
  assert.equal(image.kind, 'found')
  if (image.kind === 'found') assert.equal(image.entry.kind, 'image')
  const deck = catalog.resolve('Notes.fc', 'host.md')
  assert.equal(deck.kind, 'found')
  if (deck.kind === 'found') assert.equal(referenceRoute(deck.entry).slug, 'Notes/flashcards')
  const base = catalog.resolve('Library.base', 'host.md')
  assert.equal(base.kind, 'found')
  if (base.kind === 'found') {
    const first = interpretReference(base.entry, { fragments: ['Books'] })
    const second = interpretReference(base.entry, { fragments: ['To read'] })
    assert.equal(referenceRoute(base.entry, first.selector).slug, 'Library')
    assert.equal(referenceRoute(base.entry, second.selector).slug, 'Library/To-read')
  }
})

test('PDF selectors accept page/height combinations and report invalid values', () => {
  const cases: { anchor: string; expected: { page?: string; height?: string } }[] = [
    { anchor: 'page=3', expected: { page: '3' } },
    { anchor: 'height=400', expected: { height: '400' } },
    { anchor: '#page=3&height=400', expected: { page: '3', height: '400' } },
  ]
  for (const { anchor, expected } of cases) {
    assert.deepEqual(interpretPdfFragment(anchor), { ...expected, diagnostics: [] })
  }
  for (const anchor of ['page=0', 'page=2.5', 'height=-1', 'page=1&page=2', 'zoom=50']) {
    assert.ok(interpretPdfFragment(anchor).diagnostics.length > 0, anchor)
  }
})

test('every image settings body stays CSS, including custom properties and unfamiliar declarations', () => {
  const image = catalog.entries.find(entry => entry.kind === 'image')
  assert.ok(image)
  for (const raw of ['{opacity:0.5}', '{--frame-color:purple}', '{border-radius:12px}']) {
    const result = interpretReference(image, { settings: raw })
    assert.deepEqual(result.settings, { kind: 'css', raw: raw.slice(1, -1) })
    assert.deepEqual(result.diagnostics, [])
  }
})

test('implicit-label lookup retains candidate ambiguity rather than reporting a missing target', () => {
  const result = catalog.resolve('Twin label', 'host.md', { implicitAlias: true })
  assert.equal(result.kind, 'ambiguous')
  if (result.kind === 'ambiguous') assert.equal(result.candidates.length, 2)
})

test('heading paths, literal suffixes and block case use metadata without changing authored text', () => {
  const snapshot = createReferenceCatalog([
    {
      path: 'Source.md',
      headings: [
        { text: 'Repeated', anchor: 'repeated', depth: 1 },
        { text: 'Repeated', anchor: 'repeated-1', depth: 1 },
        { text: 'Repeated-1', anchor: 'repeated-1-1', depth: 1 },
        { text: 'Parent', anchor: 'parent', depth: 1 },
        { text: 'Child', anchor: 'child', depth: 2 },
      ],
      blocks: ['Mixed-ID'],
    },
  ])
  const entry = snapshot.entries[0]
  assert.equal(
    referenceRoute(entry, interpretReference(entry, { fragments: ['Repeated-1'] }).selector).anchor,
    '#repeated-1-1',
  )
  assert.equal(
    referenceRoute(entry, interpretReference(entry, { fragments: ['Parent', 'Child'] }).selector)
      .anchor,
    '#child',
  )
  assert.equal(
    referenceRoute(entry, interpretReference(entry, { fragments: ['^mixed-id'] }).selector).anchor,
    '#^Mixed-ID',
  )
  assert.equal(
    interpretReference(entry, { fragments: ['Missing'] }).diagnostics[0]?.code,
    'missing-selector',
  )
})

test('heading matching preserves intraword punctuation and decodes authored inline formatting', () => {
  const entry = createReferenceCatalog([
    {
      path: 'Syntax.md',
      headings: [
        { text: 'foo_bar', anchor: 'foo_bar' },
        { text: 'Strong title', anchor: 'strong-title' },
      ],
    },
  ]).entries[0]
  assert.equal(
    interpretReference(entry, { fragments: ['foobar'] }).diagnostics[0]?.code,
    'missing-selector',
  )
  assert.equal(
    referenceRoute(entry, interpretReference(entry, { fragments: ['foo_bar'] }).selector).anchor,
    '#foo_bar',
  )
  assert.equal(
    referenceRoute(entry, interpretReference(entry, { fragments: ['**Strong** title'] }).selector)
      .anchor,
    '#strong-title',
  )
  const base = catalog.entries.find(candidate => candidate.kind === 'base')
  assert.ok(base)
  assert.equal(
    referenceRoute(base, interpretReference(base, { fragments: [' books '] }).selector).slug,
    'Library',
  )
})

test('CSS and relaxed object settings retain distinct target-specific representations', () => {
  const image = catalog.entries.find(entry => entry.kind === 'image')
  assert.ok(image)
  const css = interpretReference(image, { settings: '{float:right;width:40%}' })
  assert.deepEqual(css.settings, { kind: 'css', raw: 'float:right;width:40%' })
  const pdf = createReferenceCatalog([{ path: 'book.pdf' }]).entries[0]
  const object = interpretReference(pdf, { settings: '{page:3,height:400}' })
  assert.deepEqual(object.settings, { kind: 'object', value: { page: 3, height: 400 } })
  assert.equal(
    interpretReference(pdf, { settings: '{page:}' }).diagnostics[0]?.code,
    'invalid-settings',
  )
})

after(async () => {
  const directory = path.resolve(
    'docs/verification/obsidian-compatibility/artifacts/phase-3/catalog',
  )
  await mkdir(directory, { recursive: true })
  const receiptPath = path.join(directory, `run-${Date.now()}.json`)
  await writeFile(
    receiptPath,
    JSON.stringify(
      { command: 'node --import tsx --test quartz/util/references/catalog.test.ts', observations },
      null,
      2,
    ) + '\n',
  )
  console.log(`Catalog evidence: ${receiptPath}`)
})
