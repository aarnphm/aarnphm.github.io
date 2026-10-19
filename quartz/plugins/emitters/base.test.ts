import assert from 'node:assert/strict'
import test from 'node:test'
import type { FilePath, FullSlug, SimpleSlug } from '../../util/path'
import type { ProcessedContent } from '../vfile'
import { compileBaseConfig } from '../../util/base/compile'
import { planBaseViewPartialEmit } from '../../util/base/partial-emit'
import { defaultProcessedContent } from '../vfile'

const baseSource = `
filters:
  and:
    - file.inFolder("library")
views:
  - type: table
    name: all
    order:
      - file.name
      - file.backlinks
`

function baseFile(): ProcessedContent {
  const compiled = compileBaseConfig(baseSource, 'library.base')
  return defaultProcessedContent({
    slug: 'library' as FullSlug,
    filePath: 'library.base' as FilePath,
    relativePath: 'library.base' as FilePath,
    bases: true,
    basesConfig: compiled.config,
    basesDiagnostics: compiled.diagnostics,
    basesExpressions: compiled.expressions,
    frontmatter: { title: 'library', pageLayout: 'default', tags: ['bases'] },
  })
}

function note(slug: string, relativePath: string, links: string[] = []): ProcessedContent {
  const title = slug.split('/').pop() ?? slug
  return defaultProcessedContent({
    slug: slug as FullSlug,
    filePath: relativePath as FilePath,
    relativePath: relativePath as FilePath,
    frontmatter: { title, pageLayout: 'default' },
    links: links.map(link => link as SimpleSlug),
  })
}

const rebuiltSlugs = (plan: ReturnType<typeof planBaseViewPartialEmit>): string[] =>
  [...plan.slugsToRebuild].sort()

test('base view partial emit ignores unrelated markdown changes', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const book = note('library/book', 'library/book.md')
  const content = [base, index, book]
  const previous = planBaseViewPartialEmit(content, []).nextState

  const plan = planBaseViewPartialEmit(
    content,
    [{ type: 'change', path: 'index.md' as FilePath, file: index[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), [])
})

test('base view partial emit rebuilds when a matched file changes', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const book = note('library/book', 'library/book.md')
  const content = [base, index, book]
  const previous = planBaseViewPartialEmit(content, []).nextState

  const plan = planBaseViewPartialEmit(
    content,
    [{ type: 'change', path: 'library/book.md' as FilePath, file: book[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('base view partial emit ignores matched file body changes with stable metadata', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const previousBook = note('library/book', 'library/book.md')
  const previousContent = [base, index, previousBook]
  const previous = planBaseViewPartialEmit(previousContent, []).nextState
  const nextBook = note('library/book', 'library/book.md')

  const plan = planBaseViewPartialEmit(
    [base, index, nextBook],
    [
      {
        type: 'change',
        path: 'library/book.md' as FilePath,
        file: nextBook[1],
        previousFile: previousBook[1],
      },
    ],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), [])
})

test('base view partial emit rebuilds when matched file metadata changes', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const previousBook = note('library/book', 'library/book.md')
  const previousContent = [base, index, previousBook]
  const previous = planBaseViewPartialEmit(previousContent, []).nextState
  const nextBook = note('library/book', 'library/book.md')
  nextBook[1].data.frontmatter = { title: 'book, revised', pageLayout: 'default' }

  const plan = planBaseViewPartialEmit(
    [base, index, nextBook],
    [
      {
        type: 'change',
        path: 'library/book.md' as FilePath,
        file: nextBook[1],
        previousFile: previousBook[1],
      },
    ],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('base view partial emit rebuilds when a file enters the base result set', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const previousContent = [base, index]
  const previous = planBaseViewPartialEmit(previousContent, []).nextState
  const book = note('library/book', 'library/book.md')

  const plan = planBaseViewPartialEmit(
    [base, index, book],
    [{ type: 'add', path: 'library/book.md' as FilePath, file: book[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('base view partial emit rebuilds when backlinks to a matched file change', () => {
  const base = baseFile()
  const book = note('library/book', 'library/book.md')
  const oldIndex = note('index', 'index.md')
  const previousContent = [base, oldIndex, book]
  const previous = planBaseViewPartialEmit(previousContent, []).nextState
  const newIndex = note('index', 'index.md', ['library/book'])

  const plan = planBaseViewPartialEmit(
    [base, newIndex, book],
    [{ type: 'change', path: 'index.md' as FilePath, file: newIndex[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('base view partial emit rebuilds when a linked non-member changes', () => {
  const base = baseFile()
  const index = note('index', 'index.md')
  const book = note('library/book', 'library/book.md', ['/'])
  const content = [base, index, book]
  const previous = planBaseViewPartialEmit(content, []).nextState

  const plan = planBaseViewPartialEmit(
    content,
    [{ type: 'change', path: 'index.md' as FilePath, file: index[1] }],
    previous,
  )

  assert.deepEqual(rebuiltSlugs(plan), ['library'])
})

test('createBaseQueryData compacts empty arrays, default filePath/pageLayout, and pdfTitles while round-tripping via readBaseQueryData', async () => {
  const { createBaseQueryData, readBaseQueryData } = await import('../../util/base/browser-data')
  const book = note('library/book', 'library/book.md')[1].data
  book.frontmatter = {
    title: 'Book Title',
    pageLayout: 'default',
    author: 'Ada Lovelace',
    pdfTitles: { 'folder/paper.pdf': 'Internal PDF Title' },
  }
  const payload = createBaseQueryData([book])
  assert.equal(payload.files.length, 1)
  const rawRecord = payload.files[0] as unknown as Record<string, unknown>
  assert.equal(rawRecord.filePath, undefined)
  assert.equal(rawRecord.datePaths, undefined)
  assert.equal(rawRecord.links, undefined)
  assert.equal(rawRecord.embeds, undefined)
  assert.equal((rawRecord.frontmatter as Record<string, unknown>).pageLayout, undefined)
  assert.equal((rawRecord.frontmatter as Record<string, unknown>).pdfTitles, undefined)

  const hydrated = readBaseQueryData(JSON.parse(JSON.stringify(payload)))
  assert.equal(hydrated.length, 1)
  assert.equal(hydrated[0].slug, 'library/book')
  assert.equal(hydrated[0].filePath, 'library/book.md')
  assert.equal(hydrated[0].frontmatter?.title, 'Book Title')
  assert.equal(hydrated[0].frontmatter?.pageLayout, 'default')
  assert.equal(hydrated[0].frontmatter?.author, 'Ada Lovelace')
  assert.deepEqual(hydrated[0].links, [])
  assert.deepEqual(hydrated[0].embeds, [])
})

test('createBaseQueryCatalog excludes nested dictionary bloat and this.file.* from sort properties while keeping top-level and configured properties', async () => {
  const { createBaseQueryCatalog } = await import('../../components/scripts/base-query-editor')
  const compiled = compileBaseConfig(
    `
formulas:
  age_years: 2026 - number(year)
properties:
  note.custom.nested:
    displayName: Custom Nested
views:
  - type: table
    name: all
`,
    'library.base',
  )
  const book = note('library/book', 'library/book.md')[1].data
  book.frontmatter = {
    title: 'Book Title',
    pageLayout: 'default',
    author: 'Ada Lovelace',
    year: 1843,
    pdfTitles: { 'folder/scan.pdf': 'Scan Title' },
    socials: { twitter: 'https://x.com/ada' },
  }
  const catalog = createBaseQueryCatalog([book], compiled.config)
  assert.ok(catalog.properties.includes('note.author'))
  assert.ok(catalog.properties.includes('note.year'))
  assert.ok(catalog.properties.includes('formula.age_years'))
  assert.ok(catalog.properties.includes('note.custom.nested'))
  assert.ok(catalog.properties.includes('file.name'))
  assert.ok(!catalog.properties.some(prop => prop.startsWith('note.pdfTitles')))
  assert.ok(!catalog.properties.some(prop => prop.startsWith('note.socials.')))
  assert.ok(!catalog.properties.some(prop => prop.startsWith('this.')))
})

test('renderBaseViewsForFile marks card images for thumbnail generation and omits empty table cell wrappers', async () => {
  const { renderBaseViewsForFile } = await import('../../util/base/render')
  const { toHtml } = await import('hast-util-to-html')
  const compiled = compileBaseConfig(
    `
filters:
  and:
    - file.inFolder("library")
views:
  - type: cards
    name: gallery
    image: note.posters
    order:
      - title
      - author
  - type: table
    name: editors
    order:
      - title
      - author
      - missing_col
      - read_flag
`,
    'library.base',
  )
  const baseData = defaultProcessedContent({
    slug: 'library' as FullSlug,
    filePath: 'library.base' as FilePath,
    relativePath: 'library.base' as FilePath,
    bases: true,
    basesConfig: compiled.config,
    basesDiagnostics: compiled.diagnostics,
    basesExpressions: compiled.expressions,
  })[1].data
  const book = note('library/book', 'library/book.md')[1].data
  book.frontmatter = {
    title: 'Book Title',
    pageLayout: 'default',
    author: 'Ada Lovelace',
    posters: '[[library/posters/book.jpg]]',
    read_flag: true,
  }
  const rendered = renderBaseViewsForFile(baseData, [book], baseData)
  const galleryHtml = toHtml(rendered.views[0].tree)
  assert.match(galleryHtml, /data-bookcase-cover="library\/posters\/book\.jpg"/)

  const tableHtml = toHtml(rendered.views[1].tree)
  assert.match(tableHtml, /<td><\/td>/)
  assert.doesNotMatch(tableHtml, /<div class="base-table-cell-content"><\/div>/)
})
