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
