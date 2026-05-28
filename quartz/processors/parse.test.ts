import type { Root as HtmlRoot } from 'hast'
import assert from 'node:assert/strict'
import test from 'node:test'
import { VFile } from 'vfile'
import { ProcessedContent } from '../plugins/vfile'
import {
  canReuseProcessedHtml,
  HTML_PARSE_CHUNK_SIZE,
  parseWorkerConcurrency,
  TEXT_PARSE_CHUNK_SIZE,
  titleOnlyFrontmatterChange,
} from './parse'

function processedContent(source: string, frontmatter: Record<string, unknown>): ProcessedContent {
  const tree: HtmlRoot = { type: 'root', children: [] }
  const file = new VFile(source)
  file.data.frontmatter = { title: 'note', pageLayout: 'default', ...frontmatter }
  return [tree, file]
}

function parsedFile(source: string, frontmatter: Record<string, unknown>): VFile {
  const file = new VFile(source)
  file.data.frontmatter = { title: 'note', pageLayout: 'default', ...frontmatter }
  return file
}

test('parse worker concurrency avoids worker overhead for tiny rebuilds', () => {
  assert.equal(parseWorkerConcurrency(0, 10), 1)
  assert.equal(parseWorkerConcurrency(1, 10), 1)
})

test('parse worker concurrency uses html jobs to balance AST to HAST work', () => {
  assert.equal(parseWorkerConcurrency(TEXT_PARSE_CHUNK_SIZE - 1, 10), 8)
  assert.equal(parseWorkerConcurrency(TEXT_PARSE_CHUNK_SIZE, 10), 8)
  assert.equal(parseWorkerConcurrency(HTML_PARSE_CHUNK_SIZE * 14, 10), 10)
})

test('processed html can be reused for title-only frontmatter changes', () => {
  const previous = processedContent('---\ntitle: old\n---\n# Body\n', { title: 'old' })
  const current = parsedFile('---\ntitle: new\n---\n# Body\n', { title: 'new' })

  assert.equal(canReuseProcessedHtml(current, previous), true)
})

test('title-only frontmatter change returns the current title', () => {
  assert.equal(
    titleOnlyFrontmatterChange('---\ntitle: new\n---\n# Body\n', '---\ntitle: old\n---\n# Body\n'),
    'new',
  )
})

test('title-only frontmatter change is blocked by body changes', () => {
  assert.equal(
    titleOnlyFrontmatterChange(
      '---\ntitle: new\n---\n# Changed\n',
      '---\ntitle: old\n---\n# Body\n',
    ),
    undefined,
  )
})

test('processed html reuse is blocked by body changes', () => {
  const previous = processedContent('---\ntitle: old\n---\n# Body\n', { title: 'old' })
  const current = parsedFile('---\ntitle: new\n---\n# Other\n', { title: 'new' })

  assert.equal(canReuseProcessedHtml(current, previous), false)
})

test('processed html reuse is blocked by html-derived frontmatter changes', () => {
  const previous = processedContent('---\ndescription: old\n---\n# Body\n', { description: 'old' })
  const current = parsedFile('---\ndescription: new\n---\n# Body\n', { description: 'new' })

  assert.equal(canReuseProcessedHtml(current, previous), false)
})
