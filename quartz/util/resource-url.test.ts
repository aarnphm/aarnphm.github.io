import assert from 'node:assert/strict'
import test from 'node:test'
import type { FullSlug, TransformOptions } from './path'
import { isFullSlug } from './path'
import { transformResourceUrl } from './resource-url'

const transformOptions: TransformOptions = { strategy: 'absolute', allSlugs: [] }

function fullSlug(value: string): FullSlug {
  assert.ok(isFullSlug(value))
  return value
}

test('normalizes pdf resource paths from the vault root', () => {
  assert.equal(
    transformResourceUrl(
      fullSlug('courses/18.901-fall-2004/resources/18901/index'),
      'courses/18.901-fall-2004/static_resources/0162d186ff55f17b25d9c57f6fd211cc_18901.pdf',
      transformOptions,
    ),
    '/courses/18.901-fall-2004/static_resources/0162d186ff55f17b25d9c57f6fd211cc_18901.pdf',
  )
})

test('preserves explicit relative pdf resource paths', () => {
  assert.equal(
    transformResourceUrl(
      fullSlug('courses/example/resources/lecture/index'),
      '../static_resources/paper.pdf',
      transformOptions,
    ),
    '../static_resources/paper.pdf',
  )
})
