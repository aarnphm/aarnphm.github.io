import type { Element } from 'hast'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { ProcessedContent } from '../plugins/vfile'
import type { ChangeEvent } from '../types/plugin'
import { defaultProcessedContent } from '../plugins/vfile'
import { isFilePath, isFullSlug } from './path'
import { transclusionChangeEvents } from './transclusion-dependencies'

function note(name: string, children: Element[] = []): ProcessedContent {
  const path = `content/${name}.md`
  const relativePath = `${name}.md`
  if (!isFullSlug(name) || !isFilePath(path) || !isFilePath(relativePath)) throw new Error(name)
  const result = defaultProcessedContent({ slug: name, filePath: path, relativePath })
  result[0].children = children
  result[1].data.htmlAst = result[0]
  return result
}

function embed(target: string, anchor = '', serialized = false): Element {
  return h('blockquote', { className: ['transclude'], 'data-block': anchor }, [
    h('a', { [serialized ? 'data-slug' : 'dataSlug']: target }, target),
  ])
}

function change(file: ProcessedContent, type: ChangeEvent['type'] = 'change'): ChangeEvent {
  const path = file[1].data.relativePath
  if (!path) throw new Error('missing path')
  return { type, path, file: file[1] }
}

function slugs(content: ProcessedContent[], events: ChangeEvent[]): string[] {
  return transclusionChangeEvents(content, events, 'content')
    .flatMap(event => (event.file?.data.slug ? [event.file.data.slug] : []))
    .sort()
}

test('nested section and block transclusions rebuild through to stream', () => {
  const source = note('source')
  const nested = note('nested', [embed('source', '#^block', true)])
  const stream = note('stream', [embed('nested', '#section')])
  const linkOnly = note('link-only', [h('a', { dataSlug: 'source' }, 'source')])

  assert.deepEqual(slugs([source, nested, stream, linkOnly], [change(source)]), [
    'nested',
    'source',
    'stream',
  ])
})

test('separate triathlon anchors and cycles emit each dependent once', () => {
  const triathlon = note('triathlon', [embed('stream')])
  const stream = note('stream', [
    embed('triathlon', '#2026-09-08'),
    embed('triathlon', '#2026-09-09'),
  ])

  assert.deepEqual(slugs([triathlon, stream], [change(triathlon)]), ['stream', 'triathlon'])
})

test('adding a previously missing target and deleting a target both rebuild stream', () => {
  const source = note('source')
  const stream = note('stream', [embed('source')])

  assert.deepEqual(slugs([stream, source], [change(source, 'add')]), ['source', 'stream'])
  assert.deepEqual(slugs([stream], [change(source, 'delete')]), ['source', 'stream'])
})

test('base view transclusions depend on the owning Base file', () => {
  const base = note('library')
  base[1].data.bases = true
  const stream = note('stream', [embed('library/books')])

  assert.deepEqual(slugs([base, stream], [change(base)]), ['library', 'stream'])
})

test('embedded Base queries rebuild when their input corpus changes', () => {
  const base = note('library')
  base[1].data.bases = true
  const member = note('library/book')
  const nested = note('nested', [embed('library/books')])
  const stream = note('stream', [embed('nested')])

  assert.deepEqual(slugs([base, member, nested, stream], [change(member)]), [
    'library',
    'library/book',
    'nested',
    'stream',
  ])
})

test('code dependency changes propagate through embedded notes for all watcher path forms', () => {
  const source = note('source')
  source[1].data.codeDependencies = ['notes/example.py']
  const stream = note('stream', [embed('source')])

  for (const path of ['notes/example.py', 'content/notes/example.py']) {
    if (!isFilePath(path)) throw new Error(path)
    assert.deepEqual(slugs([source, stream], [{ type: 'change', path }]), ['source', 'stream'])
  }
})

test('stream edits remain a single event and unrelated edits leave stream untouched', () => {
  const source = note('source')
  const stream = note('stream', [embed('source')])
  const other = note('other')

  assert.deepEqual(slugs([source, stream, other], [change(stream)]), ['stream'])
  assert.deepEqual(slugs([source, stream, other], [change(other)]), ['other'])
})
