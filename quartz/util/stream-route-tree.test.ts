import type { Element, ElementContent, Root } from 'hast'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { StreamEntry } from '../plugins/transformers/stream'
import { isFullSlug, type FullSlug } from './path'
import { buildStreamRouteTree, cloneStreamEntries, rebaseStreamEntries } from './stream-route-tree'

function slug(value: string): FullSlug {
  if (!isFullSlug(value)) throw new Error(`invalid slug: ${value}`)
  return value
}

function text(value: string): ElementContent {
  return { type: 'text', value }
}

function el(
  tagName: string,
  properties: Element['properties'],
  children: ElementContent[],
): Element {
  return { type: 'element', tagName, properties, children }
}

function entry(id: string, content: ElementContent[]): StreamEntry {
  return { id, metadata: {}, content }
}

test('buildStreamRouteTree includes only selected entry content', () => {
  const first = el('p', { id: 'first' }, [text('first')])
  const second = el('p', { id: 'second' }, [text('second')])
  const sourceTree: Root = { type: 'root', children: [first, second] }

  const routeTree = buildStreamRouteTree([entry('first', [first])], sourceTree)

  assert.deepEqual(routeTree.children, [first])
})

test('buildStreamRouteTree carries footnotes only when selected content references them', () => {
  const withoutRef = el('p', { id: 'plain' }, [text('plain')])
  const footnoteRef = el('a', { href: '#fn-one', dataFootnoteRef: '' }, [text('1')])
  const withRef = el('p', { id: 'with-ref' }, [text('with ref'), footnoteRef])
  const footnotes = el('section', { dataFootnotes: '' }, [
    el('h2', { id: 'footnote-label' }, [text('notes')]),
    el('ol', {}, [el('li', { id: 'fn-one' }, [text('note')])]),
  ])
  const sourceTree: Root = { type: 'root', children: [withoutRef, withRef, footnotes] }

  assert.deepEqual(buildStreamRouteTree([entry('plain', [withoutRef])], sourceTree).children, [
    withoutRef,
  ])
  assert.deepEqual(buildStreamRouteTree([entry('with-ref', [withRef])], sourceTree).children, [
    withRef,
    footnotes,
  ])
})

test('buildStreamRouteTree carries bibliography only when selected content references it', () => {
  const citationRef = el('a', { href: '#bib-paper' }, [text('paper')])
  const withRef = el('p', { id: 'with-citation' }, [citationRef])
  const references = el('section', { dataReferences: '' }, [
    el('h2', { id: 'reference-label' }, [text('refs')]),
    el('ul', {}, [el('li', { id: 'bib-paper' }, [text('paper')])]),
  ])
  const sourceTree: Root = { type: 'root', children: [withRef, references] }

  assert.deepEqual(buildStreamRouteTree([entry('with-citation', [withRef])], sourceTree).children, [
    withRef,
    references,
  ])
})

test('stream entry cloning and rebasing do not mutate source entries', () => {
  const link = el('a', { href: './image.png' }, [text('image')])
  const original = entry('source', [link])
  const [cloned] = cloneStreamEntries([original])
  const [rebased] = rebaseStreamEntries([original], slug('stream/on/2026/06/19'), slug('stream'))

  assert.notEqual(cloned, original)
  assert.notEqual(cloned.content[0], original.content[0])
  assert.notEqual(rebased.content[0], original.content[0])

  if (cloned.content[0].type !== 'element') assert.fail('expected cloned element')
  cloned.content[0].properties.href = '/mutated'

  assert.equal(link.properties.href, './image.png')
})
