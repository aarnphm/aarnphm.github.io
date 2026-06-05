import type { Element, Root } from 'hast'
import assert from 'node:assert'
import { describe, test } from 'node:test'
import { collectHeadingIndex, resolveNestedAnchor } from './nested-anchor'

function heading(depth: number, id: string, text: string): Element {
  return {
    type: 'element',
    tagName: `h${depth}`,
    properties: { id },
    children: [{ type: 'text', value: text }],
  }
}

function doc(...headings: Element[]): Root {
  return { type: 'root', children: headings }
}

describe('collectHeadingIndex', () => {
  test('reads depth, id, and dedup-stripped base', () => {
    const index = collectHeadingIndex(
      doc(
        heading(2, 'notation', 'Notation'),
        heading(3, 'empty', 'Empty'),
        heading(2, 'operations', 'Operations'),
        heading(3, 'empty-1', 'Empty'),
        heading(2, 'a-cup-b', '(A ∪ B)'),
        heading(2, 'foo-1-2', 'foo 1'),
      ),
    )
    assert.deepStrictEqual(index, [
      { depth: 2, id: 'notation', base: 'notation' },
      { depth: 3, id: 'empty', base: 'empty' },
      { depth: 2, id: 'operations', base: 'operations' },
      { depth: 3, id: 'empty-1', base: 'empty' },
      { depth: 2, id: 'a-cup-b', base: 'a-cup-b' },
      { depth: 2, id: 'foo-1-2', base: 'foo-1' },
    ])
  })

  test('skips headings without an id', () => {
    const noId = heading(2, '', 'Untitled')
    noId.properties = {}
    assert.deepStrictEqual(collectHeadingIndex(doc(noId)), [])
  })
})

describe('resolveNestedAnchor', () => {
  const headings = collectHeadingIndex(
    doc(
      heading(2, 'notation', 'Notation'),
      heading(3, 'empty', 'Empty'),
      heading(2, 'operations', 'Operations'),
      heading(3, 'empty-1', 'Empty'),
    ),
  )

  test('disambiguates duplicate headings by parent segment', () => {
    assert.strictEqual(resolveNestedAnchor(['notation', 'empty'], headings), 'empty')
    assert.strictEqual(resolveNestedAnchor(['operations', 'empty'], headings), 'empty-1')
  })

  test('resolves a single segment to the first match', () => {
    assert.strictEqual(resolveNestedAnchor(['empty'], headings), 'empty')
    assert.strictEqual(resolveNestedAnchor(['notation'], headings), 'notation')
  })

  test('returns undefined when a child segment is not under the parent', () => {
    assert.strictEqual(resolveNestedAnchor(['notation', 'operations'], headings), undefined)
    assert.strictEqual(resolveNestedAnchor(['notation', 'missing'], headings), undefined)
  })

  test('returns undefined for an empty segment list', () => {
    assert.strictEqual(resolveNestedAnchor([], headings), undefined)
  })

  test('honors deeper nesting', () => {
    const deep = collectHeadingIndex(
      doc(
        heading(1, 'sets', 'Sets'),
        heading(2, 'notation', 'Notation'),
        heading(3, 'empty', 'Empty'),
        heading(1, 'maps', 'Maps'),
        heading(2, 'notation-1', 'Notation'),
        heading(3, 'empty-1', 'Empty'),
      ),
    )
    assert.strictEqual(resolveNestedAnchor(['sets', 'notation', 'empty'], deep), 'empty')
    assert.strictEqual(resolveNestedAnchor(['maps', 'notation', 'empty'], deep), 'empty-1')
  })
})
