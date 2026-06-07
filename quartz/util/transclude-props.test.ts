import type { Element, RootContent } from 'hast'
import assert from 'node:assert/strict'
import test from 'node:test'
import {
  findHeadingSectionBounds,
  getHastClassNames,
  hasHastClass,
  readTranscludeTarget,
} from './transclude-props'

function transcludeElement(
  properties: Element['properties'],
  innerProperties: Element['properties'],
): Element {
  return {
    type: 'element',
    tagName: 'blockquote',
    properties,
    children: [
      {
        type: 'element',
        tagName: 'a',
        properties: innerProperties,
        children: [{ type: 'text', value: 'craft #open-source' }],
      },
    ],
  }
}

test('reads camelCase transclude target properties from render HAST', () => {
  const target = readTranscludeTarget(
    transcludeElement(
      {
        className: ['transclude'],
        dataUrl: 'thoughts/craft',
        dataBlock: '#open-source',
        dataEmbedAlias: 'open-source work',
        dataMetadata: '{"collapsed":true}',
      },
      {
        href: './thoughts/craft#open-source',
        className: ['internal', 'alias'],
        dataSlug: 'thoughts/craft',
      },
    ),
  )

  assert.ok(target)
  assert.equal(target.targetSlug, 'thoughts/craft')
  assert.equal(target.url, 'thoughts/craft')
  assert.equal(target.blockRef, '#open-source')
  assert.equal(target.alias, 'open-source work')
  assert.equal(target.rawMetadata, '{"collapsed":true}')
})

test('reads dashed transclude target properties from serialized-style HAST', () => {
  const target = readTranscludeTarget(
    transcludeElement(
      {
        class: 'transclude',
        'data-url': 'thoughts/craft',
        'data-block': '#open-source',
        'data-embed-alias': 'undefined',
      },
      {
        href: './thoughts/craft#open-source',
        className: ['internal', 'alias'],
        'data-slug': 'thoughts/craft',
      },
    ),
  )

  assert.ok(target)
  assert.equal(target.targetSlug, 'thoughts/craft')
  assert.equal(target.url, 'thoughts/craft')
  assert.equal(target.blockRef, '#open-source')
  assert.equal(target.alias, '#open-source')
})

test('reads HAST classes from parser and render property names', () => {
  const node = transcludeElement(
    { class: 'blockquote transclude', className: ['transclude', 'is-collapsed'] },
    { 'data-slug': 'thoughts/craft' },
  )

  assert.deepEqual(getHastClassNames(node), ['transclude', 'is-collapsed', 'blockquote'])
  assert.equal(hasHastClass(node, 'transclude'), true)
  assert.equal(hasHastClass(node, 'missing'), false)
})

test('ignores transcludes without a valid target slug', () => {
  const target = readTranscludeTarget(
    transcludeElement(
      { className: ['transclude'], dataUrl: 'thoughts/craft' },
      { href: './thoughts/craft#open-source', className: ['internal', 'alias'] },
    ),
  )

  assert.equal(target, undefined)
})

test('finds direct heading section bounds', () => {
  const children: RootContent[] = [
    {
      type: 'element',
      tagName: 'h2',
      properties: { id: 'open-source' },
      children: [{ type: 'text', value: 'open source' }],
    },
    {
      type: 'element',
      tagName: 'p',
      properties: {},
      children: [{ type: 'text', value: 'section body' }],
    },
    {
      type: 'element',
      tagName: 'h2',
      properties: { id: 'lives' },
      children: [{ type: 'text', value: 'lives' }],
    },
  ]

  assert.deepEqual(findHeadingSectionBounds(children, 'open-source'), { startIdx: 0, endIdx: 2 })
})

test('finds wrapped collapsible heading section bounds', () => {
  const children: RootContent[] = [
    { type: 'element', tagName: 'p', properties: {}, children: [{ type: 'text', value: 'intro' }] },
    {
      type: 'element',
      tagName: 'section',
      properties: { className: ['collapsible-header'], id: 'open-source' },
      children: [
        {
          type: 'element',
          tagName: 'div',
          properties: {},
          children: [
            {
              type: 'element',
              tagName: 'h2',
              properties: { id: 'open-source' },
              children: [{ type: 'text', value: 'open source' }],
            },
          ],
        },
      ],
    },
    {
      type: 'element',
      tagName: 'section',
      properties: { className: ['collapsible-header'], id: 'lives' },
      children: [
        {
          type: 'element',
          tagName: 'h2',
          properties: { id: 'lives' },
          children: [{ type: 'text', value: 'lives' }],
        },
      ],
    },
  ]

  assert.deepEqual(findHeadingSectionBounds(children, 'open-source'), { startIdx: 1, endIdx: 2 })
})
