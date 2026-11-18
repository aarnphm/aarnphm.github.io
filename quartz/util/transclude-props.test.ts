import type { Element, RootContent } from 'hast'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { ReferenceDependency } from './references/dependencies'
import { isFullSlug, type FullSlug } from './path'
import {
  findHeadingSectionBounds,
  getHastClassNames,
  hasHastClass,
  readTranscludeTarget,
  transcludeVisitKey,
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

const request = (overrides: Partial<ReferenceDependency> = {}): ReferenceDependency => ({
  target: 'craft',
  origin: 'host.md',
  fragments: ['Open source'],
  embed: true,
  decodeUri: false,
  implicitAlias: true,
  status: 'found',
  lookupKeys: ['name:craft'],
  resolvedPath: 'thoughts/craft.md',
  resolvedSlug: slug('thoughts/craft'),
  route: { slug: slug('thoughts/craft'), anchor: '#open-source' },
  selector: { kind: 'heading', segments: ['Open source'], anchor: '#open-source' },
  label: 'open-source work',
  resolvedSettings: { kind: 'object', value: { collapsed: true } },
  diagnostics: [],
  ...overrides,
})

function slug(value: string): FullSlug {
  assert.ok(isFullSlug(value))
  return value
}

function typedTransclude(record: ReferenceDependency): Element {
  const node = transcludeElement(
    { className: ['transclude'] },
    { href: './thoughts/craft#open-source', className: ['internal', 'alias'] },
  )
  node.data = { position: {}, referenceRequest: record }
  return node
}

test('reads resolved transclusion semantics without serialized transport properties', () => {
  const target = readTranscludeTarget(typedTransclude(request()))
  assert.ok(target)
  assert.equal(target.targetSlug, 'thoughts/craft')
  assert.equal(target.url, 'thoughts/craft')
  assert.equal(target.blockRef, '#open-source')
  assert.equal(target.alias, 'open-source work')
  assert.deepEqual(target.fragments, ['Open source'])
  assert.deepEqual(target.metadata, { collapsed: true })
})

test('retains an explicitly empty label ahead of an implicit label', () => {
  const target = readTranscludeTarget(
    typedTransclude(request({ label: '', implicitLabel: 'derived caption' })),
  )
  assert.ok(target)
  assert.equal(target.alias, '')
})

test('unlabelled transclusions keep an empty caption for domain presentation defaults', () => {
  const target = readTranscludeTarget(typedTransclude(request({ label: undefined })))
  assert.ok(target)
  assert.equal(target.alias, '')
})

test('canonical heading selections share cycle identity across authored paths and labels', () => {
  const nested = readTranscludeTarget(
    typedTransclude(
      request({
        fragments: ['Parent', 'Open source'],
        selector: { kind: 'heading', segments: ['Parent', 'Open source'], anchor: '#open-source' },
      }),
    ),
  )
  const short = readTranscludeTarget(typedTransclude(request({ label: 'another label' })))
  assert.ok(nested)
  assert.ok(short)
  assert.equal(transcludeVisitKey(nested), transcludeVisitKey(short))
})

test('distinct provider dates and feature fragments retain separate cycle identities', () => {
  const feature = (fragments: string[]) =>
    readTranscludeTarget(
      typedTransclude(
        request({
          target: 'triathlon',
          resolvedPath: 'triathlon.md',
          resolvedSlug: slug('triathlon'),
          route: { slug: slug('triathlon') },
          fragments,
          selector: { kind: 'triathlon', raw: fragments.join('#') },
        }),
      ),
    )
  const fondos = feature(['2026-08-01', 'cycling'])
  const julyThirty = feature(['2026-07-30', 'cycling'])
  assert.ok(fondos)
  assert.ok(julyThirty)
  assert.notEqual(transcludeVisitKey(fondos), transcludeVisitKey(julyThirty))
})

test('whole-page transcludes retain the canonical source slug as cycle identity', () => {
  const target = readTranscludeTarget(
    typedTransclude(
      request({
        fragments: [],
        selector: { kind: 'whole' },
        route: { slug: slug('thoughts/craft') },
      }),
    ),
  )
  assert.ok(target)
  assert.equal(transcludeVisitKey(target), 'thoughts/craft')
})

test('unresolved requests and serialized transport attributes cannot authorize expansion', () => {
  const statuses: ReferenceDependency['status'][] = [
    'missing',
    'ambiguous',
    'external',
    'unsupported',
  ]
  for (const status of statuses) {
    const node = typedTransclude(request({ status }))
    node.properties = {
      ...node.properties,
      'data-url': 'thoughts/craft',
      'data-block': '#open-source',
      'data-metadata': '{"collapsed":true}',
    }
    node.children[0] = {
      type: 'element',
      tagName: 'a',
      properties: { href: './thoughts/craft#open-source', 'data-slug': 'thoughts/craft' },
      children: [],
    }
    assert.equal(readTranscludeTarget(node), undefined, status)
  }
  assert.equal(
    readTranscludeTarget(
      transcludeElement(
        { className: ['transclude'], 'data-url': 'thoughts/craft' },
        { href: './thoughts/craft', 'data-slug': 'thoughts/craft' },
      ),
    ),
    undefined,
  )
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
