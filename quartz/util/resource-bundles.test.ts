import assert from 'node:assert/strict'
import test from 'node:test'
import type { CSSResource, JSResource } from './resources'
import {
  splitCssBundles,
  splitJsBundles,
  staticCssBundleSlug,
  staticJsBundleSlug,
} from './resource-bundles'

test('extracts css resources by content without ordinal bundle names', () => {
  const resources: CSSResource[] = [
    { content: '.a{}', inline: true },
    { content: '/theme.css' },
    { content: '.b{}', inline: true },
  ]

  assert.equal(staticCssBundleSlug, 'static/resource-style')
  assert.deepEqual(splitCssBundles(resources, ['.lead{}']), [
    { type: 'bundle', content: '.lead{}' },
    { type: 'bundle', content: '.a{}' },
    { type: 'resource', resource: { content: '/theme.css' } },
    { type: 'bundle', content: '.b{}' },
  ])
})

test('extracts javascript resources by content without ordinal bundle names', () => {
  const resources: JSResource[] = [
    { script: 'a()', contentType: 'inline', loadTime: 'afterDOMReady' },
    { src: '/vendor.js', contentType: 'external', loadTime: 'afterDOMReady' },
    { script: 'b()', contentType: 'inline', loadTime: 'afterDOMReady' },
    { script: 'before()', contentType: 'inline', loadTime: 'beforeDOMReady' },
  ]

  assert.equal(staticJsBundleSlug('afterDOMReady'), 'static/resource-after')
  assert.deepEqual(splitJsBundles(resources, 'afterDOMReady', ['lead()']), [
    { type: 'bundle', scripts: ['lead()'], loadTime: 'afterDOMReady' },
    { type: 'bundle', scripts: ['a()'], loadTime: 'afterDOMReady' },
    {
      type: 'resource',
      resource: { src: '/vendor.js', contentType: 'external', loadTime: 'afterDOMReady' },
    },
    { type: 'bundle', scripts: ['b()'], loadTime: 'afterDOMReady' },
  ])
})
