import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readSharedView, sharedViewUrl, type SharedView } from './share'

const view: SharedView = {
  v: 1,
  base: '/embeddings/latent/build',
  focus: 5457,
  tab: 'features',
  features: {
    selected: { dict: 'text', f: 887 },
    steer: [
      ['text:887', 3.2],
      ['tag:français', -1.85],
    ],
    explain: true,
    query: '',
    pin: false,
    cluster: false,
  },
  bench: {
    yaw: 1.2,
    pitch: -0.4,
    zoom: 2,
    panX: 0.1,
    panY: -0.2,
    axes: [2, 1, 0],
    level: 'feats',
    hidden: 2,
    featsOn: true,
    picked: true,
    feature: { dict: 'image', f: 4 },
  },
}

test('a shared view round-trips signed steering, unicode and the whole camera', () => {
  const url = sharedViewUrl('https://example.com/embeddings/?focus=1#old', view)
  assert.equal(new URL(url).searchParams.get('focus'), '5457')
  assert.deepEqual(readSharedView(new URL(url).hash), view)
})

test('untrusted links reject malformed versions, values and oversized payloads', () => {
  const hash = (value: unknown) => '#view=' + encodeURIComponent(JSON.stringify(value))
  for (const value of [
    null,
    [],
    {},
    { ...view, v: 2 },
    { ...view, focus: -1 },
    { ...view, bench: { ...view.bench, zoom: 0 } },
    { ...view, features: { ...view.features, steer: [['text:1', 99]] } },
    { ...view, features: { ...view.features, selected: { dict: '../bad', f: 1 } } },
  ])
    assert.equal(readSharedView(hash(value)), null)
  assert.equal(readSharedView('#view=%'), null)
  assert.equal(readSharedView('#view=' + 'x'.repeat(100000)), null)
  assert.equal(readSharedView('#ordinary-heading'), null)
})
