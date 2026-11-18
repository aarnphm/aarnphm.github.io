import assert from 'node:assert/strict'
import test from 'node:test'
import { createPopupContent, readBaseMapData, readPopupFields } from './base-map-data'

test('readBaseMapData validates markers and applies config defaults', () => {
  const data = readBaseMapData({
    markers: JSON.stringify([
      {
        lat: 43.26,
        lon: -79.92,
        title: 'Hamilton',
        slug: 'places/hamilton',
        popupFields: { status: 'visited' },
      },
    ]),
    currentSlug: 'places/index',
    config: JSON.stringify({ clustering: false }),
    properties: JSON.stringify({ status: { displayName: 'Status' } }),
  })

  assert.ok(data)
  assert.equal(data.markers.length, 1)
  assert.equal(data.markers[0].slug, 'places/hamilton')
  assert.deepEqual(data.config, { defaultZoom: 12, defaultCenter: undefined, clustering: false })
  assert.deepEqual(data.properties, { status: { displayName: 'Status' } })
})

test('readBaseMapData rejects malformed serialized state', () => {
  assert.equal(
    readBaseMapData({ markers: '{', currentSlug: 'places/index', config: '{}' }),
    undefined,
  )
  assert.equal(
    readBaseMapData({
      markers: JSON.stringify([{ lat: '43', lon: -79, title: 'invalid', slug: 'places/x' }]),
      currentSlug: 'places/index',
    }),
    undefined,
  )
  assert.equal(
    readBaseMapData({
      markers: JSON.stringify([{ lat: 143, lon: -79, title: 'invalid', slug: 'places/x' }]),
      currentSlug: 'places/index',
    }),
    undefined,
  )
  assert.equal(readBaseMapData({ markers: '[]', currentSlug: '/invalid/' }), undefined)
})

test('createPopupContent escapes authored values and renders projected reference tokens', () => {
  const data = readBaseMapData({
    markers: JSON.stringify([
      {
        lat: 43.26,
        lon: -79.92,
        title: '<script>alert(1)</script>',
        slug: 'places/hamilton',
        popupFields: {
          raw: '<img src=x onerror=alert(1)>',
          related: { kind: 'reference', href: '/places/toronto#details', label: 'Toronto <b>' },
          icon: { kind: 'icon', value: 'lucide:map-pin' },
        },
      },
    ]),
    currentSlug: 'places/index',
    config: '{}',
    properties: JSON.stringify({ raw: { displayName: '<Raw>' } }),
  })
  assert.ok(data)

  const popup = createPopupContent(data.markers[0], data.currentSlug, data.properties)
  assert.match(popup, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/)
  assert.match(popup, /&lt;Raw&gt;/)
  assert.match(popup, /&lt;img src=x onerror=alert\(1\)&gt;/)
  assert.match(
    popup,
    /href="\/places\/toronto#details" class="internal" data-slug="places\/toronto#details">Toronto &lt;b&gt;<\/a>/,
  )
  assert.match(popup, /<i class="icon-map-pin" aria-hidden="true"><\/i>/)
  assert.doesNotMatch(popup, /<script>|<img/)
})

test('readPopupFields accepts object and serialized Mapbox properties', () => {
  assert.deepEqual(readPopupFields({ status: 'visited' }), { status: 'visited' })
  assert.deepEqual(readPopupFields('{"status":"visited"}'), { status: 'visited' })
  assert.deepEqual(readPopupFields('{'), {})
})

test('popup references validate projected destinations and retain unprojected values as literal text', () => {
  const data = readBaseMapData({
    markers: JSON.stringify([
      {
        lat: 43,
        lon: -79,
        title: 'Place',
        slug: 'places/place',
        popupFields: {
          styled: { kind: 'reference', href: '/places/toronto', label: 'Toronto' },
          view: { kind: 'reference', href: '/places/Visited', label: 'Places' },
          malformed: '[[places/toronto]] trailing]]',
          external: '[[https://example.test|External]]',
          forged: { kind: 'reference', href: 'javascript:alert(1)', label: 'Unsafe' },
          protocolRelative: { kind: 'reference', href: '//evil.test/', label: 'Unsafe' },
        },
      },
    ]),
    currentSlug: 'places/index',
  })
  assert.ok(data)
  const popup = createPopupContent(data.markers[0], data.currentSlug, {})
  assert.match(
    popup,
    /href="\/places\/toronto" class="internal" data-slug="places\/toronto">Toronto<\/a>/,
  )
  assert.match(
    popup,
    /href="\/places\/Visited" class="internal" data-slug="places\/Visited">Places<\/a>/,
  )
  assert.ok(popup.includes('[[places/toronto]] trailing]]'))
  assert.ok(popup.includes('[[https://example.test|External]]'))
  assert.ok(!popup.includes('href="javascript:'))
  assert.ok(!popup.includes('href="//evil.test'))
})
