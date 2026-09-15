import assert from 'node:assert/strict'
import test from 'node:test'
import { mapboxStyleUrl } from '../quartz/util/mapbox-style'
import { saunaMapTarget } from './sauna-map'

test('sauna static maps use the same Mapbox base styles as the interactive maps', () => {
  for (const location of ['Othership Adelaide', 'Othership Yorkville']) {
    for (const theme of ['light', 'dark']) {
      const target = saunaMapTarget(new URLSearchParams({ location, theme }))
      assert.ok(target)
      assert.equal(target.origin, 'https://api.mapbox.com')
      assert.ok(
        target.pathname.startsWith(
          `/styles/v1/${mapboxStyleUrl('mono', theme === 'dark' ? 'dark' : 'light').replace('mapbox://styles/', '')}/static/`,
        ),
      )
      assert.ok(target.pathname.endsWith('/240x240@2x'))
      assert.equal(target.searchParams.has('access_token'), false)
      assert.ok(
        target.pathname.includes(
          location === 'Othership Adelaide' ? '-79.3977406,43.6460984' : '-79.3922208,43.6694504',
        ),
      )
    }
  }
})

test('sauna map requests only accept registered locations and supported themes', () => {
  for (const params of [
    new URLSearchParams(),
    new URLSearchParams({ location: 'https://example.com' }),
    new URLSearchParams({ location: 'Othership Adelaide', theme: 'streets' }),
  ])
    assert.equal(saunaMapTarget(params), null)
  assert.equal(
    saunaMapTarget(new URLSearchParams({ location: 'Othership Adelaide' }))?.pathname.includes(
      '/light-v11/',
    ),
    true,
  )
})
