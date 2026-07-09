import assert from 'node:assert/strict'
import test from 'node:test'
import render from 'preact-render-to-string'
import { isFullSlug, type FullSlug } from '../util/path'
import { BaseViewSelectorMarkup } from './BaseViewSelectorMarkup'

function fullSlug(value: string): FullSlug {
  if (!isFullSlug(value)) throw new Error(`Invalid test slug: ${value}`)
  return value
}

test('base view selector renders native accessible controls', () => {
  const baseSlug = fullSlug('reading')
  const tableSlug = fullSlug('reading/table')
  const mapSlug = fullSlug('reading/map')
  const html = render(
    <BaseViewSelectorMarkup
      fileData={{
        slug: tableSlug,
        basesMetadata: {
          baseSlug,
          currentView: 'Table',
          allViews: [
            { name: 'Table', type: 'table', slug: tableSlug },
            { name: 'Map', type: 'map', slug: mapSlug },
          ],
        },
      }}
    />,
  )

  assert.match(html, /<button type="button" class="text-icon-button"/)
  assert.match(html, /aria-label="Select view" aria-expanded="false"/)
  assert.doesNotMatch(html, /aria-haspopup/)
  assert.match(html, /class="menu-scroll" data-dropdown="true" hidden/)
  assert.match(html, /type="search" placeholder="Search\.\.\." aria-label="Search views"/)
  assert.match(html, /aria-label="Clear view search" data-clear-search="true" hidden/)
  assert.match(html, /data-view-name="Table" data-view-type="table" aria-current="page"/)
  assert.doesNotMatch(html, /role="button"/)
})

test('base view selector omits single-view bases', () => {
  const slug = fullSlug('reading/table')
  const html = render(
    <BaseViewSelectorMarkup
      fileData={{
        slug,
        basesMetadata: {
          baseSlug: fullSlug('reading'),
          currentView: 'Table',
          allViews: [{ name: 'Table', type: 'table', slug }],
        },
      }}
    />,
  )

  assert.equal(html, '')
})
