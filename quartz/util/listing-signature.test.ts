import assert from 'node:assert/strict'
import test from 'node:test'
import type { QuartzPluginData } from '../plugins/vfile'
import type { FullSlug, SimpleSlug } from './path'
import {
  pageListingChanged,
  pageNavigationChanged,
  pageSearchChanged,
  pageSitemapChanged,
} from './listing-signature'

function pageData(overrides: Partial<QuartzPluginData>): QuartzPluginData {
  return {
    slug: 'thoughts/example' as FullSlug,
    description: 'same',
    text: 'body',
    frontmatter: { title: 'Example', pageLayout: 'default', tags: ['seed'] },
    ...overrides,
  }
}

test('page listing signature ignores body-only changes', () => {
  assert.equal(
    pageListingChanged(pageData({ text: 'new body' }), pageData({ text: 'old body' })),
    false,
  )
})

test('page listing signature changes when list metadata changes', () => {
  assert.equal(
    pageListingChanged(
      pageData({ frontmatter: { title: 'Example', pageLayout: 'default', tags: ['fruit'] } }),
      pageData({ frontmatter: { title: 'Example', pageLayout: 'default', tags: ['seed'] } }),
    ),
    true,
  )
})

test('page listing signature changes when feed membership changes', () => {
  assert.equal(
    pageListingChanged(
      pageData({ frontmatter: { title: 'Example', pageLayout: 'default', noindex: true } }),
      pageData({ frontmatter: { title: 'Example', pageLayout: 'default', noindex: false } }),
    ),
    true,
  )
})

test('page navigation signature tracks links without tracking body text', () => {
  assert.equal(
    pageNavigationChanged(
      pageData({ text: 'new body', links: ['thoughts/a' as SimpleSlug] }),
      pageData({ text: 'old body', links: ['thoughts/a' as SimpleSlug] }),
    ),
    false,
  )
  assert.equal(
    pageNavigationChanged(
      pageData({ links: ['thoughts/b' as SimpleSlug] }),
      pageData({ links: ['thoughts/a' as SimpleSlug] }),
    ),
    true,
  )
})

test('page search signature tracks body text', () => {
  assert.equal(pageSearchChanged(pageData({ text: 'body' }), pageData({ text: 'body' })), false)
  assert.equal(
    pageSearchChanged(pageData({ text: 'new body' }), pageData({ text: 'old body' })),
    true,
  )
})

test('page sitemap signature ignores title changes and tracks date changes', () => {
  assert.equal(
    pageSitemapChanged(
      pageData({ frontmatter: { title: 'New', pageLayout: 'default', tags: ['seed'] } }),
      pageData({ frontmatter: { title: 'Old', pageLayout: 'default', tags: ['seed'] } }),
    ),
    false,
  )
  assert.equal(
    pageSitemapChanged(
      pageData({
        frontmatter: {
          title: 'Example',
          pageLayout: 'default',
          tags: ['seed'],
          modified: '2026-05-27',
        },
      }),
      pageData({
        frontmatter: {
          title: 'Example',
          pageLayout: 'default',
          tags: ['seed'],
          modified: '2026-05-26',
        },
      }),
    ),
    true,
  )
})
