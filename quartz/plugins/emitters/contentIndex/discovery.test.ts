import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../../util/ctx'
import type { FullSlug } from '../../../util/path'
import type { ContentDetails, ContentIndexMap } from './link-index'
import { writeSiteMap } from './discovery'

// Failure modes: restricted notes leak into discovery; filtering also drops public notes
// or generated pages without frontmatter; an omitted date crashes sitemap generation.
test('written sitemap preserves public pages and omits indexing restrictions', async () => {
  const output = await mkdtemp(path.join(tmpdir(), 'sitemap-discovery-'))
  const ctx = { argv: { output } } as BuildCtx
  const entries: ContentIndexMap = new Map()
  for (const [slug, frontmatter] of [
    ['public', {}],
    ['explicit-public', { noindex: false, protected: false }],
    ['excluded', { noindex: true }],
    ['protected', { protected: true }],
    ['generated', undefined],
  ] as const) {
    entries.set(
      slug as FullSlug,
      { slug, ...(frontmatter ? { fileData: { frontmatter } } : {}) } as ContentDetails,
    )
  }
  try {
    await writeSiteMap(ctx, { baseUrl: 'example.com' } as BuildCtx['cfg']['configuration'], entries)
    const xml = await readFile(path.join(output, 'sitemap.xml'), 'utf8')
    const urls = [...xml.matchAll(/<loc>(.*?)<\/loc>/g)].map(match => match[1])
    assert.deepEqual(urls, [
      'https://example.com/explicit-public',
      'https://example.com/generated',
      'https://example.com/public',
    ])
    assert.equal(entries.size, 5)
  } finally {
    await rm(output, { recursive: true, force: true })
  }
})
