import { XMLParser } from 'fast-xml-parser'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug } from '../../util/path'
import type { QuartzPluginData } from '../vfile'
import { generateStreamAtomFeed } from '../../util/stream-feed'
import { buildStreamManifestGroup } from '../../util/stream-manifest'
import { isRecord, type UnknownRecord } from '../../util/type-guards'
import { renderedStreamEntries } from './streamRenderedText'

function testCtx(): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: 'content',
      verbose: false,
      output: 'public',
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'test garden',
        enableSPA: true,
        enablePopovers: true,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'modified',
        baseUrl: 'example.com',
        locale: 'en-US',
        theme: {} as BuildCtx['cfg']['configuration']['theme'],
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

function readRecord(record: UnknownRecord, key: string): UnknownRecord {
  const value = record[key]
  assert.ok(isRecord(value))
  return value
}

function readString(record: UnknownRecord, key: string): string {
  const value = record[key]
  if (typeof value !== 'string') {
    throw new Error(`expected ${key} to be a string`)
  }
  return value
}

function streamEntries(xml: string): UnknownRecord[] {
  const parsed: unknown = new XMLParser({ ignoreAttributes: false }).parse(xml)
  assert.ok(isRecord(parsed))
  const feed = readRecord(parsed, 'feed')
  const entries = feed.entry
  if (Array.isArray(entries)) {
    assert.ok(entries.every(isRecord))
    return entries
  }
  assert.ok(isRecord(entries))
  return [entries]
}

function streamFeed(xml: string): UnknownRecord {
  const parsed: unknown = new XMLParser({ ignoreAttributes: false }).parse(xml)
  assert.ok(isRecord(parsed))
  return readRecord(parsed, 'feed')
}

test('stream atom feed uses entry descriptions as metadata summaries', () => {
  const fileData: QuartzPluginData = {
    slug: 'stream' as FullSlug,
    filePath: 'stream.md' as FilePath,
    frontmatter: {
      title: 'stream',
      pageLayout: 'default',
      description: 'stream index',
      modified: '2026-06-11T00:00:00.000Z',
    },
    streamData: {
      entries: [
        {
          id: 'protected-entry',
          title: 'protected title',
          description: 'the grief is real and $O(n)$',
          descriptionHtml: '<p>the grief is real and <em>O(n)</em></p>',
          metadata: { protected: true, tags: ['o/m'] },
          content: [
            {
              type: 'element',
              tagName: 'p',
              properties: {},
              children: [{ type: 'text', value: 'secret body' }],
            },
          ],
          date: '2026-06-10T00:00:00.000Z',
          timestamp: Date.parse('2026-06-10T00:00:00.000Z'),
        },
        {
          id: 'public-entry',
          title: 'public title',
          description: 'public summary',
          descriptionHtml: '<p>public summary</p>',
          metadata: { tags: ['note'] },
          content: [
            {
              type: 'element',
              tagName: 'p',
              properties: {},
              children: [{ type: 'text', value: 'body only' }],
            },
          ],
          date: '2026-06-09T00:00:00.000Z',
          timestamp: Date.parse('2026-06-09T00:00:00.000Z'),
        },
      ],
    },
  }

  const [protectedEntry, publicEntry] = streamEntries(generateStreamAtomFeed(testCtx(), fileData))
  assert.equal(readString(protectedEntry, 'summary'), 'the grief is real and O(n)')
  assert.equal(readRecord(protectedEntry, 'content')['#text'], undefined)

  assert.equal(readString(publicEntry, 'summary'), 'public summary')
  assert.equal(readString(readRecord(publicEntry, 'content'), '#text'), '<p>body only</p>')
})

test('stream manifest keeps searchable public metadata without rendered entry markup', () => {
  const manifest = buildStreamManifestGroup(
    {
      id: 'day-2026-06-09',
      timestamp: Date.parse('2026-06-09T00:00:00.000Z'),
      isoDate: '2026-06-09T00:00:00.000Z',
      entries: [
        {
          id: 'public-entry',
          title: 'public title',
          description: 'public summary',
          metadata: { tags: ['note'] },
          content: [
            {
              type: 'element',
              tagName: 'p',
              properties: {},
              children: [{ type: 'text', value: 'body only' }],
            },
          ],
          date: '2026-06-09T01:00:00.000Z',
        },
        {
          id: 'private-entry',
          title: 'private title',
          metadata: { private: true },
          content: [{ type: 'text', value: 'secret body' }],
          date: '2026-06-09T02:00:00.000Z',
        },
      ],
    },
    () => ({ content: 'body only', wordCount: 6 }),
  )

  assert.ok(manifest)
  assert.equal(manifest.path, '/stream/on/2026/06/09')
  assert.equal(manifest.groupSize, 1)
  assert.equal(manifest.entries.length, 1)
  assert.equal(manifest.entries[0].id, 'public-entry')
  assert.equal(manifest.entries[0].content, 'body only')
  assert.equal(manifest.entries[0].wordCount, 6)
  assert.equal(Object.hasOwn(manifest.entries[0], 'html'), false)
})

test('stream manifest content comes from the rendered daily entry', () => {
  const entries = renderedStreamEntries(`
    <ol class="stream-feed">
      <li class="stream-entry" data-entry-id="entry-1">
        <div class="stream-entry-body">
          <h2>ride</h2>
          <div class="stream-entry-content">
            <blockquote><table><tr><th>intensity factor</th><td>0.768</td></tr></table></blockquote>
            <p>Almost got hit by BMW.</p>
          </div>
          <div class="stream-entry-wordcount"><em>12 words</em></div>
        </div>
      </li>
    </ol>
  `)

  assert.deepEqual(entries.get('entry-1'), {
    content: 'intensity factor 0.768 Almost got hit by BMW.',
    wordCount: 12,
  })
})

test('stream atom feed uses the canonical stream host for feed and entry URLs', () => {
  const fileData: QuartzPluginData = {
    slug: 'stream' as FullSlug,
    filePath: 'stream.md' as FilePath,
    frontmatter: { title: 'stream', pageLayout: 'default', modified: '2026-06-11T00:00:00.000Z' },
    streamData: {
      entries: [
        {
          id: 'public-entry',
          title: 'public title',
          metadata: {},
          content: [],
          date: '2026-06-09T00:00:00.000Z',
          timestamp: Date.parse('2026-06-09T00:00:00.000Z'),
        },
      ],
    },
  }

  const feed = streamFeed(generateStreamAtomFeed(testCtx(), fileData))
  const links = feed.link
  assert.ok(Array.isArray(links))
  assert.ok(links.every(isRecord))
  assert.deepEqual(
    links.map(link => link['@_href']),
    ['https://stream.aarnphm.xyz/', 'https://stream.aarnphm.xyz/'],
  )
  assert.equal(readString(feed, 'id'), 'https://stream.aarnphm.xyz/')
  assert.equal(readString(feed, 'logo'), 'https://stream.aarnphm.xyz/icon.png')
  assert.equal(readString(feed, 'icon'), 'https://stream.aarnphm.xyz/icon.png')

  const [entry] = streamEntries(generateStreamAtomFeed(testCtx(), fileData))
  assert.equal(
    readRecord(entry, 'link')['@_href'],
    'https://stream.aarnphm.xyz/on/2026/06/09?entry=public-entry',
  )
  assert.equal(
    readString(entry, 'id'),
    'https://stream.aarnphm.xyz/on/2026/06/09?entry=public-entry#public-entry',
  )
})
