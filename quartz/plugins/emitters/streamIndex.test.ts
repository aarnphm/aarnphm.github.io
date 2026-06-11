import { XMLParser } from 'fast-xml-parser'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug } from '../../util/path'
import type { QuartzPluginData } from '../vfile'
import { generateStreamAtomFeed } from '../../util/stream-feed'
import { isRecord, type UnknownRecord } from '../../util/type-guards'

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
