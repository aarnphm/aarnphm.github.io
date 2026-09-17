import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { toHast } from 'mdast-util-to-hast'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { access, mkdtemp, readFile, rm } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { unified } from 'unified'
import type { ArenaBlock, ArenaChannel } from '../plugins/transformers/arena'
import type { QuartzEmitterPluginInstance } from '../types/plugin'
import type { BuildCtx } from './ctx'
import {
  ARENA_CARD_PAGE_SIZE,
  arenaChannelAssets,
  arenaCardPageSource,
  arenaModalSource,
  parseArenaCardPage,
} from '../components/arena/channel-data'
import { defaultProcessedContent, type ProcessedContent } from '../plugins/vfile'
import {
  buildArenaFeedManifest,
  isArenaReadingEntry,
  normalizeArenaFeedUrl,
  orderArenaFeedEntries,
  parseArenaFeedManifest,
} from './arena-feed'
import { isFilePath, isFullSlug } from './path'
import { staticCssBundleKey } from './resource-bundles'

function block(id: string, url: string, changes: Partial<ArenaBlock> = {}): ArenaBlock {
  return {
    id,
    content: id,
    title: id,
    url,
    htmlNode: h('div', [h('a', { href: url }, id)]),
    ...changes,
  }
}

function channel(slug: string, blocks: ArenaBlock[]): ArenaChannel {
  return { id: slug, name: slug, slug, blocks }
}

test('feed URL normalization removes tracking while preserving content and URL distinctions', () => {
  assert.equal(
    normalizeArenaFeedUrl(
      'HTTPS://Example.COM:443/a/../article?b=two%20words&a=%2f&a=3&utm_source=email&curius=1#part-2',
    ),
    'https://example.com/article?b=two%20words&a=%2f&a=3#part-2',
  )
  assert.equal(
    normalizeArenaFeedUrl(
      'https://www.youtube.com/watch?v=AbCdEfGhI12&t=45&list=PL123&s=search&utm_medium=share',
    ),
    'https://www.youtube.com/watch?v=AbCdEfGhI12&t=45&list=PL123&s=search',
  )
  assert.equal(
    normalizeArenaFeedUrl('https://example.com/a?utm%5Fsource=email&gclid=123#intro'),
    'https://example.com/a#intro',
  )
  for (const url of [
    'http://example.com/a',
    'https://example.com/a',
    'https://example.com/a/',
    'https://example.com/a?source=journal',
    'https://example.com/a?ref=appendix',
    'https://example.com/a#one',
    'https://example.com/a#two',
  ]) {
    assert.equal(normalizeArenaFeedUrl(url), url)
  }
  for (const url of [
    'javascript:alert(1)',
    'ftp://example.com/a',
    'https://reader:password@example.com/a',
    'not a url',
  ]) {
    assert.equal(normalizeArenaFeedUrl(url), null)
  }
})

test('article identity survives title edits, channel moves, inserted blocks and tracker changes', async () => {
  const first = await buildArenaFeedManifest(
    [channel('essays', [block('block-1', 'https://example.com/read?curius=1')])],
    'aarnphm.xyz',
  )
  const moved = await buildArenaFeedManifest(
    [
      channel('library', [
        block('block-1', 'https://example.com/new'),
        block('block-8', 'https://example.com/read', { title: 'Edited title' }),
      ]),
    ],
    'aarnphm.xyz',
  )
  const original = first.entries[0]
  const updated = moved.entries.find(entry => entry.sourceUrl === original.sourceUrl)
  assert.ok(updated)
  const hash = createHash('sha256')
    .update('arena-article-v1\0https://example.com/read')
    .digest('hex')
  assert.equal(original.articleId, `article-v1-${hash}`)
  assert.equal(updated.articleId, original.articleId)
  assert.equal(updated.title, 'Edited title')
  assert.notEqual(moved.revision, first.revision)
  assert.equal(updated.occurrences[0].blockId, 'block-8')
  assert.equal(updated.occurrences[0].channelSlug, 'library')
})

test('duplicate saved links share identity, priority, tags and distinct authored notes', async () => {
  const source = 'https://example.com/shared'
  const alpha = channel('alpha', [
    block('a', source, {
      later: false,
      tags: ['systems'],
      metadata: { date: '9/15/2026' },
      subItems: [{ id: 'a-note', content: 'Keep <this> evidence.' }],
    }),
  ])
  alpha.tags = ['reading']
  const beta = channel('beta', [
    block('b', `${source}?utm_campaign=paper`, {
      later: true,
      tags: ['systems', 'design'],
      metadata: { date: '2026-08-02' },
      subItems: [
        {
          id: 'b-note',
          content: 'Second note.',
          htmlNode: h('p', ['Second ', h('strong', 'note'), '.']),
        },
      ],
    }),
  ])
  const snapshot = structuredClone([alpha, beta])
  const manifest = await buildArenaFeedManifest([alpha, beta], 'aarnphm.xyz')
  assert.equal(manifest.entries.length, 1)
  const entry = manifest.entries[0]
  assert.equal(entry.later, true)
  assert.deepEqual(entry.tags, ['design', 'reading', 'systems'])
  assert.equal(entry.savedAt, '2026-08-02')
  assert.deepEqual(
    entry.occurrences.map(occurrence => occurrence.channelSlug),
    ['alpha', 'beta'],
  )
  assert.match(entry.occurrences[0].notesHtml ?? '', /Keep &#x3C;this> evidence\./)
  assert.match(entry.occurrences[1].notesHtml ?? '', /<strong>note<\/strong>/)
  assert.deepEqual([alpha, beta], snapshot)
})

test('nested saved links inherit priority and preserve explicit false through descendants', async () => {
  const root = block('parent', 'https://example.com/parent', {
    later: true,
    subItems: [
      block('inherited', 'https://example.com/inherited'),
      block('false-child', 'https://example.com/false', {
        metadata: { later: 'no' },
        subItems: [block('grandchild', 'https://example.com/grandchild')],
      }),
      block('explicit-child', 'https://example.com/explicit', { later: false }),
    ],
  })
  const manifest = await buildArenaFeedManifest([channel('saved', [root])], 'aarnphm.xyz')
  const entries = new Map(manifest.entries.map(entry => [entry.sourceUrl, entry]))
  assert.equal(entries.get('https://example.com/parent')?.later, true)
  assert.equal(entries.get('https://example.com/inherited')?.later, true)
  assert.equal(entries.get('https://example.com/false')?.later, false)
  assert.equal(entries.get('https://example.com/grandchild')?.later, false)
  assert.equal(entries.get('https://example.com/explicit')?.later, false)
  assert.equal(entries.get('https://example.com/inherited')?.occurrences[0].parentBlockId, 'parent')
  assert.equal(
    entries.get('https://example.com/grandchild')?.occurrences[0].parentBlockId,
    'false-child',
  )
})

test('incidental note links stay with their parent while leading saved links enter the feed', async () => {
  const incidental = block('note', 'https://example.com/citation', {
    content: 'Compare this citation.',
    htmlNode: h('p', [
      'Compare ',
      h('a', { href: 'https://example.com/citation' }, 'this citation'),
      '.',
    ]),
    subItems: [block('nested-saved', 'https://example.com/nested', { metadata: { later: 'yes' } })],
  })
  const root = block('parent', 'https://example.com/article', { subItems: [incidental] })
  const manifest = await buildArenaFeedManifest([channel('saved', [root])], 'aarnphm.xyz')
  assert.equal(manifest.entries.length, 2)
  assert.ok(!manifest.entries.some(entry => entry.sourceUrl === 'https://example.com/citation'))
  const parent = manifest.entries.find(entry => entry.sourceUrl === root.url)
  assert.match(
    parent?.occurrences[0].notesHtml ?? '',
    /Compare <a href="https:\/\/example.com\/citation">this citation<\/a>/,
  )
  const nested = manifest.entries.find(entry => entry.sourceUrl === 'https://example.com/nested')
  assert.equal(nested?.later, true)
  assert.equal(nested?.occurrences[0].parentBlockId, 'note')
})

test('a parser URL inherited from a descendant does not turn unlinked parent prose into an article', async () => {
  const note = block('prose', 'https://example.com/child', {
    content: 'An idea without a saved source',
    htmlNode: h('p', 'An idea without a saved source'),
    subItems: [block('child', 'https://example.com/child')],
  })
  const manifest = await buildArenaFeedManifest([channel('ideas', [note])], 'aarnphm.xyz')
  assert.equal(manifest.entries.length, 1)
  assert.deepEqual(
    manifest.entries[0].occurrences.map(occurrence => occurrence.blockId),
    ['child'],
  )
})

test('kind hints preserve internal anchors and identify supported PDFs and videos', async () => {
  const blocks = [
    block('internal', '/thoughts/example#section', {
      url: undefined,
      internalSlug: 'thoughts/example',
      internalHref: '/thoughts/example#section',
    }),
    block('pdf', 'https://example.com/PAPER.PDF?download=1#page=4'),
    block('arxiv', 'https://arxiv.org/pdf/2609.00001'),
    block('arxiv-abstract', 'https://arxiv.org/abs/2206.00759v3'),
    block('arxiv-html', 'https://arxiv.org/html/2206.00759v3'),
    block('arxiv-list', 'https://arxiv.org/list/cs.LG/recent'),
    block('video', 'https://www.youtube.com/watch?v=AbCdEfGhI12&t=45'),
    block('vimeo', 'https://vimeo.com/12345'),
    block('spoof', 'https://youtube.com.example.com/watch?v=AbCdEfGhI12'),
  ]
  const manifest = await buildArenaFeedManifest([channel('library', blocks)], 'https://aarnphm.xyz')
  const kinds = new Map(manifest.entries.map(entry => [entry.occurrences[0].blockId, entry.kind]))
  assert.deepEqual(
    kinds,
    new Map([
      ['internal', 'internal'],
      ['pdf', 'pdf'],
      ['arxiv', 'pdf'],
      ['arxiv-abstract', 'pdf'],
      ['arxiv-html', 'pdf'],
      ['arxiv-list', 'html'],
      ['video', 'video'],
      ['vimeo', 'video'],
      ['spoof', 'html'],
    ]),
  )
  assert.equal(
    manifest.entries.find(entry => entry.kind === 'internal')?.sourceUrl,
    'https://aarnphm.xyz/thoughts/example#section',
  )
  const paper = manifest.entries.find(entry => entry.occurrences[0].blockId === 'arxiv-abstract')
  assert.ok(paper)
  assert.equal(paper.sourceUrl, 'https://arxiv.org/abs/2206.00759v3')
  assert.equal(
    paper.articleId,
    `article-v1-${createHash('sha256').update(`arena-article-v1\0${paper.sourceUrl}`).digest('hex')}`,
  )
})

test('reading eligibility excludes video sources while keeping articles about video', async () => {
  const videoUrls = [
    'https://www.youtube.com/watch?v=AbCdEfGhI12',
    'https://m.youtube.com/shorts/AbCdEfGhI12',
    'https://youtube.com/live/AbCdEfGhI12',
    'https://youtube.com/playlist?list=PL123',
    'https://youtube.com/@creator',
    'https://www.youtube-nocookie.com/embed/AbCdEfGhI12',
    'https://youtu.be/AbCdEfGhI12?t=42',
    'https://www.youtube.com87/watch?v=qX6NztnPU-4',
    'https://vimeo.com/12345',
    'https://player.vimeo.com/video/12345',
    'https://example.com/film.MP4?download=1#t=30',
    'https://example.com/film.webm',
    'https://example.com/stream.m3u8',
    'https://aarnphm.xyz/film.mov',
  ]
  const articleUrls = [
    'https://example.com/how-video-works',
    'https://example.com/article?watch=https://youtube.com/watch?v=AbCdEfGhI12',
    'https://youtube.com.example.com/about',
    'https://example.com/paper.pdf',
    'https://arxiv.org/abs/2206.00759',
    'https://aarnphm.xyz/thoughts/video',
  ]
  const manifest = await buildArenaFeedManifest(
    [
      channel('saved', [
        ...videoUrls.map((url, index) => block(`video-${index}`, url)),
        ...articleUrls.map((url, index) => block(`article-${index}`, url, { tags: ['video'] })),
      ]),
      channel('video', [block('video-channel', 'https://example.com/recorded-talk')]),
    ],
    'aarnphm.xyz',
  )
  const before = structuredClone(manifest)
  assert.deepEqual(
    manifest.entries
      .filter(isArenaReadingEntry)
      .map(entry => entry.sourceUrl)
      .sort(),
    [...articleUrls].sort(),
  )
  assert.equal(isArenaReadingEntry({ ...manifest.entries[0], kind: 'video' }), false)
  assert.deepEqual(manifest, before)
})

test('manifest revision reflects note edits and removed links without changing remaining identities', async () => {
  const kept = block('kept', 'https://example.com/kept')
  const removed = block('removed', 'https://example.com/removed')
  const first = await buildArenaFeedManifest([channel('saved', [kept, removed])], 'aarnphm.xyz')
  const repeated = await buildArenaFeedManifest([channel('saved', [kept, removed])], 'aarnphm.xyz')
  assert.equal(first.revision, repeated.revision)
  const edited = await buildArenaFeedManifest(
    [channel('saved', [{ ...kept, subItems: [{ id: 'new-note', content: 'New evidence' }] }])],
    'aarnphm.xyz',
  )
  assert.equal(edited.entries.length, 1)
  assert.notEqual(edited.revision, first.revision)
  assert.equal(
    edited.entries[0].articleId,
    first.entries.find(entry => entry.sourceUrl === kept.url)?.articleId,
  )
})

test('invalid dates and unsupported URLs do not create misleading feed values', async () => {
  const manifest = await buildArenaFeedManifest(
    [
      channel('saved', [
        block('invalid-date', 'https://example.com/a', { metadata: { date: '02/30/2026' } }),
        block('leap-date', 'https://example.com/b', { metadata: { date: '2024-02-29' } }),
        block('invalid-url', 'javascript:alert(1)'),
      ]),
    ],
    'aarnphm.xyz',
  )
  assert.equal(manifest.entries.length, 2)
  assert.equal(manifest.entries.find(entry => entry.sourceUrl.endsWith('/a'))?.savedAt, null)
  assert.equal(
    manifest.entries.find(entry => entry.sourceUrl.endsWith('/b'))?.savedAt,
    '2024-02-29',
  )
})

test('manifest validation rejects malformed, duplicate and unsupported records', async () => {
  const valid = await buildArenaFeedManifest(
    [channel('saved', [block('one', 'https://example.com/one')])],
    'aarnphm.xyz',
  )
  const serialized: unknown = JSON.parse(JSON.stringify(valid))
  assert.deepEqual(parseArenaFeedManifest(serialized), valid)
  const entry = valid.entries[0]
  for (const bad of [
    null,
    { ...valid, schemaVersion: 2 },
    { ...valid, revision: 'unknown' },
    { ...valid, entries: [entry, entry] },
    { ...valid, entries: [{ ...entry, sourceUrl: 'javascript:alert(1)' }] },
    { ...valid, entries: [{ ...entry, later: 'true' }] },
    { ...valid, entries: [{ ...entry, kind: 'unknown' }] },
    { ...valid, entries: [{ ...entry, occurrences: [] }] },
    {
      ...valid,
      entries: [{ ...entry, occurrences: [{ ...entry.occurrences[0], parentBlockId: 4 }] }],
    },
    { ...valid, entries: [{ ...entry, savedAt: '2026-02-30' }] },
    { ...valid, entries: [{ ...entry, articleId: 'block-1' }] },
    { ...valid, entries: [{ ...entry, tags: [true] }] },
  ])
    assert.equal(parseArenaFeedManifest(bad), null)
})

test('stable shuffle visits Later first and keeps survivor order after read filtering', async () => {
  const blocks = Array.from({ length: 24 }, (_, index) =>
    block(`block-${index}`, `https://example.com/${index}`, { later: index % 2 === 0 }),
  )
  const manifest = await buildArenaFeedManifest([channel('saved', blocks)], 'aarnphm.xyz')
  const before = [...manifest.entries]
  const first = orderArenaFeedEntries(manifest.entries, 'pass-one')
  assert.deepEqual(first, orderArenaFeedEntries([...manifest.entries].reverse(), 'pass-one'))
  assert.deepEqual(
    first.map(entry => entry.later),
    [...Array.from({ length: 12 }, () => true), ...Array.from({ length: 12 }, () => false)],
  )
  const read = new Set([first[2].articleId, first[10].articleId, first[18].articleId])
  assert.deepEqual(
    orderArenaFeedEntries(
      manifest.entries.filter(entry => !read.has(entry.articleId)),
      'pass-one',
    ),
    first.filter(entry => !read.has(entry.articleId)),
  )
  assert.notDeepEqual(first, orderArenaFeedEntries(manifest.entries, 'pass-two'))
  assert.deepEqual(manifest.entries, before)
})

test('feed route is reserved before any manifest is emitted, including empty catalogues', async () => {
  await assert.rejects(
    buildArenaFeedManifest([channel('feed', [])], 'aarnphm.xyz'),
    /reserved reader route \/arena\/feed/,
  )
  const empty = await buildArenaFeedManifest([], 'aarnphm.xyz')
  assert.deepEqual(empty.entries, [])
  assert.deepEqual(parseArenaFeedManifest(empty), empty)
})

test('real Arena parsing emits the reader shell and refreshes its catalogue on partial changes', async t => {
  const assets = registerHooks({
    resolve(specifier, context, next) {
      return next(specifier.endsWith('.inline') ? `${specifier}.ts` : specifier, context)
    },
    load(url, context, next) {
      if (/\.(?:scss|css|inline\.ts)$/.test(url)) {
        return {
          format: 'module',
          source: `export default ${JSON.stringify(readFileSync(new URL(url), 'utf8'))}`,
          shortCircuit: true,
        }
      }
      return next(url, context)
    },
  })
  t.after(() => assets.deregister())
  const { Arena } = await import('../plugins/transformers/arena')
  const { ArenaPage } = await import('../plugins/emitters/arenaPage')
  const { default: collapseHeaderStyle } =
    await import('../components/styles/collapseHeader.inline.scss')
  const output = await mkdtemp(join(tmpdir(), 'quartz-arena-feed-'))
  t.after(() => rm(output, { recursive: true, force: true }))
  const colors = {
    light: '#fff',
    lightgray: '#eee',
    gray: '#999',
    darkgray: '#555',
    dark: '#111',
    secondary: '#123',
    tertiary: '#456',
    highlight: '#def',
    textHighlight: '#fed',
  }
  const emitter = ArenaPage()
  const ctx: BuildCtx = {
    buildId: 'arena-feed-fixture',
    argv: {
      directory: 'content',
      output,
      verbose: false,
      serve: false,
      watch: false,
      port: 0,
      wsPort: 0,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'Arena fixture',
        enableSPA: false,
        enablePopovers: false,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'created',
        locale: 'en-US',
        baseUrl: 'aarnphm.xyz',
        theme: {
          typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
          cdnCaching: false,
          colors: { lightMode: colors, darkMode: colors },
          fontOrigin: 'local',
        },
      },
      plugins: { transformers: [], filters: [], emitters: [emitter] },
    },
    allFiles: [],
    allSlugs: [],
    incremental: true,
    extractedStaticResources: new Map([
      [staticCssBundleKey(collapseHeaderStyle), 'static/collapse.css'],
    ]),
  }
  const slug = 'are.na'
  const filePath = 'content/are.na.md'
  assert.ok(isFullSlug(slug))
  assert.ok(isFilePath(filePath))
  const parse = async (markdown: string): Promise<ProcessedContent> => {
    const [, file] = defaultProcessedContent({
      slug,
      filePath,
      frontmatter: { title: 'Saved links', pageLayout: 'default', tags: [] },
    })
    const tree = toHast(fromMarkdown(markdown))
    assert.ok(tree)
    assert.ok(tree.type === 'root')
    const plugins = Arena().htmlPlugins?.(ctx)
    assert.ok(plugins)
    await unified().use(plugins).run(tree, file)
    return [tree, file]
  }
  const collect = async (
    result: ReturnType<QuartzEmitterPluginInstance['emit']> | null | undefined,
  ) => {
    const paths: string[] = []
    for await (const path of (await result) ?? []) paths.push(path)
    return paths
  }
  const readManifest = async () => {
    const data: unknown = JSON.parse(await readFile(join(output, 'static/arena-feed.json'), 'utf8'))
    const manifest = parseArenaFeedManifest(data)
    assert.ok(manifest)
    return manifest
  }
  const resources = { css: [], js: [], additionalHead: [] }
  await t.test(
    'virtual channel cards support random access and preserve separate modal content',
    async () => {
      const fixture = await parse(
        `## lazy\n\n${['pinned', 'later', 'blocks']
          .map(section =>
            Array.from(
              { length: section === 'blocks' ? 55 : 30 },
              (_, index) =>
                `- [${section} ${index}](https://example.com/${section}-${index}.pdf)\n  - [meta]:\n    - date: 09/15/2026\n${section === 'blocks' ? '' : `    - ${section}: true\n`}  - **Delayed note ${section} ${index}**\n`,
            ).join('\n'),
          )
          .join('\n')}`,
      )
      const channels = fixture[1].data.arenaData?.channels
      assert.ok(channels)
      const channel = channels[0]
      const lazyEmitter = ArenaPage()
      await collect(lazyEmitter.emit(ctx, [fixture], resources))
      const html = await readFile(join(output, 'arena/lazy.html'), 'utf8')
      const initialIds = Array.from(html.matchAll(/data-block-id="([^"]+)"/g), match => match[1])
      assert.equal(initialIds.length, 3 * ARENA_CARD_PAGE_SIZE)
      assert.equal(new Set(initialIds).size, initialIds.length)
      assert.doesNotMatch(html, /arena-block-modal-data|Delayed note|data-arena-pdf-src/)
      assert.doesNotMatch(html, /arena-load-more/)
      assert.match(html, /data-arena-count="55"/)
      const markdown = await readFile(join(output, 'arena/lazy.md'), 'utf8')
      assert.match(markdown, /# lazy/)
      assert.match(markdown, /\*\*Delayed note pinned 29\*\*/)
      assert.match(markdown, /\*\*Delayed note later 29\*\*/)
      assert.match(markdown, /\*\*Delayed note blocks 54\*\*/)
      assert.match(markdown, /https:\/\/example.com\/blocks-54.pdf/)
      assert.match(markdown, /pinned: true/)
      assert.match(markdown, /later: true/)
      const assets = arenaChannelAssets(channel.slug)
      const allIds: string[] = []
      for (const section of ['pinned', 'later', 'blocks'] as const) {
        const total = section === 'blocks' ? 55 : 30
        for (let offset = 0; offset < total; offset += ARENA_CARD_PAGE_SIZE) {
          const source = arenaCardPageSource(assets, section, offset)
          const page = parseArenaCardPage(
            JSON.parse(await readFile(join(output, source.slice(1)), 'utf8')),
          )
          const ids = Array.from(page.html.matchAll(/data-block-id="([^"]+)"/g), match => match[1])
          assert.equal(page.total, total)
          assert.equal(page.offset, offset)
          assert.equal(ids.length, Math.min(ARENA_CARD_PAGE_SIZE, total - offset))
          assert.doesNotMatch(page.html, /arena-block-modal-data|Delayed note/)
          allIds.push(...ids)
        }
      }
      assert.equal(allIds.length, channel.blocks.length)
      assert.deepEqual(new Set(allIds), new Set(channel.blocks.map(block => block.id)))
      const last = channel.blocks.at(-1)
      assert.ok(last)
      const modal = await readFile(join(output, arenaModalSource(assets, last.id).slice(1)), 'utf8')
      assert.match(modal, /Delayed note blocks 54/)
      assert.match(modal, /data-arena-pdf-src=/)
      assert.doesNotMatch(modal, /data-pdf-src=/)
      const search = JSON.parse(
        await readFile(join(output, assets.slice(1), 'search.json'), 'utf8'),
      )
      assert.equal(search.blocks.length, channel.blocks.length)
      assert.equal(search.blocks.at(-1).id, last.id)
      assert.ok(search.blocks.every((block: { hasModalInDom: boolean }) => !block.hasModalInDom))
      await collect(
        lazyEmitter.partialEmit?.(ctx, [], resources, [
          { type: 'delete', path: filePath, previousFile: fixture[1] },
        ]),
      )
      await assert.rejects(access(join(output, assets.slice(1))), { code: 'ENOENT' })
      await assert.rejects(access(join(output, 'arena/lazy.md')), { code: 'ENOENT' })
    },
  )
  await t.test('watch builds emit a separate Markdown source for every channel', async () => {
    const fixture = await parse(`## first

- [First article](https://example.com/first)
  - **First channel note.**

## second

- [Second article](https://example.com/second)
  - Second channel note.
`)
    const watchCtx = { ...ctx, argv: { ...ctx.argv, watch: true } }
    const sourceEmitter = ArenaPage()
    const paths = await collect(sourceEmitter.emit(watchCtx, [fixture], resources))
    for (const name of ['first', 'second']) {
      assert.ok(paths.includes(join(output, `arena/${name}.md`)))
      const source = await readFile(join(output, `arena/${name}.md`), 'utf8')
      assert.match(source, new RegExp(`slug: arena/${name}`))
      assert.match(source, new RegExp(`https://example.com/${name}`))
      assert.doesNotMatch(
        source,
        new RegExp(`${name === 'first' ? 'Second' : 'First'} channel note`),
      )
      const html = await readFile(join(output, `arena/${name}.html`), 'utf8')
      assert.match(html, new RegExp(`href="[^"]*/${name}\\.md"[^>]*class="llm-source"`))
    }
    const remaining = await parse('## second\n\n- [Second article](https://example.com/second)\n')
    await collect(
      sourceEmitter.partialEmit?.(watchCtx, [remaining], resources, [
        { type: 'change', path: filePath, file: remaining[1], previousFile: fixture[1] },
      ]),
    )
    await assert.rejects(access(join(output, 'arena/first.md')), { code: 'ENOENT' })
    await access(join(output, 'arena/second.md'))
  })
  await t.test(
    'unlocked links use Wayback consistently across blocks, anchors, and feed entries',
    async () => {
      const original = 'https://example.com/article?edition=1#section'
      const archived = `https://web.archive.org/web/*/${original}`
      const snapshot = `https://web.archive.org/web/20260901000000/${original}`
      const wrapped = `https://removepaywalls.com/${original}`
      for (const [url, unlocked, expected] of [
        [original, 'true', archived],
        [original, 'YES', archived],
        [original, 'false', original],
        [original, 'no', original],
        [original, undefined, original],
        [snapshot, 'true', snapshot],
        [archived, 'true', archived],
        [wrapped, 'true', archived],
        [wrapped, undefined, archived],
      ]) {
        const parsed = await parse(`## saved

- [Article](<${url}>)
  - [meta]:
    - later: true
${unlocked === undefined ? '' : `    - unlocked: ${unlocked}\n`}  - Keep [this citation](https://example.com/citation).
  - [Nested article](<${original}>)
    - [meta]:
      - unlocked: true
`)
        const parent = parsed[1].data.arenaData?.channels[0].blocks[0]
        assert.ok(parent)
        assert.equal(parent.url, expected)
        assert.equal(parent.metadata?.unlocked, undefined)
        for (const node of [parent.htmlNode, parent.titleHtmlNode]) {
          assert.ok(node)
          assert.ok(toHtml(node).includes(`href="${expected}"`))
        }
        const nested = parent.subItems?.find(item => item.title === 'Nested article')
        assert.ok(nested)
        assert.equal(nested.url, archived)
        assert.ok(nested.htmlNode)
        assert.ok(toHtml(nested.htmlNode).includes(`href="${archived}"`))
        const channels = parsed[1].data.arenaData?.channels
        assert.ok(channels)
        const manifest = await buildArenaFeedManifest(channels, 'aarnphm.xyz')
        const entry = manifest.entries.find(item => item.title === 'Article')
        assert.ok(entry)
        assert.equal(entry.sourceUrl, expected)
        assert.ok(entry.occurrences[0].notesHtml?.includes('href="https://example.com/citation"'))
      }
    },
  )
  const first = await parse(`## saved

- <https://example.com/parent> -- Parent
  - [meta]:
    - later: true
    - date: 09/15/2026
  - Keep [this citation](https://example.com/incidental) as context.
  - <https://example.com/child> -- Child
    - [meta]:
      - later: false
    - Child note.
`)
  const initialPaths = await collect(emitter.emit(ctx, [first], resources))
  assert.ok(initialPaths.includes(join(output, 'arena/feed.html')))
  assert.ok(initialPaths.includes(join(output, 'static/arena-feed.json')))
  assert.ok(initialPaths.includes(join(output, 'arena/saved.md')))
  const source = await readFile(join(output, 'arena/saved.md'), 'utf8')
  assert.match(source, /# saved/)
  assert.match(source, /\[this citation\]\(https:\/\/example.com\/incidental\)/)
  assert.match(source, /later: false/)
  assert.equal(source.split('Child note.').length - 1, 1)
  const shell = await readFile(join(output, 'arena/feed.html'), 'utf8')
  assert.match(shell, /data-arena-feed/)
  assert.match(shell, /<a[^>]*href="\/arena"[^>]*>arena<\/a>/)
  assert.match(shell, /<span>reader<\/span>/)
  assert.doesNotMatch(shell, /Keep.*this citation/)
  const initial = await readManifest()
  assert.equal(initial.entries.length, 2)
  assert.equal(initial.entries.find(entry => entry.sourceUrl.endsWith('/parent'))?.later, true)
  assert.equal(initial.entries.find(entry => entry.sourceUrl.endsWith('/child'))?.later, false)
  assert.equal(initial.entries.find(entry => entry.sourceUrl.endsWith('/child'))?.title, 'Child')
  assert.match(
    initial.entries.find(entry => entry.sourceUrl.endsWith('/parent'))?.occurrences[0].notesHtml ??
      '',
    /this citation/,
  )

  const updated = await parse(`## saved

- <https://example.com/parent> -- Updated parent
  - [meta]:
    - later: true
  - Fresh evidence.
`)
  const changedPaths = await collect(
    emitter.partialEmit?.(ctx, [updated], resources, [
      { type: 'change', path: filePath, file: updated[1], previousFile: first[1] },
    ]),
  )
  assert.ok(changedPaths.includes(join(output, 'arena/feed.html')))
  assert.ok(changedPaths.includes(join(output, 'static/arena-feed.json')))
  assert.ok(changedPaths.includes(join(output, 'arena/saved.md')))
  const updatedSource = await readFile(join(output, 'arena/saved.md'), 'utf8')
  assert.match(updatedSource, /Updated parent/)
  assert.match(updatedSource, /Fresh evidence/)
  assert.doesNotMatch(updatedSource, /Child note/)
  const changed = await readManifest()
  assert.equal(changed.entries.length, 1)
  assert.equal(
    changed.entries[0].articleId,
    initial.entries.find(entry => entry.sourceUrl.endsWith('/parent'))?.articleId,
  )
  assert.notEqual(changed.revision, initial.revision)
  assert.equal(changed.entries[0].title, 'Updated parent')

  const reserved = await parse('## feed\n\n- <https://example.com/conflict> -- Conflict\n')
  await assert.rejects(collect(emitter.emit(ctx, [reserved], resources)), /reserved reader route/)
  assert.deepEqual(await readManifest(), changed)

  await collect(
    emitter.partialEmit?.(ctx, [], resources, [
      { type: 'delete', path: filePath, previousFile: updated[1] },
    ]),
  )
  for (const path of [
    'arena/feed.html',
    'static/arena-feed.json',
    'arena/saved.html',
    'arena/saved.md',
    'arena.html',
    'static/arena-search.json',
  ]) {
    await assert.rejects(access(join(output, path)), { code: 'ENOENT' })
  }
})
