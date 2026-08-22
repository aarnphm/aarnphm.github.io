import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug } from '../../util/path'
import { defaultProcessedContent } from '../vfile'
import { AliasRedirects, aliasRedirectRules } from './aliases'

function testCtx(root: string): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: path.join(root, 'content'),
      verbose: false,
      output: path.join(root, 'public'),
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

async function collectEmitted(
  emitted: Promise<FilePath[]> | AsyncGenerator<FilePath> | null,
): Promise<FilePath[]> {
  const result = await emitted
  if (!result) return []
  if (!(Symbol.asyncIterator in result)) {
    return result
  }
  const files: FilePath[] = []
  for await (const file of result) {
    files.push(file)
  }
  return files
}

test('emits HTML and Markdown redirects for aliases and permalinks', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-aliases-markdown-'))
  try {
    const ctx = testCtx(root)
    const index = defaultProcessedContent({
      slug: 'index' as FullSlug,
      filePath: path.join(ctx.argv.directory, 'index.md') as FilePath,
      relativePath: 'index.md' as FilePath,
      frontmatter: { title: 'home', pageLayout: 'default', tags: [], permalinks: ['/about'] },
      text: '',
      links: [],
      aliases: ['about' as FullSlug],
    })
    const privacy = defaultProcessedContent({
      slug: 'privacy-policy' as FullSlug,
      filePath: path.join(ctx.argv.directory, 'privacy policy.md') as FilePath,
      relativePath: 'privacy policy.md' as FilePath,
      frontmatter: {
        title: 'privacy policy',
        pageLayout: 'default',
        tags: [],
        permalinks: ['/privacy'],
      },
      text: '',
      links: [],
      aliases: ['privacy' as FullSlug],
    })
    const content = [index, privacy]

    assert.equal(
      aliasRedirectRules(content),
      [
        '/about / 308',
        '/about.md /llms.txt 308',
        '/privacy /privacy-policy 308',
        '/privacy.md /privacy-policy.md 308',
      ].join('\n'),
    )

    const emitted = await collectEmitted(
      AliasRedirects().emit(ctx, content, { css: [], js: [], additionalHead: [] }),
    )
    assert.ok(emitted.some(file => file.endsWith('/_redirects')))
    assert.ok(emitted.some(file => file.endsWith('/about.html')))
    assert.ok(emitted.some(file => file.endsWith('/privacy.html')))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('rejects aliases that point to different canonical notes', () => {
  const first = defaultProcessedContent({
    slug: 'first' as FullSlug,
    aliases: ['shared' as FullSlug],
  })
  const second = defaultProcessedContent({
    slug: 'second' as FullSlug,
    aliases: ['shared' as FullSlug],
  })

  assert.throws(
    () => aliasRedirectRules([first, second]),
    /conflicting alias redirect for \/shared/,
  )
})

test('enforces Cloudflare static redirect limits', () => {
  const content = defaultProcessedContent({
    slug: 'canonical' as FullSlug,
    aliases: Array.from({ length: 1001 }, (_, index) => `alias-${index}` as FullSlug),
  })

  assert.throws(
    () => aliasRedirectRules([content]),
    /static alias redirects exceed the 2,000 rule limit: 2002/,
  )
})

test('alias redirects skip body-only changes with unchanged aliases', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-aliases-unchanged-'))
  try {
    const ctx = testCtx(root)
    const filePath = path.join(ctx.argv.directory, 'thoughts/note.md') as FilePath
    const content = defaultProcessedContent({
      slug: 'thoughts/note' as FullSlug,
      filePath,
      relativePath: 'thoughts/note.md' as FilePath,
      frontmatter: { title: 'note', pageLayout: 'default', tags: [], aliases: ['old-note'] },
      text: 'new body',
      links: [],
    })
    const previous = defaultProcessedContent({
      slug: 'thoughts/note' as FullSlug,
      filePath,
      relativePath: 'thoughts/note.md' as FilePath,
      frontmatter: { title: 'note', pageLayout: 'default', tags: [], aliases: ['old-note'] },
      text: 'old body',
      links: [],
    })
    const plugin = AliasRedirects()

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], { css: [], js: [], additionalHead: [] }, [
        {
          type: 'change',
          path: 'thoughts/note.md' as FilePath,
          file: content[1],
          previousFile: previous[1],
        },
      ]) ?? null,
    )

    assert.deepEqual(emitted, [])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
