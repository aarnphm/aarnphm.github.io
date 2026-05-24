import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug, SimpleSlug } from '../../util/path'
import type { StaticResources } from '../../util/resources'
import { defaultProcessedContent } from '../vfile'
import { ContentIndex } from './contentIndex'

function testCtx(root: string): BuildCtx {
  const contentDir = path.join(root, 'content')
  return {
    buildId: 'test',
    argv: {
      directory: contentDir,
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
    allSlugs: ['bases/ideas' as FullSlug, 'thoughts/lecture/notebook' as FullSlug],
    allFiles: ['bases/ideas.base' as FilePath, 'thoughts/lecture/notebook.ipynb' as FilePath],
    incremental: false,
  }
}

test('content index includes notebook and base file graph pages', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-content-index-'))
  try {
    const ctx = testCtx(root)
    const baseFilePath = path.join(ctx.argv.directory, 'bases/ideas.base') as FilePath
    const baseContent = defaultProcessedContent({
      slug: 'bases/ideas' as FullSlug,
      filePath: baseFilePath,
      relativePath: 'bases/ideas.base' as FilePath,
      frontmatter: { title: 'parsed ideas base', pageLayout: 'default', tags: ['bases'] },
      text: '',
      links: ['thoughts/lecture/notebook' as SimpleSlug],
    })

    const plugin = ContentIndex({ enableAtom: false, enableSiteMap: false, enableSecurity: false })
    const resources: StaticResources = { css: [], js: [], additionalHead: [] }

    const emitted: FilePath[] = []
    const result = plugin.emit(ctx, [baseContent], resources)
    if (Symbol.asyncIterator in result) {
      for await (const file of result) {
        emitted.push(file)
      }
    } else {
      emitted.push(...(await result))
    }
    assert.ok(emitted.some(file => file.endsWith('static/contentIndex.json')))

    const raw = await readFile(path.join(ctx.argv.output, 'static/contentIndex.json'), 'utf8')
    const index = JSON.parse(raw) as Record<string, { title: string; fileName: string }>

    assert.equal(index['bases/ideas']?.title, 'parsed ideas base')
    assert.equal(index['bases/ideas']?.fileName, 'bases/ideas.base')
    assert.equal(index['thoughts/lecture/notebook']?.title, 'notebook')
    assert.equal(index['thoughts/lecture/notebook']?.fileName, 'thoughts/lecture/notebook.ipynb')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
