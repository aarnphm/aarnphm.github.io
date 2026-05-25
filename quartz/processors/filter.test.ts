import assert from 'node:assert/strict'
import test from 'node:test'
import type { ProcessedContent } from '../plugins/vfile'
import type { BuildCtx } from '../util/ctx'
import type { FilePath, FullSlug } from '../util/path'
import { RemoveDrafts } from '../plugins/filters'
import { defaultProcessedContent } from '../plugins/vfile'
import { filterContentResult } from './filter'

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
      plugins: { transformers: [], filters: [RemoveDrafts()], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

function content(slug: string, draft: boolean): ProcessedContent {
  return defaultProcessedContent({
    slug: slug as FullSlug,
    relativePath: `${slug}.md` as FilePath,
    frontmatter: { title: slug, pageLayout: 'default', draft },
  })
}

const slugs = (content: ProcessedContent[]): string[] =>
  content.map(([, file]) => String(file.data.slug)).sort()

test('filter content result returns published and removed files', () => {
  const published = content('published', false)
  const draft = content('draft', true)

  const result = filterContentResult(testCtx(), [published, draft])

  assert.deepEqual(slugs(result.published), ['published'])
  assert.deepEqual(slugs(result.removed), ['draft'])
})
