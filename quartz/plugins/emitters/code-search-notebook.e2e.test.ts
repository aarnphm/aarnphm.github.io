import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { visit } from 'unist-util-visit'
import { isFilePath } from '../../util/path'
import { CodeSearchBlocks } from '../transformers/code-search'
import { SyntaxHighlighting } from '../transformers/syntax'
import { codeSearchTestContext } from './code-search-test-context'

registerHooks({
  load(url, context, nextLoad) {
    if (/\.(scss|inline(?:\.ts)?)$/.test(url))
      return { format: 'module', shortCircuit: true, source: 'export default ""' }
    return nextLoad(url, context)
  },
})
const { parseNotebookPages, resetNotebookCaches, notebookPageCacheEntries } =
  await import('./notebook')
const { ObsidianFlavoredMarkdown } = await import('../transformers/ofm')
const { compactNotebookPluginData } = await import('./notebook-page-cache')

test('notebook rendered anchors agree with source records after cold and warm cache parsing', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'code-notebook-'))
  const fp = `code-search-${path.basename(root)}.ipynb`
  assert.ok(isFilePath(fp))
  try {
    const ctx = codeSearchTestContext(root)
    ctx.cfg.plugins.transformers = [
      CodeSearchBlocks(),
      SyntaxHighlighting(),
      ObsidianFlavoredMarkdown(),
    ]
    ctx.allFiles = [fp]
    await mkdir(ctx.argv.directory)
    await writeFile(
      path.join(ctx.argv.directory, fp),
      JSON.stringify({
        nbformat: 4,
        nbformat_minor: 5,
        metadata: { language_info: { name: 'python' } },
        cells: [
          {
            cell_type: 'markdown',
            metadata: {},
            source: '# Title\n\n```python\nx = 1\n```\n\n```python\nx = 1\n```',
          },
          {
            cell_type: 'code',
            metadata: {},
            source: 'x = 2\r\n\t# 😀  \n',
            execution_count: null,
            outputs: [
              {
                output_type: 'display_data',
                data: { 'text/markdown': '```python\nOUTPUT\n```' },
                metadata: {},
              },
            ],
          },
        ],
      }),
    )
    resetNotebookCaches()
    const cold = await parseNotebookPages(ctx, [fp], 1)
    assert.equal(cold.pages.length, 1)
    const data = cold.pages[0].content[1].data
    assert.equal(data.codeSearchBlocks?.length, 3)
    assert.equal(data.codeSearchBlocks.at(-1)?.text, 'x = 2\r\n\t# 😀  \n')
    const anchors = new Set<string>()
    visit(cold.pages[0].content[0], 'element', node => {
      if (typeof node.properties.id === 'string') anchors.add(node.properties.id)
    })
    for (const record of data.codeSearchBlocks) assert.ok(anchors.has(record.anchor), record.anchor)
    assert.deepEqual(compactNotebookPluginData(data).codeSearchBlocks, data.codeSearchBlocks)
    const warm = await parseNotebookPages(ctx, [fp], 1)
    assert.equal(warm.cacheHits, 1)
    assert.deepEqual(warm.pages[0].content[1].data.codeSearchBlocks, data.codeSearchBlocks)
  } finally {
    const entries = await notebookPageCacheEntries()
    const entry = entries.get(fp)
    if (entry) {
      await rm(entry.blob, { force: true })
      await rm(entry.blob.replace(/\.bin$/, '.meta.json'), { force: true })
      entries.delete(fp)
    }
    await rm(root, { recursive: true, force: true })
  }
})
