import rehypeRaw from 'rehype-raw'
import rehypeStringify from 'rehype-stringify'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import { notebookToMarkdownChunks } from '../../util/notebook/markdown'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { isNotebookParseError } from '../../util/notebook/types'

export async function notebookPreviewFixture(
  runtime: boolean,
  description?: string,
): Promise<string> {
  const notebook = parseNotebookDoc(
    JSON.stringify({
      metadata: { language_info: { name: 'python' }, description },
      cells: [
        {
          cell_type: 'markdown',
          source: '# Parsing notebook\n\nBuild parsers from small functions.',
        },
        {
          cell_type: 'markdown',
          source: '## Functional parsing\n\nCombine parsers into a grammar.',
        },
        {
          cell_type: 'code',
          source: 'print(1)',
          outputs: [
            {
              output_type: 'display_data',
              data: { 'text/html': '<h2>Generated output</h2><p>Output is not a summary.</p>' },
            },
          ],
        },
        { cell_type: 'markdown', source: '' },
        { cell_type: 'raw', source: 'Export-only text' },
        { cell_type: 'code', source: '%%javascript\nconsole.log(2)', outputs: [] },
      ],
    }),
    'parsing.ipynb',
  )
  if (isNotebookParseError(notebook)) throw Error(notebook.reason)
  const markdown = notebookToMarkdownChunks(notebook, 'parsing.ipynb', {
    runtime: runtime ? { enabled: true } : false,
  })
    .slice(1)
    .join('\n\n')
  const html = await unified()
    .use(remarkParse)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeStringify)
    .process(markdown)
  return `<body data-slug="notebook"><h1 class="article-title">Parsing notebook</h1><article class="popover-hint notebook-page">${html}</article>`
}
