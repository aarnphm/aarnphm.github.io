import assert from 'node:assert'
import test, { describe } from 'node:test'
import rehypeRaw from 'rehype-raw'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import {
  notebookRuntimeData,
  notebookTitle,
  notebookToMarkdown,
  notebookToMarkdownChunks,
  parseNotebook,
} from './notebook'

type HastElement = {
  type: string
  tagName?: string
  properties?: Record<string, unknown>
  children?: HastElement[]
}

function findElement(
  node: HastElement,
  predicate: (node: HastElement) => boolean,
): HastElement | undefined {
  if (predicate(node)) return node
  for (const child of node.children ?? []) {
    const found = findElement(child, predicate)
    if (found) return found
  }
}

describe('notebook parser', () => {
  test('converts markdown and code cells into quartz markdown', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [
          { cell_type: 'markdown', source: ['### Random Number Generator\n', 'body'] },
          {
            cell_type: 'code',
            source: ['print("hi")\n'],
            outputs: [{ output_type: 'stream', name: 'stdout', text: ['hi\n'] }],
          },
        ],
      }),
      'lecture.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'lecture.ipynb')

    assert.strictEqual(notebookTitle(notebook, 'lecture.ipynb'), 'Random Number Generator')
    assert.match(markdown, /title: "Random Number Generator"/)
    assert.match(markdown, /collapseHeadings: false/)
    assert.match(markdown, /\nbody\n/)
    assert.match(markdown, /notebook-markdown-cell-boundary/)
    assert.doesNotMatch(markdown, /### Random Number Generator/)
    assert.match(markdown, /```python\nprint\("hi"\)\n```/)
    assert.match(
      markdown,
      /class="notebook-output notebook-output-stream notebook-output-stream-stdout"/,
    )
    assert.match(markdown, /data-output-name="stdout"><samp>hi<\/samp><\/pre>/)
    assert.doesNotMatch(markdown, /```text\nhi\n```/)
  })

  test('uses the notebook filename title fallback and preserves rich outputs', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'code',
            source: 'x',
            outputs: [
              { output_type: 'display_data', data: { 'text/html': '<strong>ok</strong>' } },
            ],
          },
        ],
      }),
      '01 Copy Language.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, '01 Copy Language.ipynb')

    assert.match(markdown, /title: "01 Copy Language"/)
    assert.match(markdown, /<div class="notebook-output notebook-output-html">/)
    assert.match(markdown, /<strong>ok<\/strong>/)
  })

  test('resolves markdown attachment image paths into data URLs', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'markdown',
            source:
              '<img src="attachment:diagram%20one.png" alt="diagram"> ![plot](attachment:plot.png)',
            attachments: {
              'diagram one.png': { 'image/png': 'a b\nc' },
              'plot.png': { 'image/svg+xml': '<svg></svg>' },
            },
          },
        ],
      }),
      'attachments.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'attachments.ipynb')

    assert.match(markdown, /src="data:image\/png;base64,abc"/)
    assert.match(markdown, /!\[plot\]\(data:image\/svg\+xml,%3Csvg%3E%3C%2Fsvg%3E\)/)
    assert.doesNotMatch(markdown, /attachment:/)
  })

  test('resolves notebook-relative markdown image paths against the notebook directory', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'markdown',
            source:
              '<img src="./img/Task5.svg"> ![phase](img/Phases.svg) <img src="https://example.com/remote.svg"> <img src="/static/icon.png">',
          },
        ],
      }),
      'thoughts/university/twenty-five-twenty-six/sfwr-4tb3/00 Notebooks on Compiler Construction/Notebooks on Compiler Construction.ipynb',
    )

    const markdown = notebookToMarkdown(
      notebook,
      'thoughts/university/twenty-five-twenty-six/sfwr-4tb3/00 Notebooks on Compiler Construction/Notebooks on Compiler Construction.ipynb',
    )

    assert.match(
      markdown,
      /src="thoughts\/university\/twenty-five-twenty-six\/sfwr-4tb3\/00 Notebooks on Compiler Construction\/img\/Task5\.svg"/,
    )
    assert.match(
      markdown,
      /!\[phase\]\(thoughts\/university\/twenty-five-twenty-six\/sfwr-4tb3\/00 Notebooks on Compiler Construction\/img\/Phases\.svg\)/,
    )
    assert.match(markdown, /src="https:\/\/example\.com\/remote\.svg"/)
    assert.match(markdown, /src="\/static\/icon\.png"/)
  })

  test('separates standalone html images from following markdown prose', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'markdown',
            source:
              '<img style="width:18em;float:right" src="./img/FrontEndBackEnd.svg"></img>\nA common split uses a _front end_ and a _back end_.',
          },
        ],
      }),
      'notes/compiler.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'notes/compiler.ipynb')

    assert.match(markdown, /FrontEndBackEnd\.svg"><\/img>\n\nA common split uses/)
  })

  test('renders text results separately from source blocks', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'code',
            source: '1 + 1',
            outputs: [
              { output_type: 'execute_result', data: { 'text/plain': '2' }, execution_count: 1 },
            ],
          },
        ],
      }),
      'result.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'result.ipynb')

    assert.match(markdown, /```python\n1 \+ 1\n```/)
    assert.match(markdown, /class="notebook-output notebook-output-text"/)
    assert.match(markdown, /data-output-name="result"><samp>2<\/samp><\/pre>/)
  })

  test('renders latex results inside notebook output blocks', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'code',
            source: 'x**2',
            outputs: [
              {
                output_type: 'execute_result',
                data: { 'text/latex': '$\\displaystyle x^{2}$', 'text/plain': 'x**2' },
                execution_count: 1,
              },
            ],
          },
        ],
      }),
      'latex.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'latex.ipynb')

    assert.match(markdown, /class="notebook-output notebook-output-latex"/)
    assert.match(markdown, /data-output-name="result"/)
    assert.match(markdown, /class="katex"/)
    assert.doesNotMatch(markdown, /^\$\\displaystyle/m)
    assert.doesNotMatch(markdown, /<samp>x\*\*2<\/samp>/)
  })

  test('drops inert IPython display placeholders when a richer mime exists', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'code',
            source: 'runwasm("arithmetic.wasm")',
            outputs: [
              {
                output_type: 'display_data',
                data: {
                  'application/javascript': 'element.append("ok")',
                  'text/plain': '<IPython.core.display.Javascript object>',
                },
              },
            ],
          },
        ],
      }),
      'javascript.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'javascript.ipynb')

    assert.doesNotMatch(markdown, /IPython\.core\.display\.Javascript/)
  })

  test('emits ordered runtime metadata for python code cells', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [
          { cell_type: 'markdown', source: 'body' },
          { cell_type: 'code', source: 'x = 1', execution_count: 7 },
          { cell_type: 'code', source: 'x + 1' },
        ],
      }),
      'runtime.ipynb',
    )

    const data = notebookRuntimeData(notebook, 'runtime.ipynb', {
      enabled: true,
      sourcePath: 'notes/runtime.ipynb',
      pyodideIndexUrl: 'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/',
    })

    assert.deepStrictEqual(
      data?.cells.map(cell => ({
        id: cell.id,
        source: cell.source,
        executionIndex: cell.executionIndex,
      })),
      [
        { id: 'cell-1', source: 'x = 1', executionIndex: 7 },
        { id: 'cell-2', source: 'x + 1', executionIndex: null },
      ],
    )
    assert.strictEqual(data?.sourcePath, 'notes/runtime.ipynb')
    assert.match(data?.id ?? '', /^notebook-runtime-/)
  })

  test('embeds escaped runtime json without raw script terminators', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [{ cell_type: 'code', source: '</script><img src=x onerror=alert(1)>' }],
      }),
      'runtime.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'runtime.ipynb', {
      runtime: { enabled: true, sourcePath: 'runtime.ipynb' },
    })

    assert.match(markdown, /data-notebook-runtime-data/)
    assert.match(markdown, /notebook-runtime-toolbar/)
    assert.match(markdown, /data-notebook-run-all/)
    assert.match(markdown, /data-notebook-stop/)
    assert.match(markdown, /data-notebook-reset/)
    assert.match(markdown, /data-notebook-debug aria-pressed="false"/)
    assert.match(markdown, /data-notebook-vim-mode aria-pressed="false"/)
    assert.match(markdown, /Run all/)
    assert.match(markdown, /class="notebook-code-cell" data-notebook-cell-frame="cell-1"/)
    assert.match(markdown, /data-notebook-cell="cell-1"/)
    assert.match(markdown, /data-notebook-execution-label="cell-1"/)
    assert.match(markdown, /In \[ \]:/)
    assert.match(markdown, /data-notebook-run-cell="cell-1"/)
    assert.match(markdown, /data-notebook-edit-cell="cell-1"/)
    assert.match(markdown, /data-notebook-local-source-status="cell-1" hidden/)
    assert.match(markdown, /data-notebook-source-editor="cell-1"/)
    assert.match(markdown, /class="notebook-output notebook-output-success"/)
    assert.match(markdown, /data-output-name="exit 0"/)
    assert.doesNotMatch(markdown, /executed successfully|<em>/)
    const payload = markdown.match(/<script type="application\/json"[^>]*>(.*?)<\/script>/s)
    assert(payload)
    assert.doesNotMatch(payload[1], /<\/script>/i)
    assert.match(markdown, /\\u003c\/script\\u003e/)
  })

  test('keeps parsed code fences inside the pre-emitted runtime cell frame', async () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [{ cell_type: 'code', source: 'x = 1' }],
      }),
      'runtime.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'runtime.ipynb', {
      runtime: { enabled: true, sourcePath: 'runtime.ipynb' },
    })
    const processor = unified()
      .use(remarkParse)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeRaw)
    const tree = (await processor.run(processor.parse(markdown))) as HastElement
    const frame = findElement(
      tree,
      node => node.tagName === 'div' && node.properties?.dataNotebookCellFrame === 'cell-1',
    )

    assert(frame)
    assert(findElement(frame, node => node.tagName === 'pre'))
    const runtimeCell = findElement(
      frame,
      node => node.tagName === 'div' && node.properties?.dataNotebookCell === 'cell-1',
    )
    assert(runtimeCell)
    assert(
      findElement(
        runtimeCell,
        node => node.tagName === 'span' && node.properties?.dataNotebookExecutionLabel === 'cell-1',
      ),
    )
    assert(!findElement(runtimeCell, node => node.tagName === 'p'))
    assert(
      findElement(
        frame,
        node => node.tagName === 'div' && node.properties?.dataNotebookSourceEditor === 'cell-1',
      ),
    )
  })

  test('clears raw markdown floats before the next notebook cell', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'markdown',
            source: '<div style="float:left">\\n\\n```text\\nleft\\n```\\n\\n</div>',
          },
          { cell_type: 'markdown', source: 'after' },
        ],
      }),
      'floats.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'floats.ipynb')

    assert.match(
      markdown,
      /<\/div>\n\n<div class="notebook-markdown-cell-boundary" aria-hidden="true"><\/div>\n\nafter/,
    )
  })

  test('clears raw markdown floats before following markdown blocks in the same cell', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'markdown',
            source:
              '<div style="float:left">\n\n```text\nleft\n```\n\n</div>\n\n```text\nright\n```',
          },
        ],
      }),
      'floats.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'floats.ipynb')

    assert.match(
      markdown,
      /<\/div>\n\n<div class="notebook-markdown-cell-boundary" aria-hidden="true"><\/div>\n+```text\nright/,
    )
  })

  test('closes dangling markdown fences at notebook cell boundaries', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          { cell_type: 'markdown', source: '```EBNF\ninstr ::= "throw" name\n' },
          { cell_type: 'markdown', source: 'after' },
        ],
      }),
      'dangling-fence.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'dangling-fence.ipynb')

    assert.match(markdown, /```EBNF\ninstr ::= "throw" name\n```/)
    assert.match(
      markdown,
      /```\n\n<div class="notebook-markdown-cell-boundary" aria-hidden="true"><\/div>\n\nafter/,
    )
  })

  test('emits markdown cell chunks for independent parsing', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          { cell_type: 'markdown', source: '```EBNF\ninstr ::= "throw" name\n' },
          { cell_type: 'markdown', source: 'after' },
        ],
      }),
      'dangling-fence.ipynb',
    )

    const chunks = notebookToMarkdownChunks(notebook, 'dangling-fence.ipynb')

    assert.match(chunks[1], /^```EBNF\ninstr ::= "throw" name\n```$/)
    assert.strictEqual(
      chunks[2],
      '<div class="notebook-markdown-cell-boundary" aria-hidden="true"></div>',
    )
    assert.strictEqual(chunks[3], 'after')
  })

  test('leaves non-python notebooks without runtime controls', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'javascript' } },
        cells: [{ cell_type: 'code', source: 'console.log("hi")' }],
      }),
      'runtime.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'runtime.ipynb', {
      runtime: { enabled: true, sourcePath: 'runtime.ipynb' },
    })

    assert.doesNotMatch(markdown, /data-notebook-runtime/)
    assert.doesNotMatch(markdown, /Run cell/)
  })
})
