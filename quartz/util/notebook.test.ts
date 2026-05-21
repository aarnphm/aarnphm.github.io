import assert from 'node:assert'
import test, { describe } from 'node:test'
import { notebookRuntimeData, notebookTitle, notebookToMarkdown, parseNotebook } from './notebook'

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
    assert.match(markdown, /data-notebook-execution-label="cell-1"/)
    assert.match(markdown, /In \[ \]:/)
    const payload = markdown.match(/<script type="application\/json"[^>]*>(.*?)<\/script>/s)
    assert(payload)
    assert.doesNotMatch(payload[1], /<\/script>/i)
    assert.match(markdown, /\\u003c\/script\\u003e/)
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
