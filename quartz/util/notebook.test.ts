import assert from 'node:assert'
import test, { describe } from 'node:test'
import { notebookTitle, notebookToMarkdown, parseNotebook } from './notebook'

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
    assert.match(markdown, /```python\nprint\("hi"\)\n```/)
    assert.match(markdown, /```text\nhi\n```/)
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
})
