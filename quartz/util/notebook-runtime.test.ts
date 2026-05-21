import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  notebookRuntimeImportCandidates,
  notebookRuntimeModuleSource,
  renderNotebookRuntimeOutput,
  unsupportedNotebookRuntimeReason,
} from './notebook-runtime'

describe('notebook browser runtime output', () => {
  test('renders stdout and stderr into notebook output blocks', () => {
    const stdout = renderNotebookRuntimeOutput({ type: 'stream', name: 'stdout', text: 'hi\n' })
    const stderr = renderNotebookRuntimeOutput({ type: 'stream', name: 'stderr', text: 'nope\n' })

    assert.match(stdout, /notebook-output-stream-stdout/)
    assert.match(stdout, /data-output-name="stdout"><samp>hi<\/samp>/)
    assert.match(stderr, /notebook-output-stream-stderr/)
    assert.match(stderr, /data-output-name="stderr"><samp>nope<\/samp>/)
  })

  test('renders exceptions as error output blocks', () => {
    const html = renderNotebookRuntimeOutput({
      type: 'error',
      ename: 'ValueError',
      evalue: 'bad',
      traceback: 'Traceback\nValueError: bad',
    })

    assert.match(html, /notebook-output-error/)
    assert.match(html, /data-output-name="error"/)
    assert.match(html, /ValueError: bad/)
  })

  test('escapes text results before insertion', () => {
    const html = renderNotebookRuntimeOutput({ type: 'text', text: '<script>alert(1)</script>' })

    assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/)
    assert.doesNotMatch(html, /<script>/)
  })

  test('keeps display html as a display output container', () => {
    const html = renderNotebookRuntimeOutput({ type: 'html', html: '<b>ok</b>' })

    assert.match(html, /notebook-output-html/)
    assert.match(html, /<b>ok<\/b>/)
    assert.doesNotMatch(html, /<script>/)
  })

  test('reports unsupported browser runtime features before execution', () => {
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!wat2wasm arithmetic.wat'),
      'shell escapes are unavailable in the browser runtime',
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('%%writefile x.py'),
      'cell magics are unavailable in the browser runtime',
    )
    assert.strictEqual(unsupportedNotebookRuntimeReason('print("hi")'), undefined)
  })

  test('finds sibling notebook imports and skips import hook packages', () => {
    assert.deepStrictEqual(
      notebookRuntimeImportCandidates(
        [
          'import import_ipynb, textwrap',
          'import SC',
          'from ST import Var, Const',
          'from P0.parser import compileString',
        ].join('\n'),
      ),
      ['textwrap', 'SC', 'ST', 'P0'],
    )
  })

  test('extracts code cells for notebook module imports', () => {
    const source = notebookRuntimeModuleSource(
      JSON.stringify({
        cells: [
          { cell_type: 'markdown', source: '# module' },
          { cell_type: 'code', source: ['x = 1\n', 'y = 2\n'] },
          { cell_type: 'code', source: 'z = x + y' },
        ],
      }),
      'SC.ipynb',
    )

    assert.strictEqual(source, 'x = 1\ny = 2\n\nz = x + y')
  })
})
