import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  notebookRuntimeImportCandidates,
  notebookRuntimeLocalSourceKey,
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

  test('preserves stdout newlines inside output blocks', () => {
    const html = renderNotebookRuntimeOutput({
      type: 'stream',
      name: 'stdout',
      text: '.data\nx_:\t.space 4\n\t.text\n',
    })

    assert.match(html, /\.data\nx_:/)
    assert.match(html, /x_:\t\.space 4\n\t\.text/)
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

  test('builds a stable local browser source key per notebook cell', () => {
    assert.strictEqual(
      notebookRuntimeLocalSourceKey(
        'content/thoughts/university/sfwr-4tb3/03 Out of Registers.ipynb',
        'cell-1',
      ),
      'quartz:notebook-source:content%2Fthoughts%2Funiversity%2Fsfwr-4tb3%2F03%20Out%20of%20Registers.ipynb:cell-1',
    )
  })

  test('finds sibling notebook imports and skips import hook packages', () => {
    assert.deepStrictEqual(
      notebookRuntimeImportCandidates(
        [
          'import import_ipynb, textwrap',
          'import nbimporter; nbimporter.options["only_defs"] = False',
          'import SC',
          'from ST import Var, Const',
          "if target == 'riscv': import CGriscv as CG",
          'from P0.parser import compileString',
        ].join('\n'),
      ),
      ['textwrap', 'SC', 'ST', 'CGriscv', 'P0'],
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
