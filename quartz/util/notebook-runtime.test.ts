import { transform } from 'esbuild'
import assert from 'node:assert'
import { readFile } from 'node:fs/promises'
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
    assert.doesNotMatch(html, /data-output-name="debug"/)
  })

  test('renders error debug metadata only when requested', () => {
    const output = {
      type: 'error' as const,
      ename: 'ValueError',
      evalue: 'bad <input>',
      traceback: 'Traceback\nValueError: bad <input>',
      debug: {
        phase: 'running python',
        cellId: 'cell-1',
        errorName: 'PythonError',
        errorMessage: 'bad <input>',
        stack: 'stack <frame>',
      },
    }

    const normal = renderNotebookRuntimeOutput(output)
    const debug = renderNotebookRuntimeOutput(output, { debug: true })

    assert.doesNotMatch(normal, /data-output-name="debug"/)
    assert.match(debug, /notebook-output-debug/)
    assert.match(debug, /phase: running python/)
    assert.match(debug, /cell: cell-1/)
    assert.match(debug, /stack &lt;frame&gt;/)
    assert.doesNotMatch(debug, /stack <frame>/)
  })

  test('escapes text results before insertion', () => {
    const html = renderNotebookRuntimeOutput({ type: 'text', text: '<script>alert(1)</script>' })

    assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/)
    assert.doesNotMatch(html, /<script/i)
  })

  test('keeps display html as a display output container', () => {
    const html = renderNotebookRuntimeOutput({ type: 'html', html: '<b>ok</b>' })

    assert.match(html, /notebook-output-html/)
    assert.match(html, /<b>ok<\/b>/)
    assert.doesNotMatch(html, /<script/i)
  })

  test('renders json results into notebook output blocks', () => {
    const html = renderNotebookRuntimeOutput({
      type: 'json',
      text: '[\n  [\n    "E",\n    "",\n    "T",\n    0\n  ]\n]',
    })

    assert.match(html, /notebook-output-json/)
    assert.match(html, /data-output-name="json"/)
    assert.match(html, /&quot;E&quot;/)
  })

  test('reports unsupported browser runtime features before execution', () => {
    assert.strictEqual(unsupportedNotebookRuntimeReason('!cat x.py'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('!ls .'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('!ls -la arithmetic*'), undefined)
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!wat2wasm arithmetic.wat\n!ls -la arithmetic*'),
      undefined,
    )
    assert.strictEqual(unsupportedNotebookRuntimeReason('!wat2wasm arithmetic.wat'), undefined)
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!wat2wasm --enable-bulk-memory array.wat'),
      undefined,
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!wat2wasm -v arithmetic.wat'),
      'wat2wasm option -v is unavailable in the browser runtime',
    )
    assert.strictEqual(unsupportedNotebookRuntimeReason('!wasm2wat arithmetic.wasm'), undefined)
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!wasm2wat --fold-exprs arithmetic.wasm'),
      undefined,
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!wasm2wat -v arithmetic.wasm'),
      'wasm2wat option -v is unavailable in the browser runtime',
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!ls --color arithmetic*'),
      'ls option --color is unavailable in the browser runtime',
    )
    assert.strictEqual(unsupportedNotebookRuntimeReason('%%writefile x.py'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('%%writefile -a x.py\nprint(1)'), undefined)
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('%%writefile -q x.py\nprint(1)'),
      '%%writefile option -q is unavailable in the browser runtime',
    )
    assert.strictEqual(unsupportedNotebookRuntimeReason('%pip install numpy'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('%time p3b.parse(s)'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('%timeit x.block_until_ready()'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('%load_ext nb_mypy'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('load_ext nb_mypy'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('%reload_ext autoreload'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('%autoreload 2'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('%matplotlib inline'), undefined)
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('%load_ext local_extension'),
      'IPython extension local_extension is unavailable in the browser runtime',
    )
    assert.strictEqual(unsupportedNotebookRuntimeReason('!cat sum.s'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('!pip install pandas'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('!uv pip install seaborn'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('!python -m pip install rich'), undefined)
    assert.strictEqual(unsupportedNotebookRuntimeReason('print("hi")'), undefined)
  })

  test('rejects notebook shell paths outside the browser sandbox before execution', () => {
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!cat ../secret.py'),
      'cat path ../secret.py is outside the browser runtime sandbox',
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!cat /etc/passwd'),
      'cat path /etc/passwd is outside the browser runtime sandbox',
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!ls ../'),
      'ls path ../ is outside the browser runtime sandbox',
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!wat2wasm ../arithmetic.wat'),
      'wat2wasm path ../arithmetic.wat is outside the browser runtime sandbox',
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('!wasm2wat arithmetic.wasm -o ../arithmetic.wat'),
      'wasm2wat path ../arithmetic.wat is outside the browser runtime sandbox',
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('%%writefile ../arithmetic.wat\n(module)'),
      '%%writefile path ../arithmetic.wat is outside the browser runtime sandbox',
    )
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
          'import jax',
          'import torch.nn.functional as F',
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

  test('keeps inline runtime loaders valid browser javascript', async () => {
    const root = new URL('../components/scripts/', import.meta.url)
    const scripts = [await readFile(new URL('notebook-runtime.inline.ts', root), 'utf8')]

    await transform(scripts.map(script => `(function(){${script}})();`).join('\n'), {
      loader: 'js',
      minify: true,
    })
  })

  test('schedules notebook runtime mounting on initial load and navigation', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.inline.ts', import.meta.url),
      'utf8',
    )

    assert.match(
      source,
      /document\.addEventListener\('nav', scheduleNotebookRuntimeMount\)\s+scheduleNotebookRuntimeMount\(\)/,
    )
  })

  test('decodes escaped notebook runtime data before mounting', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.inline.ts', import.meta.url),
      'utf8',
    )

    assert.match(source, /const decoded = text\.replace\(/)
    assert.match(source, /decodeHtmlEntity/)
    assert.doesNotMatch(source, /replaceAll\('&amp;', '&'\)/)
    assert.match(source, /JSON\.parse\(decoded\)/)
    assert.doesNotMatch(source, /innerHTML\s*=\s*text/)
  })

  test('rewrites the page bootstrap when the notebook runtime loader changes', async () => {
    const source = await readFile(
      new URL('../plugins/emitters/componentResources.tsx', import.meta.url),
      'utf8',
    )

    assert.match(source, /const notebookRuntimeInlineEntry =/)
    assert.match(source, /function isNotebookRuntimePageScriptChange/)
    assert.match(source, /slug: 'postscript' as FullSlug/)
    assert.match(source, /'quartz\/util\/notebook-runtime\.ts'/)
  })

  test('keeps application/javascript display output inert in the browser runtime', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.pyodide.js', import.meta.url),
      'utf8',
    )

    assert.doesNotMatch(source, /display-javascript/)
    assert.match(source, /emitOutputForCell\(cellId, \{ type: 'text', text: textOf\(code\) \}\)/)
  })

  test('stops run all after a browser runtime cell failure', async () => {
    const clientSource = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )
    const workerSource = await readFile(
      new URL('../components/scripts/notebook-runtime.pyodide.js', import.meta.url),
      'utf8',
    )

    assert.match(clientSource, /const succeeded = await this\.runCell\(cell\)/)
    assert.match(clientSource, /this\.setStatus\(`stopped after \$\{cell\.id\}`\)/)
    assert.match(clientSource, /failed: value\.failed === true/)
    assert.match(clientSource, /waiter\.resolve\(\{ failed: message\.failed \}\)/)
    assert.match(workerSource, /post\(\{ type: 'done', cellId: message\.cellId, failed: true \}\)/)
    assert.match(
      workerSource,
      /let failed = false[\s\S]*catch \(error\) \{[\s\S]*failed = true[\s\S]*post\(\{ type: 'done', cellId: message\.cellId, failed \}\)/,
    )
  })

  test('routes browser list display payloads through formatted json output', async () => {
    const pythonSource = await readFile(
      new URL('../components/scripts/notebook-runtime.pyodide.py', import.meta.url),
      'utf8',
    )
    const workerSource = await readFile(
      new URL('../components/scripts/notebook-runtime.pyodide.js', import.meta.url),
      'utf8',
    )

    assert.match(pythonSource, /def _quartz_json_display_bundle\(value\):/)
    assert.match(pythonSource, /'application\/json': text/)
    assert.match(workerSource, /hasDisplayMime\(data, 'application\/json'\)/)
    assert.match(workerSource, /emitOutput\(\{ type: 'json', text: jsonTextOf/)
  })

  test('translates notebook time line magics into browser runtime calls', async () => {
    const pythonSource = await readFile(
      new URL('../components/scripts/notebook-runtime.pyodide.py', import.meta.url),
      'utf8',
    )
    const workerSource = await readFile(
      new URL('../components/scripts/notebook-runtime.pyodide.js', import.meta.url),
      'utf8',
    )

    assert.match(pythonSource, /def __quartz_time\(statement, global_ns, local_ns\):/)
    assert.match(workerSource, /function timeDirective\(line\)/)
    assert.match(
      workerSource,
      /__quartz_time\(\$\{JSON\.stringify\(statement\)\}, globals\(\), locals\(\)\)/,
    )
    assert.match(workerSource, /timeitDirective\(line\) \?\? timeDirective\(line\) \?\? line/)
  })

  test('renders browser runtime output without string insertion sinks', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )

    assert.match(source, /DOMPurify\.sanitize\(html/)
    assert.match(source, /RETURN_DOM_FRAGMENT: true/)
    assert.match(source, /samp\.textContent =/)
    assert.doesNotMatch(source, /insertAdjacentHTML\(\s*['"]beforeend/)
    assert.doesNotMatch(source, /template\.innerHTML = html/)
  })

  test('keeps rerun output attached to the rendered code block', async () => {
    const source = await readFile(new URL('../styles/pages/notebook.scss', import.meta.url), 'utf8')

    assert.match(source, /\.notebook-code-cell > \.notebook-runtime-cell \{[^}]*display: flex/s)
    assert.match(source, /\.notebook-code-cell > \.notebook-runtime-cell > p \{[^}]*margin: 0/s)
    assert.match(source, /\.notebook-code-cell > \.notebook-runtime-output \{/)
    assert.match(source, /margin: -0\.35rem 0 1rem/)
    assert.match(source, /\.notebook-code-cell > \.notebook-runtime-output\[hidden\] \{/)
    assert.match(
      source,
      /\.notebook-code-cell > \.notebook-runtime-output > \.notebook-output:first-child \{/,
    )
  })

  test('keeps notebook cell lookups scoped to the mounted runtime page', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )

    assert.match(source, /private cellRoot: Document \| HTMLElement/)
    assert.doesNotMatch(
      source,
      /document\.querySelector(?:All)?<[^>]+>\(\s*`?\[data-notebook-(?:cell|run-cell|output|execution-label)/,
    )
  })

  test('scopes notebook command-mode keyboard handling to selected notebook cells', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )

    assert.match(
      source,
      /document\.addEventListener\('keydown', this\.handleNotebookKeydown, true\)/,
    )
    assert.match(
      source,
      /document\.removeEventListener\('keydown', this\.handleNotebookKeydown, true\)/,
    )
    assert.match(source, /private siteShortcutLayerActive\(\): boolean/)
    assert.match(
      source,
      /target\.closest\('input, textarea, select, button, a\[href\], \[contenteditable\]'\)/,
    )
    assert.match(source, /event\.stopPropagation\(\)/)
  })

  test('wires vim-style notebook cell navigation and edit commands', async () => {
    const clientSource = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )
    const styleSource = await readFile(
      new URL('../styles/pages/notebook.scss', import.meta.url),
      'utf8',
    )

    assert.match(clientSource, /event\.key === 'g'/)
    assert.match(clientSource, /event\.key === '\[' \|\| event\.key === ']'/)
    assert.match(clientSource, /this\.selectAdjacentCell\(cell, event\.key === '\[' \? -1 : 1\)/)
    assert.match(clientSource, /event\.key === 'i'/)
    assert.match(clientSource, /event\.key !== 'Escape'/)
    assert.match(clientSource, /void this\.showSourceEditor\(cell, false\)/)
    assert.match(styleSource, /\[data-notebook-active-cell\]/)
    assert.match(styleSource, /--notebook-shell-padding: 0\.55rem/)
    assert.match(styleSource, /--notebook-inner-radius: var\(--radius-5\)/)
    assert.match(
      styleSource,
      /--notebook-shell-radius: calc\(var\(--notebook-inner-radius\) \+ var\(--notebook-shell-padding\)\)/,
    )
    assert.match(
      styleSource,
      /--notebook-active-green: var\(--background-modifier-success, var\(--lime\)\)/,
    )
    assert.match(
      styleSource,
      /box-shadow: 0 0 0 1px color-mix\(in srgb, var\(--lightgray\) 78%, transparent\)/,
    )
    assert.match(styleSource, /var\(--notebook-active-green\) 82%/)
    assert.match(styleSource, /padding: var\(--notebook-shell-padding\)/)
  })
})
