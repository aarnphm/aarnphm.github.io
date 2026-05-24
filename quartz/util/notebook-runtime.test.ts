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

  test('renders successful empty cells into notebook output blocks', () => {
    const html = renderNotebookRuntimeOutput({ type: 'success' })

    assert.match(html, /notebook-output-success/)
    assert.match(html, /data-output-name="exit 0"/)
    assert.doesNotMatch(html, /executed successfully|<em>/)
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
    assert.match(source, /assetSlugForContent\(ctx, 'postscript', '\.js', postscript\)/)
    assert.match(source, /'quartz\/util\/notebook-runtime\.ts'/)
  })

  test('chunks browser basedpyright outside the notebook runtime client', async () => {
    const componentSource = await readFile(
      new URL('../plugins/emitters/componentResources.tsx', import.meta.url),
      'utf8',
    )
    const clientSource = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )
    const editorSource = await readFile(
      new URL('../components/scripts/notebook-code-editor.ts', import.meta.url),
      'utf8',
    )
    const lspSource = await readFile(
      new URL('../components/scripts/notebook-lsp.ts', import.meta.url),
      'utf8',
    )
    const styleSource = await readFile(
      new URL('../styles/pages/notebook.scss', import.meta.url),
      'utf8',
    )

    assert.match(componentSource, /const notebookRuntimeLspEntry =/)
    assert.match(componentSource, /browser-basedpyright\/dist\/pyright\.worker\.js/)
    assert.match(componentSource, /basedpyright\/dist\/typeshed-fallback\/stdlib/)
    assert.match(componentSource, /function writeNotebookPyrightAssets/)
    assert.match(componentSource, /replaceAll\('notebook-pyright\.worker\.js', pyrightWorkerName\)/)
    assert.match(
      componentSource,
      /replaceAll\('notebook-pyright-typeshed\.json', pyrightTypeshedName\)/,
    )
    assert.match(
      editorSource,
      /const \{ notebookLspExtensions \} = await import\('\.\/notebook-lsp'\)/,
    )
    assert.match(editorSource, /lspExtensions\(config\.lsp\)/)
    assert.match(
      clientSource,
      /lsp: \{[\s\S]*runtimeId: this\.payload\.id,[\s\S]*sourcePath: this\.payload\.sourcePath,[\s\S]*cellId: cell\.id/,
    )
    assert.match(
      lspSource,
      /const notebookPyrightWorkerName = '\.\.\/notebook-pyright\.worker\.js'/,
    )
    assert.match(
      lspSource,
      /const notebookPyrightTypeshedName = '\.\.\/notebook-pyright-typeshed\.json'/,
    )
    assert.match(
      lspSource,
      /function notebookRootPath\(config: NotebookLspConfig\)[\s\S]*\/quartz-notebook\/\$\{encodeURIComponent\(config\.runtimeId\)\}/,
    )
    assert.match(
      lspSource,
      /function notebookWorkspaceFiles\(typeshed: JsonObject, rootPath: string\)/,
    )
    assert.match(lspSource, /\[`\$\{rootPath\}\/\.pyright-root`\]: ''/)
    assert.match(lspSource, /mode: 'foreground'/)
    assert.match(lspSource, /message\.type !== 'browser\/newWorker'/)
    assert.match(lspSource, /initializationOptions: \{ files: typeshed \}/)
    assert.match(lspSource, /createPyrightTransport\(this\.rootUri, workspaceFiles/)
    assert.match(lspSource, /method === 'workspace\/configuration'/)
    assert.doesNotMatch(lspSource, /languageServerExtensions/)
    assert.match(styleSource, /\.cm-diagnosticRange/)
    assert.match(styleSource, /\.cm-tooltip-autocomplete\s*>\s*ul/)
    assert.match(styleSource, /\.cm-lsp-active-parameter/)
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
    assert.match(
      clientSource,
      /this\.clearOutput\(cell\.id\)[\s\S]*try \{[\s\S]*const unsupported = unsupportedNotebookRuntimeReason\(source\)/,
    )
    assert.match(workerSource, /post\(\{ type: 'done', cellId: message\.cellId, failed: true \}\)/)
    assert.match(
      workerSource,
      /let failed = false[\s\S]*catch \(error\) \{[\s\S]*failed = true[\s\S]*post\(\{ type: 'done', cellId: message\.cellId, failed \}\)/,
    )
  })

  test('acknowledges bytes written by pyodide stdout streams', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.pyodide.js', import.meta.url),
      'utf8',
    )

    assert.match(source, /function bufferStreamBytesForCell\(cellId, name, bytes, decoder\)/)
    assert.match(source, /return bytes\.byteLength/)
    assert.match(source, /function emitBufferedStreamForCell\(cellId, name, final = false\)/)
    assert.match(source, /const newline = text\.lastIndexOf\('\\n'\)/)
    assert.match(source, /emitOutputForCell\(cellId, \{ type: 'stream', name, text: ready \}\)/)
    assert.match(
      source,
      /pyodide\.setStdout\(\{[\s\S]*write: bytes => bufferStreamBytesForCell\(currentCellId, 'stdout', bytes, stdoutDecoder\)/,
    )
    assert.match(
      source,
      /pyodide\.setStderr\(\{[\s\S]*write: bytes => bufferStreamBytesForCell\(currentCellId, 'stderr', bytes, stderrDecoder\)/,
    )
  })

  test('merges streamed browser output chunks into the active output block', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )

    assert.match(
      source,
      /function appendStreamOutput\(target: HTMLElement, output: RuntimeStreamOutput\)/,
    )
    assert.match(
      source,
      /function lastNotebookOutput\(target: HTMLElement\): HTMLElement \| undefined/,
    )
    assert.match(source, /const previous = lastNotebookOutput\(target\)/)
    assert.match(source, /data-notebook-output-tabs/)
    assert.match(source, /sample\.textContent \+= output\.text/)
    assert.match(source, /previous\.scrollTop = previous\.scrollHeight/)
    assert.match(source, /\{ trimEnd: false \}/)
    assert.match(source, /appendStreamOutput\(target, output\)/)
  })

  test('lets the active running cell interrupt the browser runtime', async () => {
    const source = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )

    assert.match(source, /type NotebookIcon = 'run' \| 'stop'/)
    assert.match(source, /private runningCellId: string \| undefined/)
    assert.match(source, /this\.running && this\.runningCellId === cell\.id[\s\S]*this\.stop\(\)/)
    assert.match(source, /button\.toggleAttribute\('disabled', running && !active\)/)
    assert.match(source, /setNotebookIconButton\(button, 'stop', `Interrupt \$\{cellId\}`\)/)
    assert.match(source, /this\.worker\?\.terminate\(\)/)
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
    assert.match(source, /> \.notebook-runtime-output \{\s+grid-column: 1 \/ -1;/)
    assert.match(
      source,
      /\.notebook-code-cell > \.notebook-runtime-output,\s+\.notebook-code-cell > \.notebook-static-output \{[^}]*display: grid/s,
    )
    assert.match(source, /margin: -0\.35rem 0 1rem/)
    assert.match(
      source,
      /\.notebook-code-cell > \.notebook-runtime-output\[hidden\],\s+\.notebook-code-cell > \.notebook-static-output\[hidden\] \{[^}]*display: none/s,
    )
    assert.match(
      source,
      /\.notebook-code-cell > \.notebook-runtime-output > \.notebook-output:first-child,\s+\.notebook-code-cell > \.notebook-static-output > \.notebook-output:first-child \{[^}]*margin-top: 0/s,
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
    assert.match(source, /event\.stopImmediatePropagation\(\)/)
    assert.match(source, /frame\.addEventListener\('pointerdown', selectSource, true\)/)
    assert.match(source, /frame\.removeEventListener\('pointerdown', selectSource, true\)/)
  })

  test('wires vim-style notebook cell navigation and edit commands', async () => {
    const clientSource = await readFile(
      new URL('../components/scripts/notebook-runtime.client.ts', import.meta.url),
      'utf8',
    )
    const keybindSource = await readFile(
      new URL('../components/scripts/keybind.inline.ts', import.meta.url),
      'utf8',
    )
    const editorSource = await readFile(
      new URL('../components/scripts/notebook-code-editor.ts', import.meta.url),
      'utf8',
    )
    const styleSource = await readFile(
      new URL('../styles/pages/notebook.scss', import.meta.url),
      'utf8',
    )

    assert.doesNotMatch(clientSource, /event\.key === 'g'/)
    assert.match(keybindSource, /function navigateNotebookCell\(delta: -1 \| 1\): boolean/)
    assert.match(keybindSource, /waitingForSecondG && e\.key !== 'g'/)
    assert.match(
      keybindSource,
      /\(e\.key === '\[' \|\| e\.key === ']'\) && navigateNotebookCell\(e\.key === '\[' \? -1 : 1\)/,
    )
    assert.match(keybindSource, /clearGPrefix\(\)/)
    assert.match(clientSource, /event\.key === 'i'/)
    assert.match(clientSource, /private notebookRunKey\(event: KeyboardEvent\): boolean/)
    assert.match(clientSource, /private enterEditMode\(cell: RuntimeCell\)/)
    assert.match(clientSource, /private activeCellFromDocument\(\): RuntimeCell \| undefined/)
    assert.match(clientSource, /\[data-notebook-cell-frame\]\[data-notebook-active-cell\]/)
    assert.match(clientSource, /if \(cell\) this\.activeCellId = cell\.id/)
    assert.match(
      clientSource,
      /this\.cellFromTarget\(target\)[\s\S]*this\.activeCellFromDocument\(\)[\s\S]*this\.cellById\(this\.activeCellId\)/,
    )
    assert.match(clientSource, /event\.key === 'Enter'/)
    assert.match(clientSource, /event\.key === 'Enter' \|\| event\.key === 'i'/)
    assert.match(clientSource, /this\.enterEditMode\(cell\)/)
    assert.match(clientSource, /void this\.runCell\(cell\)/)
    assert.match(clientSource, /event\.key !== 'Escape'/)
    assert.match(clientSource, /void this\.showSourceEditor\(cell, false\)/)
    assert.match(clientSource, /controls\.frame\.focus\(\{ preventScroll: true \}\)/)
    assert.match(clientSource, /renderedSource: cell\.source/)
    assert.match(clientSource, /renderNotebookHighlightedLines/)
    assert.match(clientSource, /controls\.editor\.highlightedLines\(\)/)
    assert.match(clientSource, /function syncOutputTabs\(target: HTMLElement, cellId =/)
    assert.doesNotMatch(clientSource, /groups\.size <= 1/)
    assert.match(clientSource, /function createSuccessOutput\(\): HTMLDivElement/)
    assert.match(
      clientSource,
      /output\.classList\.add\('notebook-output', 'notebook-output-success'\)/,
    )
    assert.match(clientSource, /output\.dataset\.outputName = notebookSuccessOutputLabel/)
    assert.match(clientSource, /function createRunningOutput\(\): HTMLDivElement/)
    assert.match(clientSource, /output\.dataset\.notebookOutputPlaceholder = ''/)
    assert.match(clientSource, /function clearPlaceholderOutputs\(target: HTMLElement\)/)
    assert.match(clientSource, /function hasRenderedOutput\(target: HTMLElement\): boolean/)
    assert.match(
      clientSource,
      /function selectOutputTab\(container: HTMLElement, activeId: string \| undefined, focusId = activeId\)/,
    )
    assert.match(
      clientSource,
      /container\.toggleAttribute\('data-notebook-output-collapsed', activeId === undefined\)/,
    )
    assert.match(
      clientSource,
      /const active = activeId !== undefined && tab\.dataset\.notebookOutputTab === activeId/,
    )
    assert.match(
      clientSource,
      /panel\.hidden = activeId === undefined \|\| panel\.dataset\.notebookOutputPanel !== activeId/,
    )
    assert.match(
      clientSource,
      /const wasCollapsed = previousContainer\?\.hasAttribute\('data-notebook-output-collapsed'\) \?\? true/,
    )
    assert.match(
      clientSource,
      /const defaultActiveId = orderedGroups\.find\(\s*group => group\.label !== notebookSuccessOutputLabel,\s*\)\?\.id/,
    )
    assert.match(
      clientSource,
      /previousContainer === null[\s\S]*\? defaultActiveId[\s\S]*: !wasCollapsed && previousActive && orderedGroups\.some\(group => group\.id === previousActive\)/,
    )
    assert.match(
      clientSource,
      /selectOutputTab\(container, active \? undefined : group\.id, group\.id\)/,
    )
    assert.match(clientSource, /if \(output\.type === 'success'\)/)
    assert.match(clientSource, /if \(!result\.failed\) this\.renderSuccessOutput\(cell\.id\)/)
    assert.match(clientSource, /private renderSuccessOutput\(cellId: string\)/)
    assert.match(clientSource, /if \(!target \|\| hasRenderedOutput\(target\)\) return/)
    assert.match(clientSource, /this\.renderOutput\(cellId, \{ type: 'success' \}\)/)
    assert.match(clientSource, /target\.replaceChildren\(createRunningOutput\(\)\)/)
    assert.match(clientSource, /syncOutputTabs\(target, cellId\)/)
    assert.doesNotMatch(
      clientSource,
      /this\.hideSavedOutput\(cell\.id\)\s+await this\.ensureWorker\(\)/,
    )
    assert.match(
      clientSource,
      /function syncStaticOutputTabs\(frame: HTMLElement, cellId: string\)/,
    )
    assert.doesNotMatch(
      clientSource,
      /new Set\(outputs\.map\(output => outputTabId\(outputLabel\(output\)\)\)\)\.size <= 1/,
    )
    assert.match(clientSource, /if \(outputs\.length === 0\) return/)
    assert.match(clientSource, /syncStaticOutputTabs\(existingFrame, cell\.id\)/)
    assert.match(clientSource, /container\.dataset\.notebookStaticOutput = cellId/)
    assert.match(clientSource, /target\.dataset\.notebookOutputTabbed = ''/)
    assert.match(clientSource, /sibling\.hasAttribute\('data-notebook-static-output'\)/)
    assert.match(clientSource, /function syncStreamScrollHint\(output: HTMLElement\)/)
    assert.match(clientSource, /data-notebook-scroll-before/)
    assert.match(clientSource, /data-notebook-scroll-after/)
    assert.match(clientSource, /syncStreamScrollHints\(container\)/)
    assert.match(clientSource, /role', 'tablist'/)
    assert.match(clientSource, /aria-orientation', 'horizontal'/)
    assert.match(clientSource, /role', 'tabpanel'/)
    assert.match(editorSource, /forceParsing\(view, view\.state\.doc\.length, 500\)/)
    assert.match(editorSource, /getComputedStyle\(source\)/)
    assert.match(editorSource, /target\.style\.setProperty\('--shiki-light', style\.color\)/)
    assert.match(editorSource, /function configureNotebookVimBindings\(vimApi: NotebookVimModule\)/)
    assert.match(editorSource, /\['insert', 'jj', '<Esc>'\]/)
    assert.match(editorSource, /\['insert', 'jk', '<Esc>'\]/)
    assert.match(editorSource, /\['normal', ';', ':'\]/)
    assert.match(editorSource, /\['normal', '\\\\', ':noh<CR>'\]/)
    assert.match(
      editorSource,
      /const notebookRunKeys = \['Mod-Enter', 'Ctrl-Enter', 'Shift-Enter', 'Alt-Enter'\]/,
    )
    assert.match(editorSource, /\.\.\.notebookRunKeys\.map\(key => \(\{/)
    assert.match(editorSource, /key: 'Mod-s'[\s\S]*config\.onSave\?\.\(\)/)
    assert.match(editorSource, /key: 'Ctrl-s'[\s\S]*config\.onSave\?\.\(\)/)
    assert.match(editorSource, /key: 'Escape'[\s\S]*config\.onCancel\?\.\(\)/)
    assert.match(
      editorSource,
      /Vim\.mapCommand\([\s\S]*?'J'[\s\S]*?'notebookMoveSelectedLinesDown'/,
    )
    assert.match(editorSource, /Vim\.mapCommand\([\s\S]*?'K'[\s\S]*?'notebookMoveSelectedLinesUp'/)
    assert.match(editorSource, /export async function renderNotebookHighlightedLines/)
    assert.match(editorSource, /highlightedLines: \(\) => highlightedLineSpans\(view\)/)
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
      /border: 1px solid color-mix\(in srgb, var\(--lightgray\) 78%, transparent\)/,
    )
    assert.match(styleSource, /outline: 0/)
    assert.match(
      styleSource,
      /border-color: color-mix\(in srgb, var\(--notebook-active-green\) 82%, transparent\)/,
    )
    assert.match(styleSource, /padding: var\(--notebook-shell-padding\)/)
    assert.match(
      styleSource,
      /\.notebook-cell-actions \{[^}]*left: calc\(\s*var\(--notebook-shell-padding\) \+ var\(--notebook-gutter-width\) \+ var\(--notebook-code-gap\)\s*\);/s,
    )
    assert.match(
      styleSource,
      /@media all and \(\$mobile\) \{[\s\S]*\.notebook-code-cell\s*>\s*\.notebook-cell-actions \{[\s\S]*left: auto;[\s\S]*right: var\(--notebook-shell-padding\);/,
    )
    assert.match(styleSource, /\.notebook-output-tabs \{/)
    assert.match(
      styleSource,
      /--notebook-output-tab-active-surface: color-mix\([\s\S]*in srgb,[\s\S]*var\(--notebook-active-green\) 16%,[\s\S]*var\(--notebook-output-surface\)[\s\S]*\);/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-tabs \{[\s\S]*grid-template-rows: 180px var\(--notebook-action-height\);[\s\S]*grid-template-columns: minmax\(0, 1fr\);/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-tabs\[data-notebook-output-collapsed\] \{[\s\S]*grid-template-rows: 0 var\(--notebook-action-height\);/,
    )
    assert.match(styleSource, /\.notebook-output-tabs \{[\s\S]*grid-column: 2;/)
    assert.match(
      styleSource,
      /\.notebook-code-cell\s*>\s*\.notebook-runtime-output\s*>\s*\.notebook-output-tabs:first-child \{[\s\S]*margin-top: 0\.65rem/s,
    )
    assert.match(
      styleSource,
      /pre\.notebook-output-stream,\n  pre\.notebook-output-json \{[\s\S]*max-height: min\(18rem, 45vh\)/,
    )
    assert.match(
      styleSource,
      /pre\.notebook-output-stream \{[\s\S]*--notebook-stream-blur-size: 1\.2rem/,
    )
    assert.match(styleSource, /pre\.notebook-output-stream\[data-notebook-scroll-before\]/)
    assert.match(styleSource, /pre\.notebook-output-stream\[data-notebook-scroll-after\]/)
    assert.match(styleSource, /mask-image: linear-gradient/)
    assert.match(styleSource, /\.notebook-output-tablist \{/)
    assert.match(
      styleSource,
      /\.notebook-output-tablist \{[\s\S]*display: inline-flex;[\s\S]*grid-row: 2;[\s\S]*align-items: center;[\s\S]*justify-self: flex-start;[\s\S]*gap: 0;[\s\S]*padding: 0;[\s\S]*border-top: 0;[\s\S]*border-radius: 0 0 var\(--radius-3\) var\(--radius-3\);/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-tabs\[data-notebook-output-collapsed\] \.notebook-output-tablist \{[\s\S]*border: 1px solid var\(--lightgray\);[\s\S]*border-radius: var\(--radius-3\);/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-tablist:has\(\.notebook-output-tab\[aria-selected='true'\]\) \{[\s\S]*background: var\(--notebook-output-tab-active-surface\);/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-tab \{[\s\S]*height: 100%;[\s\S]*border: 0;[\s\S]*border-radius: 0;/,
    )
    assert.match(styleSource, /\.notebook-output-tab:active \{[\s\S]*scale: 0\.96;/)
    assert.match(
      styleSource,
      /\.notebook-output-tab\[aria-selected='false'\] \{[\s\S]*background: var\(--notebook-output-surface\);/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-tab\[aria-selected='true'\] \{[\s\S]*background: var\(--notebook-output-tab-active-surface\);/,
    )
    assert.doesNotMatch(styleSource, /\.notebook-output-tab\[aria-selected='true'\]::after/)
    assert.match(
      styleSource,
      /\.notebook-output-panels \{[\s\S]*grid-row: 1;[\s\S]*grid-column: 1;[\s\S]*border-radius: var\(--notebook-inner-radius\) var\(--notebook-inner-radius\)[\s\S]*var\(--notebook-inner-radius\) 0;/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-tabs\[data-notebook-output-collapsed\] \.notebook-output-panels \{[\s\S]*visibility: hidden;[\s\S]*border-width: 0;/,
    )
    assert.match(styleSource, /\.notebook-output-panel\[hidden\] \{/)
    assert.match(
      styleSource,
      /\.notebook-output-panel \{[\s\S]*height: 100%;[\s\S]*overflow: auto;/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-tabs pre\.notebook-output,[\s\S]*\.notebook-output-tabs \.notebook-output-latex \{[\s\S]*height: 100%;[\s\S]*max-height: none;/,
    )
    assert.match(
      styleSource,
      /\.notebook-output-success,[\s\S]*\.notebook-output-placeholder \{[\s\S]*min-height: 100%;/,
    )
    assert.match(
      styleSource,
      /\.notebook-code-cell\s*>\s*\.notebook-output\[data-output-name\]::before/,
    )
    assert.match(
      styleSource,
      /\.notebook-code-cell\s*>\s*\.notebook-static-output\s*>\s*\.notebook-output\[data-output-name\]::before \{[\s\S]*display: none/s,
    )
  })
})
