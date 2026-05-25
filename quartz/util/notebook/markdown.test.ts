import type { Element, Root as HtmlRoot } from 'hast'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import rehypeRaw from 'rehype-raw'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import type { FullSlug } from '../path'
import { unsupportedNotebookRuntimeReason } from '../../runtime/python/can-execute'
import { notebookCellActions, notebookCellLanguageBadge } from './cell-html'
import {
  notebookRuntimeData,
  notebookTitle,
  notebookToMarkdown,
  notebookToMarkdownChunks,
} from './markdown'
import { parseNotebookDoc } from './parse'
import { renderRuntimeOutputHtml } from './render/runtime-output-to-hast'
import { findNotebookCellFrame, notebookCellRef, notebookCellRuntimeNodes } from './transclude'
import { isNotebookParseError, type NotebookDoc } from './types'

function parseNotebook(raw: string, sourcePath: string): NotebookDoc {
  const doc = parseNotebookDoc(raw, sourcePath)
  if (isNotebookParseError(doc)) throw new Error(doc.reason)
  return doc
}

const renderNotebookRuntimeOutput = renderRuntimeOutputHtml

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

function findElements(
  node: HastElement,
  predicate: (node: HastElement) => boolean,
  results: HastElement[] = [],
): HastElement[] {
  if (predicate(node)) results.push(node)
  for (const child of node.children ?? []) findElements(child, predicate, results)
  return results
}

function elementClassNames(node: HastElement): string[] {
  const className = node.properties?.className
  return Array.isArray(className) ? className.filter(item => typeof item === 'string') : []
}

function textChild(node: HastElement): string {
  return (node.children ?? [])
    .filter((child): child is HastElement & { value: string } => child.type === 'text')
    .map(child => child.value)
    .join('')
}

async function parseHtmlFragment(value: string): Promise<HastElement> {
  const processor = unified()
    .use(remarkParse)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
  return (await processor.run(processor.parse(value))) as HastElement
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

  test('keeps html figure captions with their images', async () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'markdown',
            source:
              '<figure style="width: 20%;float: right;border-left:10px solid transparent;">\n    <img src="./pathfinder-concept.webp"/>\n    <figcaption style="text-align: center;">\n        <small>Credit: <a href="https://www.nasa.gov/mission_pages/pathfinder/overview">NASA</a></small>\n    </figcaption>\n</figure>',
          },
        ],
      }),
      'thoughts/university/twenty-three-twenty-four/sfwr-3bb4/00 Concurrent System Design/Concurrent System Design.ipynb',
    )

    const markdown = notebookToMarkdown(
      notebook,
      'thoughts/university/twenty-three-twenty-four/sfwr-3bb4/00 Concurrent System Design/Concurrent System Design.ipynb',
    )
    const processor = unified()
      .use(remarkParse)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeRaw)
    const tree = (await processor.run(processor.parse(markdown))) as HastElement
    const figure = findElement(tree, node => node.tagName === 'figure')

    assert(figure)
    assert(findElement(figure, node => node.tagName === 'img'))
    assert(findElement(figure, node => node.tagName === 'figcaption'))
    assert(!findElement(figure, node => node.tagName === 'pre'))
    assert.doesNotMatch(markdown, /pathfinder-concept\.webp"\/>\n\n    <figcaption/)
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
      indexUrl: 'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/',
      importableModules: ['ST', 'SC', 'SC'],
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
    assert.deepStrictEqual(data?.importableModules, ['SC', 'ST'])
    assert.match(data?.id ?? '', /^notebook-runtime-/)
  })

  test('emits javascript runtime cells from javascript magics', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [
          { cell_type: 'code', source: 'x = 1' },
          { cell_type: 'code', source: '%%javascript\nconsole.log("hi")' },
        ],
      }),
      'mixed-runtime.ipynb',
    )

    const data = notebookRuntimeData(notebook, 'mixed-runtime.ipynb', {
      enabled: true,
      sourcePath: 'mixed-runtime.ipynb',
    })

    assert.deepStrictEqual(
      data?.cells.map(cell => ({
        id: cell.id,
        language: cell.language,
        displayLanguage: cell.displayLanguage,
      })),
      [
        { id: 'cell-1', language: 'python', displayLanguage: undefined },
        { id: 'cell-2', language: 'javascript', displayLanguage: undefined },
      ],
    )
  })

  test('preserves empty runtime import lists to avoid probing normal python modules', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [{ cell_type: 'code', source: 'from collections.abc import Iterator' }],
      }),
      'runtime.ipynb',
    )

    const data = notebookRuntimeData(notebook, 'runtime.ipynb', {
      enabled: true,
      sourcePath: 'runtime.ipynb',
      importableModules: [],
    })

    assert.deepStrictEqual(data?.importableModules, [])
  })

  test('reports unsupported browser threading before notebook execution', () => {
    const reason =
      'Python threading and multiprocessing are unavailable in the browser runtime because Pyodide does not support starting threads or processes. Use QUARTZ_NOTEBOOK_MODE=execute or a server Python runtime for this cell.'

    assert.strictEqual(
      unsupportedNotebookRuntimeReason(
        'from threading import Thread\nThread(target=print).start()',
      ),
      reason,
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason(
        'import threading as th\nworker = th.Thread(target=print)\nworker.start()',
      ),
      reason,
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason(
        'from concurrent.futures import ThreadPoolExecutor\nThreadPoolExecutor(max_workers=2)',
      ),
      reason,
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason(
        'import concurrent.futures as futures\npool = futures.ThreadPoolExecutor(max_workers=2)',
      ),
      reason,
    )

    const html = renderNotebookRuntimeOutput({
      type: 'error',
      ename: 'UnsupportedRuntimeFeature',
      evalue: reason,
      traceback: reason,
    })

    assert.match(html, /notebook-output-error/)
    assert.match(html, /Pyodide does not support starting threads or processes/)
    assert.doesNotMatch(html, /threading\.py/)
  })

  test('allows writefile cells within the browser runtime sandbox', () => {
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('%%writefile SetXY.java\npublic class SetXY {}'),
      undefined,
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('%%writefile -a output.txt\nmore output'),
      undefined,
    )
    assert.strictEqual(
      unsupportedNotebookRuntimeReason('%%writefile ../SetXY.java\npublic class SetXY {}'),
      '%%writefile path ../SetXY.java is outside the browser runtime sandbox',
    )
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
    assert.match(markdown, /data-notebook-debug/)
    assert.match(markdown, /data-notebook-vim-mode/)
    assert.match(markdown, /data-notebook-tooltip="Run all"/)
    assert.match(markdown, /data-notebook-tooltip="Stop execution"/)
    assert.doesNotMatch(markdown, />Run all</)
    assert.match(markdown, /class="notebook-code-cell" data-notebook-cell-frame="cell-1"/)
    assert.match(markdown, /data-notebook-cell="cell-1"/)
    assert.match(markdown, /data-notebook-execution-label="cell-1"/)
    assert.match(markdown, /In \[ \]:/)
    assert.match(markdown, /data-notebook-run-cell="cell-1"/)
    assert.match(markdown, /data-notebook-edit-cell="cell-1"/)
    assert.match(markdown, /class="notebook-language-badge notebook-language-badge-python"/)
    assert.match(markdown, /data-notebook-language="python"/)
    assert.match(markdown, /notebook-language-label/)
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

  test('renders notebook runtime toolbar controls as icon buttons with tooltip labels', async () => {
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
    const tree = await parseHtmlFragment(markdown)
    const toolbar = findElement(tree, node =>
      elementClassNames(node).includes('notebook-runtime-toolbar'),
    )
    assert(toolbar)
    const buttons = findElements(toolbar, node => node.tagName === 'button')

    assert.deepStrictEqual(
      buttons.map(button => button.properties?.dataNotebookTooltip),
      ['Run all', 'Stop execution', 'Reset runtime', 'Enable debug output', 'Enable Vim mode'],
    )
    assert(buttons.every(button => elementClassNames(button).includes('notebook-icon-button')))
    assert(buttons.every(button => findElement(button, child => child.tagName === 'svg')))
    assert(buttons.every(button => textChild(button) === ''))
    const resetButton = buttons.find(button => button.properties?.dataNotebookReset === '')
    assert(resetButton)
    assert(
      findElement(
        resetButton,
        child => child.tagName === 'path' && child.properties?.d === 'M9 14 4 9l5-5',
      ),
    )
    assert.strictEqual(buttons[3]?.properties?.ariaPressed, 'false')
    assert.strictEqual(buttons[4]?.properties?.ariaPressed, 'false')
  })

  test('pre-renders runtime output tabs for stored outputs', async () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [
          {
            cell_type: 'code',
            source: 'print("hi")',
            execution_count: 1,
            outputs: [{ output_type: 'stream', name: 'stdout', text: ['hi\n'] }],
          },
        ],
      }),
      'runtime.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'runtime.ipynb', {
      runtime: { enabled: true, sourcePath: 'runtime.ipynb' },
    })
    const tree = await parseHtmlFragment(markdown)
    const frame = findElement(
      tree,
      node => node.tagName === 'div' && node.properties?.dataNotebookCellFrame === 'cell-1',
    )
    assert(frame)
    const staticOutput = findElement(
      frame,
      node => node.tagName === 'div' && node.properties?.dataNotebookStaticOutput === 'cell-1',
    )
    assert(staticOutput)
    assert.strictEqual(staticOutput.properties?.dataNotebookOutputTabbed, '')
    const tab = findElement(
      staticOutput,
      node => node.tagName === 'button' && node.properties?.dataNotebookOutputTab === 'stdout',
    )
    assert(tab)
    assert.strictEqual(tab.properties?.ariaSelected, 'true')
    const actions = findElement(staticOutput, node =>
      elementClassNames(node).includes('notebook-output-actions'),
    )
    assert(actions)
    const buttons = (actions.children ?? []).filter(child => child.tagName === 'button')
    assert.strictEqual(buttons.length, 2)
    assert.strictEqual(buttons[0]?.properties?.dataNotebookOutputExpandAction, 'stdout')
    assert.strictEqual(buttons[0]?.properties?.ariaExpanded, 'false')
    assert(elementClassNames(buttons[0]).includes('notebook-output-expand-button'))
    assert.strictEqual(buttons[1]?.properties?.dataNotebookOutputAction, 'stdout')
    assert(elementClassNames(buttons[1]).includes('notebook-output-copy-button'))
    const panel = findElement(
      staticOutput,
      node => node.tagName === 'div' && node.properties?.dataNotebookOutputPanel === 'stdout',
    )
    assert(panel)
    assert.strictEqual(panel.properties?.hidden, undefined)
    const stdout = findElement(panel, node =>
      elementClassNames(node).includes('notebook-output-stream-stdout'),
    )
    assert(stdout)
    const samp = findElement(stdout, node => node.tagName === 'samp')
    assert(samp)
    assert.strictEqual(textChild(samp), 'hi')
    assert(
      findElement(
        frame,
        node => node.tagName === 'div' && node.properties?.dataNotebookOutput === 'cell-1',
      ),
    )
  })

  test('keeps indented stream output inside the static output panel', async () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [
          {
            cell_type: 'code',
            source: 'parser()',
            execution_count: 25,
            outputs: [
              {
                output_type: 'stream',
                name: 'stdout',
                text: ['     S @ 1\n', '\n', '     S => a X @ 0.4\n'],
              },
              {
                output_type: 'execute_result',
                data: { 'text/plain': "[('a d', '0.32')]" },
                execution_count: 25,
              },
            ],
          },
        ],
      }),
      'parser.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'parser.ipynb', {
      runtime: { enabled: true, sourcePath: 'parser.ipynb' },
    })
    assert.match(markdown, /<samp>     S @ 1&#10;&#10;     S =&gt; a X @ 0\.4<\/samp>/)
    const tree = await parseHtmlFragment(markdown)
    const frame = findElement(
      tree,
      node => node.tagName === 'div' && node.properties?.dataNotebookCellFrame === 'cell-1',
    )
    assert(frame)
    const staticOutput = findElement(
      frame,
      node => node.tagName === 'div' && node.properties?.dataNotebookStaticOutput === 'cell-1',
    )
    assert(staticOutput)
    const stdoutPanel = findElement(
      staticOutput,
      node => node.tagName === 'div' && node.properties?.dataNotebookOutputPanel === 'stdout',
    )
    const resultPanel = findElement(
      staticOutput,
      node => node.tagName === 'div' && node.properties?.dataNotebookOutputPanel === 'result',
    )
    assert(stdoutPanel)
    assert(resultPanel)
    const stdout = findElement(stdoutPanel, node =>
      elementClassNames(node).includes('notebook-output-stream-stdout'),
    )
    assert(stdout)
    const samp = findElement(stdout, node => node.tagName === 'samp')
    assert(samp)
    assert.strictEqual(textChild(samp), '     S @ 1\n\n     S => a X @ 0.4')
    assert(!findElement(samp, node => node.tagName === 'pre'))
  })

  test('renders language badges with svg icons and accessible labels', async () => {
    const tree = await parseHtmlFragment(notebookCellLanguageBadge('javascript'))
    const badge = findElement(tree, node =>
      elementClassNames(node).includes('notebook-language-badge-javascript'),
    )

    assert(badge)
    assert.strictEqual(badge.properties?.dataNotebookLanguage, 'javascript')
    assert.strictEqual(badge.properties?.title, 'JavaScript cell')
    const icon = findElement(badge, node =>
      elementClassNames(node).includes('notebook-language-icon'),
    )
    assert(icon)
    assert.strictEqual(icon.properties?.ariaHidden, 'true')
    assert(findElement(icon, node => elementClassNames(node).includes('notebook-language-svg')))
    const label = findElement(badge, node =>
      elementClassNames(node).includes('notebook-language-label'),
    )
    assert(label)
    assert.strictEqual(textChild(label), 'JavaScript cell')

    const fallbackTree = await parseHtmlFragment(notebookCellLanguageBadge('nixlang'))
    const fallbackBadge = findElement(fallbackTree, node =>
      elementClassNames(node).includes('notebook-language-badge-nixlang'),
    )
    assert(fallbackBadge)
    assert(
      !findElement(fallbackBadge, node =>
        elementClassNames(node).includes('notebook-language-svg'),
      ),
    )
    const fallbackText = findElement(fallbackBadge, node =>
      elementClassNames(node).includes('notebook-language-text'),
    )
    assert(fallbackText)
    assert.strictEqual(textChild(fallbackText), 'ni')
    const fallbackLabel = findElement(fallbackBadge, node =>
      elementClassNames(node).includes('notebook-language-label'),
    )
    assert(fallbackLabel)
    assert.strictEqual(textChild(fallbackLabel), 'nixlang cell')
  })

  test('renders one vim action with centered icon geometry', async () => {
    const tree = await parseHtmlFragment(
      notebookCellActions({
        id: 'cell-1',
        source: 'print("hi")',
        language: 'python',
        executionIndex: null,
      }),
    )
    const vimButtons = findElements(tree, node => node.properties?.dataNotebookVimCell === 'cell-1')

    assert.strictEqual(vimButtons.length, 1)
    const icon = findElement(vimButtons[0], node =>
      elementClassNames(node).includes('notebook-vim-icon'),
    )
    assert(icon)
    assert.strictEqual(icon.properties?.viewBox, '0 0 602 734')
  })

  test('adds anchor ids to runtime cell frames', () => {
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

    assert.match(
      markdown,
      /class="notebook-code-cell" data-notebook-cell-frame="cell-1" id="cell-1"/,
    )
  })

  test('builds scoped runtime payload nodes for a transcluded notebook cell', () => {
    const sourceCell: Element = {
      type: 'element',
      tagName: 'div',
      properties: { className: ['notebook-code-cell'], dataNotebookCellFrame: 'cell-1' },
      children: [
        {
          type: 'element',
          tagName: 'pre',
          properties: {},
          children: [{ type: 'text', value: 'x = 1' }],
        },
      ],
    }
    const sourceTree: HtmlRoot = {
      type: 'root',
      children: [
        sourceCell,
        {
          type: 'element',
          tagName: 'script',
          properties: { type: 'application/json', dataNotebookRuntimeData: '' },
          children: [
            {
              type: 'text',
              value: JSON.stringify({
                id: 'source-runtime',
                sourcePath: 'thoughts/runtime.ipynb',
                language: 'python',
                indexUrl: 'https://cdn.example/pyodide/',
                cells: [
                  { id: 'cell-1', source: 'x = 1', language: 'python', executionIndex: null },
                  { id: 'cell-2', source: 'y = 2', language: 'python', executionIndex: 2 },
                ],
              }),
            },
          ],
        },
      ],
    }

    assert.strictEqual(notebookCellRef('#cell-1'), 'cell-1')
    assert.strictEqual(notebookCellRef('#^cell-1'), 'cell-1')
    assert.strictEqual(findNotebookCellFrame(sourceTree, 'cell-1'), sourceCell)

    const nodes = notebookCellRuntimeNodes(sourceTree, {
      slug: 'thoughts/craft' as FullSlug,
      transcludeTarget: 'thoughts/runtime' as FullSlug,
      cellId: 'cell-1',
      count: 0,
    })

    const runtime = nodes[0] as HastElement
    assert.ok(runtime)
    assert(elementClassNames(runtime).includes('notebook-runtime'))
    const script = nodes[1] as HastElement
    assert.strictEqual(script.tagName, 'script')
    assert.strictEqual(script.properties?.dataNotebookRuntimeData, '')
    assert.ok(script)
    const payload = JSON.parse(textChild(script)) as {
      id: string
      toolbar?: boolean
      cells: Array<{ id: string; source: string }>
    }
    assert.notStrictEqual(payload.id, 'source-runtime')
    assert.strictEqual(payload.toolbar, false)
    assert.deepStrictEqual(
      payload.cells.map(cell => ({ id: cell.id, source: cell.source })),
      [{ id: 'cell-1', source: 'x = 1' }],
    )
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

  test('shows cell display languages from magics and writefile targets', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { language_info: { name: 'python' } },
        cells: [
          { cell_type: 'code', source: 'print("hi")' },
          { cell_type: 'code', source: '%%writefile SetXY.go\npackage main' },
          { cell_type: 'code', source: '%%capture output\n%%bash\necho hi' },
          { cell_type: 'code', source: '%%javascript\nconsole.log("hi")' },
        ],
      }),
      'languages.ipynb',
    )

    const markdown = notebookToMarkdown(notebook, 'languages.ipynb')

    assert.match(markdown, /notebook-language-badge-python/)
    assert.match(markdown, /notebook-language-badge-go/)
    assert.match(markdown, /notebook-language-badge-bash/)
    assert.match(markdown, /notebook-language-badge-javascript/)
    assert.match(markdown, /```go\n%%writefile SetXY\.go\npackage main\n```/)
    assert.match(markdown, /```bash\n%%capture output\n%%bash\necho hi\n```/)
    assert.match(markdown, /```javascript\n%%javascript\nconsole\.log\("hi"\)\n```/)
  })

  test('emits runtime controls for javascript notebooks', () => {
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

    assert.match(markdown, /data-notebook-runtime/)
    assert.match(markdown, /data-notebook-run-all/)
    assert.match(markdown, /data-notebook-run-cell="cell-1"/)
    assert.match(markdown, /notebook-language-badge-javascript/)
    assert.match(markdown, /```javascript\nconsole\.log\("hi"\)\n```/)
  })
})
