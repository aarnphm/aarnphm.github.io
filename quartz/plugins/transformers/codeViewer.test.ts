import type { Code, Html, Root, RootContent } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import assert from 'node:assert'
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import os from 'node:os'
import nodePath from 'node:path'
import test, { describe } from 'node:test'
import { VFile } from 'vfile'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath } from '../../util/path'
import { notebookToMarkdownChunks } from '../../util/notebook/markdown'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { isNotebookParseError } from '../../util/notebook/types'
import { CodeViewer } from './codeViewer'

type Transformer = (tree: Root, file: VFile) => Promise<void> | void

type RuntimePayload = {
  id: string
  sourcePath: string
  language: string
  toolbar?: boolean
  debug?: boolean
  vimMode?: boolean
  cells: Array<{ id: string; source: string; language: string }>
}

function buildCtx(root: string, allFiles: FilePath[] = []): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: root,
      verbose: false,
      output: nodePath.join(root, '.out'),
      serve: false,
      watch: false,
      port: 0,
      wsPort: 0,
      force: false,
    },
    cfg: undefined as never,
    allSlugs: [],
    allFiles,
    incremental: false,
  }
}

function vfile(root: string, rel: string): VFile {
  const file = new VFile({ path: nodePath.join(root, rel), value: '' })
  file.data.relativePath = rel as FilePath
  return file
}

async function runCodeViewer(tree: Root, file: VFile, ctx: BuildCtx) {
  const plugins = CodeViewer().markdownPlugins?.(ctx)
  assert.ok(plugins)
  const createTransformer = plugins[0] as () => Transformer
  await createTransformer()(tree, file)
}

function collectHtml(node: Root | RootContent): string[] {
  if (node.type === 'html') return [(node as Html).value]
  if (!('children' in node) || !Array.isArray(node.children)) return []
  return node.children.flatMap(child => collectHtml(child as RootContent))
}

function collectCode(node: Root | RootContent): Code[] {
  if (node.type === 'code') return [node as Code]
  if (!('children' in node) || !Array.isArray(node.children)) return []
  return node.children.flatMap(child => collectCode(child as RootContent))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isRuntimePayload(value: unknown): value is RuntimePayload {
  if (!isRecord(value)) return false
  return (
    typeof value.id === 'string' &&
    typeof value.sourcePath === 'string' &&
    typeof value.language === 'string' &&
    Array.isArray(value.cells)
  )
}

function runtimePayload(tree: Root): RuntimePayload {
  const [payload] = runtimePayloads(tree)
  assert.ok(payload)
  return payload
}

function runtimePayloads(tree: Root): RuntimePayload[] {
  const html = collectHtml(tree).join('\n')
  const matches = html.matchAll(
    /<script type="application\/json" data-notebook-runtime-data>([\s\S]*?)<\/script>/g,
  )
  return Array.from(matches, match => {
    const parsed: unknown = JSON.parse(match[1])
    assert.ok(isRuntimePayload(parsed))
    return parsed
  })
}

describe('code viewer runtime cells', () => {
  test('wraps python shell fences without a global runtime toolbar', async () => {
    const root = await mkdtemp(nodePath.join(os.tmpdir(), 'quartz-code-viewer-'))
    const tree: Root = {
      type: 'root',
      children: [{ type: 'code', lang: 'python', meta: 'shell', value: 'print("hi")' }],
    }

    await runCodeViewer(tree, vfile(root, 'notes/page.md'), buildCtx(root))

    const html = collectHtml(tree).join('\n')
    assert.match(html, /data-notebook-runtime=/)
    assert.doesNotMatch(html, /notebook-runtime-toolbar/)
    assert.doesNotMatch(html, /data-notebook-debug/)
    assert.doesNotMatch(html, /data-notebook-vim-mode/)
    assert.match(html, /data-notebook-run-cell="code-cell-1"/)

    const payload = runtimePayload(tree)
    assert.match(payload.id, /^notebook-runtime-/)
    assert.strictEqual(payload.sourcePath, 'notes/page.md')
    assert.strictEqual(payload.language, 'python')
    assert.strictEqual(payload.toolbar, false)
    assert.strictEqual(payload.debug, true)
    assert.strictEqual(payload.vimMode, true)
    assert.deepStrictEqual(
      payload.cells.map(cell => cell.source),
      ['print("hi")'],
    )
  })

  test('wraps parser-produced python shell fences inside collapsed callouts', async () => {
    const root = await mkdtemp(nodePath.join(os.tmpdir(), 'quartz-code-viewer-'))
    const tree = fromMarkdown(`> [!info]- test\n>\n> \`\`\`python shell\n> print("hi")\n> \`\`\`\n`)

    await runCodeViewer(tree, vfile(root, 'notes/page.md'), buildCtx(root))

    const html = collectHtml(tree).join('\n')
    assert.match(html, /data-notebook-runtime=/)
    assert.match(html, /data-notebook-cell="code-cell-1"/)

    const payload = runtimePayload(tree)
    assert.deepStrictEqual(
      payload.cells.map(cell => cell.source),
      ['print("hi")'],
    )
  })

  test('wraps shell fences emitted from notebook markdown cells', async () => {
    const root = await mkdtemp(nodePath.join(os.tmpdir(), 'quartz-code-viewer-'))
    const notebook = parseNotebookDoc(
      JSON.stringify({
        cells: [
          {
            cell_type: 'markdown',
            source: ['```haskell shell\n', 'main = putStrLn "hi"\n', '```\n'],
          },
        ],
      }),
      'notes/lecture.ipynb',
    )
    if (isNotebookParseError(notebook)) throw new Error(notebook.reason)
    const shellChunk = notebookToMarkdownChunks(notebook, 'notes/lecture.ipynb').find(chunk =>
      chunk.includes('```haskell shell'),
    )
    assert.ok(shellChunk)
    const tree = fromMarkdown(shellChunk)

    await runCodeViewer(tree, vfile(root, 'notes/lecture.ipynb'), buildCtx(root))

    const html = collectHtml(tree).join('\n')
    assert.match(html, /data-notebook-runtime=/)
    assert.match(html, /notebook-language-badge-haskell/)
    assert.match(html, /data-notebook-language="haskell"/)

    const payload = runtimePayload(tree)
    assert.strictEqual(payload.sourcePath, 'notes/lecture.ipynb')
    assert.strictEqual(payload.language, 'haskell')
    assert.strictEqual(payload.toolbar, false)
    assert.strictEqual(payload.debug, true)
    assert.strictEqual(payload.vimMode, true)
    assert.deepStrictEqual(
      payload.cells.map(cell => ({ id: cell.id, source: cell.source, language: cell.language })),
      [{ id: 'code-cell-1', source: 'main = putStrLn "hi"', language: 'haskell' }],
    )
  })

  test('wraps multiple shell backends on the same page', async () => {
    const root = await mkdtemp(nodePath.join(os.tmpdir(), 'quartz-code-viewer-'))
    const tree = fromMarkdown(
      [
        '```python shell',
        'print("hi")',
        '```',
        '',
        '```javascript shell',
        'console.log("hi")',
        '```',
        '',
        '```ocaml shell',
        'print_endline "hi"',
        '```',
        '',
        '```go shell',
        'package main',
        'import "fmt"',
        'func main() { fmt.Println("hi") }',
        '```',
        '',
        '```rust shell',
        'println!("hi");',
        '```',
      ].join('\n'),
    )

    await runCodeViewer(tree, vfile(root, 'notes/page.md'), buildCtx(root))

    const payloads = runtimePayloads(tree)
    assert.deepStrictEqual(
      payloads.map(payload => ({
        language: payload.language,
        cells: payload.cells.map(cell => cell.id),
      })),
      [
        { language: 'python', cells: ['code-cell-1'] },
        { language: 'javascript', cells: ['code-cell-2'] },
        { language: 'ocaml', cells: ['code-cell-3'] },
        { language: 'go', cells: ['code-cell-4'] },
        { language: 'rust', cells: ['code-cell-5'] },
      ],
    )
    const html = collectHtml(tree).join('\n')
    assert.match(html, /notebook-language-badge-python/)
    assert.match(html, /notebook-language-badge-javascript/)
    assert.match(html, /notebook-language-badge-ocaml/)
    assert.match(html, /notebook-language-badge-go/)
    assert.match(html, /notebook-language-badge-rust/)
  })

  test('lets python shell meta disable debug and vim defaults', async () => {
    const root = await mkdtemp(nodePath.join(os.tmpdir(), 'quartz-code-viewer-'))
    const tree: Root = {
      type: 'root',
      children: [
        { type: 'code', lang: 'python', meta: 'shell vim=false debug=false', value: 'print("hi")' },
      ],
    }

    await runCodeViewer(tree, vfile(root, 'notes/page.md'), buildCtx(root))

    const payload = runtimePayload(tree)
    assert.strictEqual(payload.toolbar, false)
    assert.strictEqual(payload.debug, false)
    assert.strictEqual(payload.vimMode, false)
  })

  test('leaves ordinary python fences static', async () => {
    const root = await mkdtemp(nodePath.join(os.tmpdir(), 'quartz-code-viewer-'))
    const tree: Root = {
      type: 'root',
      children: [{ type: 'code', lang: 'python', value: 'print("hi")' }],
    }

    await runCodeViewer(tree, vfile(root, 'notes/page.md'), buildCtx(root))

    assert.strictEqual(collectHtml(tree).length, 0)
    assert.deepStrictEqual(
      collectCode(tree).map(node => node.value),
      ['print("hi")'],
    )
  })

  test('leaves ordinary native-language fences static', async () => {
    const root = await mkdtemp(nodePath.join(os.tmpdir(), 'quartz-code-viewer-'))
    const tree: Root = {
      type: 'root',
      children: [
        { type: 'code', lang: 'haskell', value: 'main = putStrLn "hi"' },
        { type: 'code', lang: 'ocaml', value: 'print_endline "hi"' },
        { type: 'code', lang: 'go', value: 'package main' },
      ],
    }

    await runCodeViewer(tree, vfile(root, 'notes/page.md'), buildCtx(root))

    assert.strictEqual(collectHtml(tree).length, 0)
    assert.deepStrictEqual(
      collectCode(tree).map(node => node.lang),
      ['haskell', 'ocaml', 'go'],
    )
  })

  test('leaves transcluded python files static', async () => {
    const root = await mkdtemp(nodePath.join(os.tmpdir(), 'quartz-code-viewer-'))
    await mkdir(nodePath.join(root, 'notes'), { recursive: true })
    await writeFile(nodePath.join(root, 'notes', 'script.py'), 'print("from file")\n')

    const tree: Root = {
      type: 'root',
      children: [
        {
          type: 'wikilink',
          data: { wikilink: { embed: true, target: 'script.py' } },
        } as unknown as RootContent,
      ],
    }
    const file = vfile(root, 'notes/page.md')

    await runCodeViewer(tree, file, buildCtx(root, ['notes/script.py' as FilePath]))

    const [code] = collectCode(tree)
    assert.strictEqual(code.lang, 'python')
    assert.strictEqual(code.value, 'print("from file")\n')
    assert.deepStrictEqual(file.data.codeDependencies, ['notes/script.py'])
    assert.strictEqual(collectHtml(tree).length, 0)
  })
})
