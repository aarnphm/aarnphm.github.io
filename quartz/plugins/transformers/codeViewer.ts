import type { Code, Html, Root, RootContent } from 'mdast'
import type { Parent } from 'unist'
import { readFile } from 'fs/promises'
import path from 'path'
import { visit, SKIP } from 'unist-util-visit'
import type { Wikilink } from '../../util/wikilinks'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import {
  defaultNotebookPyodideIndexUrl,
  notebookCellActions,
  notebookCellControls,
  notebookCellFrameOpen,
  notebookCellRuntimeOutput,
  notebookRuntimeControls,
  notebookRuntimeDataScript,
  notebookRuntimeId,
  notebookSourceEditor,
  type NotebookRuntimeCell,
  type NotebookRuntimeData,
} from '../../util/notebook-runtime'
import { FilePath } from '../../util/path'

type Options = { exts?: string[]; pyodideIndexUrl?: string }

type ResolvedOptions = { exts: string[]; pyodideIndexUrl: string }

const DEFAULT_EXTS = new Set<string>([
  '.py',
  '.rs',
  '.go',
  '.c',
  '.cu',
  '.cc',
  '.cpp',
  '.h',
  '.hpp',
  '.m',
  '.mm',
  '.java',
  '.kt',
  '.swift',
  '.scala',
  '.ts',
  '.tsx',
  '.js',
  '.jsx',
  '.sh',
  '.bash',
  '.zsh',
  '.fish',
  '.sql',
  '.yaml',
  '.yml',
  '.toml',
  '.json',
  '.mdx',
  '.css',
  '.scss',
  '.hs',
  '.rb',
  '.php',
])

function languageFromExt(ext: string): string | undefined {
  const e = ext.replace(/^\./, '').toLowerCase()
  switch (e) {
    case 'py':
      return 'python'
    case 'ts':
    case 'tsx':
      return e
    case 'js':
    case 'jsx':
      return e
    case 'rs':
      return 'rust'
    case 'go':
      return 'go'
    case 'c':
      return 'c'
    case 'cc':
    case 'cpp':
    case 'hpp':
      return 'cpp'
    case 'h':
      return 'c'
    case 'm':
    case 'mm':
      return 'objective-c'
    case 'java':
      return 'java'
    case 'kt':
      return 'kotlin'
    case 'swift':
      return 'swift'
    case 'scala':
      return 'scala'
    case 'sh':
    case 'bash':
    case 'zsh':
    case 'fish':
      return 'bash'
    case 'sql':
      return 'sql'
    case 'yaml':
    case 'yml':
      return 'yaml'
    case 'toml':
      return 'toml'
    case 'json':
      return 'json'
    case 'mdx':
      return 'mdx'
    case 'css':
    case 'scss':
      return e
    case 'hs':
      return 'haskell'
    case 'rb':
      return 'ruby'
    case 'php':
      return 'php'
    case 'cu':
      return 'cuda'
    default:
      return e
  }
}

function resolveToRelativePath(
  ctx: BuildCtx,
  target: string,
  currentMdRel: string,
): FilePath | null {
  const relDir = path.posix.dirname(currentMdRel)
  const targetBase = path.posix.basename(target)

  const sibling = path.posix.join(relDir, target)
  if (ctx.allFiles.includes(sibling as FilePath)) {
    return sibling as FilePath
  }

  if (target.includes('/') && ctx.allFiles.includes(target as FilePath)) {
    return target as FilePath
  }

  const match = ctx.allFiles.find(fp => path.posix.basename(fp) === targetBase)
  return (match ?? null) as FilePath | null
}

async function readCodeFile(ctx: BuildCtx, resolvedRel: FilePath) {
  const abs = path.posix.join(ctx.argv.directory, resolvedRel)
  const buf = await readFile(abs)
  return buf.toString('utf8')
}

function html(value: string): Html {
  return { type: 'html', value }
}

type CodeNodeData = { codeTranscludeTarget?: unknown; codeTranscludePath?: unknown }

type CodeWithViewerData = Code & { data?: CodeNodeData }

function codeNodeData(node: Code): CodeNodeData {
  const code = node as CodeWithViewerData
  code.data ??= {}
  return code.data
}

function codeTranscludeTarget(node: Code): string | undefined {
  const target = (node as CodeWithViewerData).data?.codeTranscludeTarget
  return typeof target === 'string' ? target : undefined
}

function codeTranscludePath(node: Code): string | undefined {
  const target = (node as CodeWithViewerData).data?.codeTranscludePath
  return typeof target === 'string' ? target : undefined
}

function hasRootChildren(parent: Parent): parent is Parent & { children: RootContent[] } {
  return Array.isArray(parent.children)
}

function sourcePathFromFile(fileData: { relativePath?: unknown }, fallback: string): string {
  return typeof fileData.relativePath === 'string' ? fileData.relativePath : fallback
}

function codeLanguage(node: Code): string {
  return node.lang?.trim().toLowerCase() ?? ''
}

function metaHasShell(meta: string | null | undefined): boolean {
  return /\bshell\b/i.test(meta ?? '')
}

function runsInPythonRuntime(node: Code): boolean {
  const transcluded = codeTranscludePath(node)
  if (transcluded && path.extname(transcluded).toLowerCase() === '.py') return true

  const lang = codeLanguage(node)
  if (lang === 'python-shell' || lang === 'py-shell') return true
  if (lang !== 'python' && lang !== 'py') return false
  return metaHasShell(node.meta)
}

function runtimeCell(id: string, node: Code): NotebookRuntimeCell {
  return { id, source: node.value, language: 'python', executionIndex: null }
}

function runtimePayload(
  sourcePath: string,
  cells: NotebookRuntimeCell[],
  pyodideIndexUrl: string,
): NotebookRuntimeData {
  return {
    id: notebookRuntimeId(`code-viewer:${sourcePath}`),
    sourcePath,
    language: 'python',
    pyodideIndexUrl,
    cells,
  }
}

function runtimeCellNodes(cell: NotebookRuntimeCell, node: Code): RootContent[] {
  return [
    html(notebookCellFrameOpen(cell.id)),
    html(notebookCellControls(cell).join('\n')),
    html(notebookCellActions(cell)),
    html(notebookSourceEditor(cell.id)),
    node,
    html(notebookCellRuntimeOutput(cell.id)),
    html('</div>'),
  ]
}

export const CodeViewer: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts: ResolvedOptions = {
    exts: userOpts?.exts ?? Array.from(DEFAULT_EXTS),
    pyodideIndexUrl: userOpts?.pyodideIndexUrl ?? defaultNotebookPyodideIndexUrl,
  }
  const exts = new Set(opts.exts.map(e => e.toLowerCase()))

  return {
    name: 'CodeViewer',
    markdownPlugins(ctx) {
      return [
        () => {
          return async (tree: Root, file) => {
            visit(
              tree,
              'wikilink',
              (node: unknown, index: number | undefined, parent: Parent | undefined) => {
                if (index === undefined || !parent) return
                const wikilinkNode = node as unknown as Wikilink
                const data = wikilinkNode.data?.wikilink
                if (!data?.embed) return

                const fp = (data.target ?? '').trim()
                if (!fp) return

                const ext = path.extname(fp).toLowerCase()
                if (!ext || !exts.has(ext)) return

                const lang = languageFromExt(ext)
                const base = path.posix.basename(fp)
                const codeNode: Code = { type: 'code', lang, meta: `title="${base}"`, value: '' }

                codeNodeData(codeNode).codeTranscludeTarget = fp

                if (!hasRootChildren(parent)) return
                parent.children.splice(index, 1, codeNode)
                return [SKIP, index]
              },
            )

            const promises: Promise<void>[] = []
            visit(tree, 'code', (node: Code) => {
              const target = codeTranscludeTarget(node)
              if (!target) return
              const currentRel = sourcePathFromFile(file.data, file.path)
              const resolved = resolveToRelativePath(ctx, target, currentRel)
              if (!resolved) return
              const titleBase = path.posix.basename(resolved)
              const ext = path.extname(resolved)
              node.lang = languageFromExt(ext)
              node.meta = node.meta
                ? `${node.meta} path="${resolved}"`
                : `title="${titleBase}" path="${resolved}"`
              codeNodeData(node).codeTranscludePath = resolved

              promises.push(
                readCodeFile(ctx, resolved).then(content => {
                  node.value = content
                  const deps: string[] = (file.data.codeDependencies as string[] | undefined) ?? []
                  if (!deps.includes(resolved)) {
                    file.data.codeDependencies = [...deps, resolved]
                  }
                }),
              )
            })
            await Promise.all(promises)

            const sourcePath = sourcePathFromFile(file.data, file.path)
            const cells: NotebookRuntimeCell[] = []
            let insertedRuntime = false

            visit(tree, 'code', (node: Code, index, parent) => {
              if (index === undefined || !parent || !hasRootChildren(parent)) return
              if (!runsInPythonRuntime(node)) return
              const cell = runtimeCell(`code-cell-${cells.length + 1}`, node)
              cells.push(cell)
              const nodes = runtimeCellNodes(cell, node)
              if (!insertedRuntime) {
                nodes.unshift(
                  ...notebookRuntimeControls(
                    runtimePayload(sourcePath, [], opts.pyodideIndexUrl),
                  ).map(html),
                )
                insertedRuntime = true
              }
              parent.children.splice(index, 1, ...nodes)
              return [SKIP, index + nodes.length]
            })

            if (cells.length > 0) {
              tree.children.push(
                html(
                  notebookRuntimeDataScript(
                    runtimePayload(sourcePath, cells, opts.pyodideIndexUrl),
                  ),
                ),
              )
            }
          }
        },
      ]
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    codeDependencies: string[]
  }
}
