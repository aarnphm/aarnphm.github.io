import type { Code, Html, Root, RootContent } from 'mdast'
import type { Parent } from 'unist'
import { readFile } from 'fs/promises'
import path from 'path'
import { visit, SKIP } from 'unist-util-visit'
import type { NotebookRuntimeCell, NotebookRuntimeData } from '../../runtime/notebook/types'
import type { Wikilink } from '../../util/wikilinks'
import {
  backendFor,
  backendForShellMagic,
  type ExecutableLanguageBackend,
} from '../../runtime/notebook/backend'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import {
  notebookCellActions,
  notebookCellControls,
  notebookCellFrameOpen,
  notebookCellRuntimeOutput,
  notebookRuntimeControls,
  notebookRuntimeDataScript,
  notebookSourceEditor,
} from '../../util/notebook/cell-html'
import { notebookId } from '../../util/notebook/identity'
import { FilePath } from '../../util/path'
import '../../runtime/notebook/registry'

type Options = { exts?: string[]; indexUrls?: Readonly<Record<string, string>> }

type ResolvedOptions = { exts: string[]; indexUrls: Map<string, string> }

const DEFAULT_EXTS = new Set<string>([
  '.py',
  '.mojo',
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
    case 'mojo':
      return 'mojo'
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

type CodeNodeData = { codeTranscludeTarget?: unknown }

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

function hasRootChildren(parent: Parent): parent is Parent & { children: RootContent[] } {
  return Array.isArray(parent.children)
}

function sourcePathFromFile(fileData: { relativePath?: unknown }, fallback: string): string {
  return typeof fileData.relativePath === 'string' ? fileData.relativePath : fallback
}

function codeLanguage(node: Code): string {
  return node.lang?.trim().toLowerCase() ?? ''
}

function metaWords(meta: string | null | undefined): string[] {
  if (!meta) return []
  const words: string[] = []
  let word = ''
  let quote = ''
  let escaped = false
  for (const char of meta) {
    if (escaped) {
      word += char
      escaped = false
      continue
    }
    if (char === '\\') {
      escaped = true
      continue
    }
    if (quote.length > 0) {
      if (char === quote) {
        quote = ''
      } else {
        word += char
      }
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (/\s/.test(char)) {
      if (word.length > 0) {
        words.push(word)
        word = ''
      }
      continue
    }
    word += char
  }
  if (word.length > 0) words.push(word)
  return words
}

function metaHasShell(words: string[]): boolean {
  return words.some(word => word.toLowerCase() === 'shell')
}

function metaBooleanOption(words: string[], key: string, fallback: boolean): boolean {
  const prefix = `${key}=`
  for (const word of words) {
    const lower = word.toLowerCase()
    if (!lower.startsWith(prefix)) continue
    const value = lower.slice(prefix.length)
    if (value === 'true') return true
    if (value === 'false') return false
  }
  return fallback
}

type CodeRuntimeOptions = Pick<NotebookRuntimeData, 'toolbar' | 'debug' | 'vimMode'>

type CodeRuntimeBinding = { backend: ExecutableLanguageBackend; options: CodeRuntimeOptions }

function shellRuntimeOptions(words: string[]): CodeRuntimeOptions {
  return {
    toolbar: false,
    debug: metaBooleanOption(words, 'debug', true),
    vimMode: metaBooleanOption(words, 'vim', true),
  }
}

function runtimeBindingForCode(node: Code): CodeRuntimeBinding | undefined {
  const lang = codeLanguage(node)
  const words = metaWords(node.meta)
  const shellBackend = backendForShellMagic(lang)
  if (shellBackend) return { backend: shellBackend, options: shellRuntimeOptions(words) }
  const fenceBackend = backendFor(lang)
  if (!fenceBackend) return undefined
  if (!metaHasShell(words)) return undefined
  return { backend: fenceBackend, options: shellRuntimeOptions(words) }
}

function runtimeCell(
  id: string,
  node: Code,
  backend: ExecutableLanguageBackend,
): NotebookRuntimeCell {
  return { id, source: node.value, language: backend.name, executionIndex: null }
}

function runtimePayload(
  sourcePath: string,
  cells: NotebookRuntimeCell[],
  backend: ExecutableLanguageBackend,
  indexUrl: string,
  options: CodeRuntimeOptions,
): NotebookRuntimeData {
  const payload: NotebookRuntimeData = {
    id: notebookId(`code-viewer:${backend.name}:${sourcePath}`),
    sourcePath,
    language: backend.name,
    indexUrl,
    cells,
  }
  if (options.toolbar !== undefined) payload.toolbar = options.toolbar
  if (options.debug !== undefined) payload.debug = options.debug
  if (options.vimMode !== undefined) payload.vimMode = options.vimMode
  return payload
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

function resolveIndexUrl(
  backend: ExecutableLanguageBackend,
  overrides: Map<string, string>,
): string {
  return overrides.get(backend.name) ?? backend.defaultIndexUrl ?? ''
}

export const CodeViewer: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts: ResolvedOptions = {
    exts: userOpts?.exts ?? Array.from(DEFAULT_EXTS),
    indexUrls: new Map(Object.entries(userOpts?.indexUrls ?? {})),
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
            let runtimeBinding: CodeRuntimeBinding | undefined
            let insertedRuntime = false

            visit(tree, 'code', (node: Code, index, parent) => {
              if (index === undefined || !parent || !hasRootChildren(parent)) return
              const binding = runtimeBindingForCode(node)
              if (!binding) return
              if (runtimeBinding && binding.backend !== runtimeBinding.backend) return
              if (!runtimeBinding) runtimeBinding = binding
              const cell = runtimeCell(`code-cell-${cells.length + 1}`, node, binding.backend)
              cells.push(cell)
              const nodes = runtimeCellNodes(cell, node)
              if (!insertedRuntime) {
                const indexUrl = resolveIndexUrl(runtimeBinding.backend, opts.indexUrls)
                nodes.unshift(
                  ...notebookRuntimeControls(
                    runtimePayload(
                      sourcePath,
                      [],
                      runtimeBinding.backend,
                      indexUrl,
                      runtimeBinding.options,
                    ),
                  ).map(html),
                )
                insertedRuntime = true
              }
              parent.children.splice(index, 1, ...nodes)
              return [SKIP, index + nodes.length]
            })

            if (cells.length > 0 && runtimeBinding) {
              const indexUrl = resolveIndexUrl(runtimeBinding.backend, opts.indexUrls)
              tree.children.push(
                html(
                  notebookRuntimeDataScript(
                    runtimePayload(
                      sourcePath,
                      cells,
                      runtimeBinding.backend,
                      indexUrl,
                      runtimeBinding.options,
                    ),
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
