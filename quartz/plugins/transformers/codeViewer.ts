import type { Element, Root as HastRoot, RootContent as HastRootContent } from 'hast'
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
  buildBlobUrl,
  fetchGithubFileLines,
  joinFileLines,
  lineRangeLabel,
  lineRangeMeta,
  parseGithubBlobUrl,
  parseLineRange,
  type GithubBlobRef,
  type LineRange,
} from '../../util/github-embed'
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

export function languageFromExt(ext: string): string | undefined {
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

interface GithubCodeTransclude {
  ref: GithubBlobRef
  range: LineRange | null
  anchorText?: string
}

type CodeNodeData = {
  codeTranscludeTarget?: unknown
  githubCodeTransclude?: GithubCodeTransclude
  hProperties?: Record<string, string>
}

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

function githubCodeTransclude(node: Code): GithubCodeTransclude | undefined {
  return (node as CodeWithViewerData).data?.githubCodeTransclude
}

function hasRootChildren(parent: Parent): parent is Parent & { children: RootContent[] } {
  return Array.isArray(parent.children)
}

function sourcePathFromFile(fileData: { relativePath?: unknown }, fallback: string): string {
  return typeof fileData.relativePath === 'string' ? fileData.relativePath : fallback
}

function quotedMetaValue(value: string): string {
  return value.replace(/"/g, "'")
}

function titleMeta(title: string): string {
  return `title="${quotedMetaValue(title)}"`
}

function githubCodeTitle(ref: GithubBlobRef, range: LineRange | null): string {
  const base = path.posix.basename(ref.filePath)
  const label = lineRangeLabel(range)
  return label
    ? `${ref.owner}/${ref.repo} · ${base}:${label}`
    : `${ref.owner}/${ref.repo} · ${base}`
}

function githubCodeNode(
  target: string,
  anchorText: string | undefined,
  exts: Set<string>,
): Code | null {
  const ref = parseGithubBlobUrl(target)
  if (!ref) return null

  const ext = path.extname(ref.filePath).toLowerCase()
  if (!ext || !exts.has(ext)) return null

  const range = parseLineRange(anchorText)
  const lang = languageFromExt(ext)
  const meta = [titleMeta(githubCodeTitle(ref, range)), lineRangeMeta(range)]
    .filter(part => part.length > 0)
    .join(' ')
  const node: Code = { type: 'code', lang, meta, value: '' }
  const data = codeNodeData(node)
  data.githubCodeTransclude = { ref, range, anchorText }
  data.hProperties = { 'data-github-href': buildBlobUrl(ref, anchorText) }
  return node
}

function isHastElement(node: unknown): node is Element {
  return typeof node === 'object' && node !== null && 'type' in node && node.type === 'element'
}

function hasHastChildren(parent: unknown): parent is { children: HastRootContent[] } {
  return (
    typeof parent === 'object' &&
    parent !== null &&
    'children' in parent &&
    Array.isArray(parent.children)
  )
}

function stringProperty(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function githubHrefForPre(node: Element): string | undefined {
  if (node.tagName !== 'pre') return undefined
  for (const child of node.children) {
    if (!isHastElement(child) || child.tagName !== 'code') continue
    return stringProperty(
      child.properties?.['data-github-href'] ?? child.properties?.dataGithubHref,
    )
  }
  return undefined
}

function hasClass(node: Element, className: string): boolean {
  const classes = node.properties.className
  return Array.isArray(classes) && classes.includes(className)
}

function githubCodeEmbedWrapper(pre: Element, href: string): Element {
  return {
    type: 'element',
    tagName: 'div',
    properties: { className: ['github-code-embed'], 'data-github-href': href },
    children: [pre],
  }
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

type CodeRuntimeGroup = CodeRuntimeBinding & { cells: NotebookRuntimeCell[]; inserted: boolean }

type RuntimeCellNode = Html | Code

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

function runtimeCellNodes(cell: NotebookRuntimeCell, node: Code): RuntimeCellNode[] {
  return [
    html(notebookCellFrameOpen(cell.id, cell.displayLanguage ?? cell.language)),
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

                const remoteCodeNode = githubCodeNode(fp, data.anchorText, exts)
                if (remoteCodeNode) {
                  if (!hasRootChildren(parent)) return
                  parent.children.splice(index, 1, remoteCodeNode)
                  return [SKIP, index]
                }

                if (/^https?:\/\//i.test(fp)) return
                const ext = path.extname(fp).toLowerCase()
                if (!ext || !exts.has(ext)) return

                const lang = languageFromExt(ext)
                const base = path.posix.basename(fp)
                const codeNode: Code = { type: 'code', lang, meta: titleMeta(base), value: '' }

                codeNodeData(codeNode).codeTranscludeTarget = fp

                if (!hasRootChildren(parent)) return
                parent.children.splice(index, 1, codeNode)
                return [SKIP, index]
              },
            )

            const promises: Promise<void>[] = []
            visit(tree, 'code', (node: Code) => {
              const remote = githubCodeTransclude(node)
              if (remote) {
                promises.push(
                  fetchGithubFileLines(remote.ref.rawUrl).then(lines => {
                    node.value = joinFileLines(lines)
                  }),
                )
                return
              }

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
                : `${titleMeta(titleBase)} path="${resolved}"`

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
            const runtimeGroups = new Map<string, CodeRuntimeGroup>()
            let runtimeCellCount = 0

            visit(tree, 'code', (node: Code, index, parent) => {
              if (index === undefined || !parent || !hasRootChildren(parent)) return
              const binding = runtimeBindingForCode(node)
              if (!binding) return
              const group = runtimeGroups.get(binding.backend.name) ?? {
                ...binding,
                cells: [],
                inserted: false,
              }
              runtimeGroups.set(binding.backend.name, group)
              runtimeCellCount += 1
              const cell = runtimeCell(`code-cell-${runtimeCellCount}`, node, binding.backend)
              group.cells.push(cell)
              const nodes = runtimeCellNodes(cell, node)
              if (!group.inserted) {
                const indexUrl = resolveIndexUrl(group.backend, opts.indexUrls)
                nodes.unshift(
                  ...notebookRuntimeControls(
                    runtimePayload(sourcePath, [], group.backend, indexUrl, group.options),
                  ).map(html),
                )
                group.inserted = true
              }
              parent.children.splice(index, 1, ...nodes)
              return [SKIP, index + nodes.length]
            })

            for (const group of runtimeGroups.values()) {
              if (group.cells.length === 0) continue
              const indexUrl = resolveIndexUrl(group.backend, opts.indexUrls)
              tree.children.push(
                html(
                  notebookRuntimeDataScript(
                    runtimePayload(sourcePath, group.cells, group.backend, indexUrl, group.options),
                  ),
                ),
              )
            }
          }
        },
      ]
    },
    htmlPlugins() {
      return [
        () => (tree: HastRoot) => {
          visit(tree, 'element', (node: Element, index, parent) => {
            if (index === undefined || !hasHastChildren(parent)) return
            if (isHastElement(parent) && hasClass(parent, 'github-code-embed')) return
            const href = githubHrefForPre(node)
            if (!href) return
            parent.children[index] = githubCodeEmbedWrapper(node, href)
            return [SKIP, index]
          })
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

declare module 'mdast' {
  interface CodeData {
    githubCodeTransclude?: GithubCodeTransclude
  }
}
