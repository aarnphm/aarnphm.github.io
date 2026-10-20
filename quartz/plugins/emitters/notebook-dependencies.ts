import fs from 'node:fs/promises'
import path from 'node:path'
import { backendFor } from '../../runtime/notebook/backend'
import { ChangeEvent } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { isNotebookParseError } from '../../util/notebook/types'
import { FilePath, joinSegments } from '../../util/path'

type NotebookDependencies = { imports: Set<FilePath>; assets: Set<FilePath> }
export type NotebookDependencyRecord = NotebookDependencies & { signature: string }
type NotebookDependencyCacheEntry = {
  mtimeMs: number
  size: number
  record: NotebookDependencyRecord
}

declare global {
  var __quartzNotebookDependencyCache: Map<FilePath, NotebookDependencyCacheEntry> | undefined
}

const htmlResourceSrcPattern =
  /<(?:img|video|audio|iframe)\b[^>]*\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi
const markdownImageTargetPattern = /!\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)[^)]*\)/g

export function isNotebookPath(fp: FilePath): boolean {
  return path.extname(fp) === '.ipynb'
}

function localNotebookResourcePath(raw: string, fp: FilePath): FilePath | undefined {
  const trimmed = raw.trim()
  const value = trimmed.startsWith('<') && trimmed.endsWith('>') ? trimmed.slice(1, -1) : trimmed
  if (
    !value ||
    value.startsWith('#') ||
    value.startsWith('/') ||
    /^[A-Za-z][A-Za-z0-9+.-]*:/.test(value)
  ) {
    return undefined
  }

  const match = value.match(/^([^?#]*)([?#].*)?$/)
  let pathname = match?.[1] ?? value
  if (!pathname) return undefined
  try {
    pathname = decodeURI(pathname)
  } catch {}

  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(fp), pathname))
  if (resolved === '.' || resolved.startsWith('../')) return undefined
  return resolved as FilePath
}

function notebookDependencies(raw: string, fp: FilePath): NotebookDependencies {
  const doc = parseNotebookDoc(raw, fp)
  const imports = new Set<FilePath>()
  const assets = new Set<FilePath>()
  if (isNotebookParseError(doc)) return { imports, assets }
  const dir = path.posix.dirname(fp)
  const backend = backendFor(doc.language)
  const moduleResolver = backend?.moduleResolver

  for (const cell of doc.cells) {
    const source = cell.source
    if (cell.cellType === 'code' && moduleResolver) {
      for (const name of moduleResolver.importNames(source)) {
        imports.add(path.posix.join(dir, `${name}.ipynb`) as FilePath)
      }
    }
    if (cell.cellType !== 'markdown') continue

    for (const match of source.matchAll(htmlResourceSrcPattern)) {
      const candidate = match[1] ?? match[2] ?? match[3]
      if (candidate === undefined) continue
      const dependency = localNotebookResourcePath(candidate, fp)
      if (dependency !== undefined) assets.add(dependency)
    }

    for (const match of source.matchAll(markdownImageTargetPattern)) {
      const dependency = localNotebookResourcePath(match[1], fp)
      if (dependency !== undefined) assets.add(dependency)
    }
  }

  return { imports, assets }
}

export async function readNotebookDependencies(
  ctx: BuildCtx,
  fp: FilePath,
): Promise<NotebookDependencyRecord> {
  const src = joinSegments(ctx.argv.directory, fp) as FilePath
  const cache = (globalThis.__quartzNotebookDependencyCache ??= new Map())
  try {
    const stat = await fs.stat(src)
    const cached = cache.get(src)
    if (cached && cached.mtimeMs === stat.mtimeMs && cached.size === stat.size) {
      return cached.record
    }
    const raw = await fs.readFile(src, 'utf8')
    const deps = notebookDependencies(raw, fp)
    const signature = `${stat.mtimeMs}:${stat.size}`
    const record: NotebookDependencyRecord = {
      imports: deps.imports,
      assets: deps.assets,
      signature,
    }
    cache.set(src, { mtimeMs: stat.mtimeMs, size: stat.size, record })
    return record
  } catch {
    cache.delete(src)
    return { imports: new Set(), assets: new Set(), signature: 'missing' }
  }
}

export async function affectedNotebookFiles(
  ctx: BuildCtx,
  fps: FilePath[],
  changeEvents: ChangeEvent[],
): Promise<FilePath[]> {
  const available = new Set(fps)
  const changedPaths = new Set(changeEvents.map(event => event.path))
  const changedNotebooks = new Set(
    changeEvents.filter(event => isNotebookPath(event.path)).map(event => event.path),
  )
  const affected = new Set<FilePath>()

  for (const event of changeEvents) {
    if (isNotebookPath(event.path) && event.type !== 'delete' && available.has(event.path)) {
      affected.add(event.path)
    }
  }

  const dependencyPairs = await Promise.all(
    fps.map(async fp => [fp, await readNotebookDependencies(ctx, fp)] as const),
  )
  const reverseImports = new Map<FilePath, Set<FilePath>>()
  for (const [fp, deps] of dependencyPairs) {
    for (const target of deps.imports) {
      if (!available.has(target)) continue
      const dependents = reverseImports.get(target) ?? new Set<FilePath>()
      dependents.add(fp)
      reverseImports.set(target, dependents)
    }
    for (const target of deps.assets) {
      if (changedPaths.has(target)) affected.add(fp)
    }
  }

  const queue = [...changedNotebooks]
  const seen = new Set<FilePath>()
  while (queue.length > 0) {
    const current = queue.shift()
    if (current === undefined) break
    if (seen.has(current)) continue
    seen.add(current)
    for (const dependent of reverseImports.get(current) ?? []) {
      if (!affected.has(dependent)) {
        affected.add(dependent)
        queue.push(dependent)
      }
    }
  }

  return fps.filter(fp => affected.has(fp))
}
