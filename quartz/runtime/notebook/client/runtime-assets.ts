import type { ExecutableLanguageBackend, RuntimeModuleResolver } from '../backend'
import type {
  NotebookModule,
  RuntimeAssetRequest,
  RuntimeAssetResult,
  RuntimeDownload,
  RuntimeFileResult,
} from '../kernel'
import type { NotebookRuntimeState } from './runtime-state'
import { showNotebookOutputToast } from './output-tabs'
import { setStatus } from './runtime-chrome'

export function downloadFileName(filename: string): string {
  const name = filename.replaceAll('\\', '/').split('/').filter(Boolean).at(-1)?.trim()
  return name && name !== '.' && name !== '..' ? name : 'notebook-file'
}

export function downloadNotebookFile(
  filename: string,
  contentType: string,
  bytes: ArrayBuffer,
): string {
  const name = downloadFileName(filename)
  const url = URL.createObjectURL(new Blob([bytes], { type: contentType }))
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.style.display = 'none'
  document.body.append(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000)
  return name
}

export async function notebookModulesFor(
  state: NotebookRuntimeState,
  source: string,
  backend: ExecutableLanguageBackend,
): Promise<NotebookModule[]> {
  const resolver = backend.moduleResolver
  if (!resolver) return []
  const seen = new Set<string>()
  const modules = new Map<string, NotebookModule>()
  await collectNotebookModules(state, source, resolver, seen, modules)
  return [...modules.values()]
}

export async function collectNotebookModules(
  state: NotebookRuntimeState,
  source: string,
  resolver: RuntimeModuleResolver,
  seen: Set<string>,
  modules: Map<string, NotebookModule>,
): Promise<void> {
  for (const name of resolver.importNames(source)) {
    if (seen.has(name)) continue
    seen.add(name)
    const module = await fetchNotebookModule(state, name, resolver)
    if (!module) continue
    modules.set(name, module)
    await collectNotebookModules(state, module.source, resolver, seen, modules)
  }
}

export async function fetchNotebookModule(
  state: NotebookRuntimeState,
  name: string,
  resolver: RuntimeModuleResolver,
): Promise<NotebookModule | null> {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) return null
  const cached = state.moduleCache.get(name)
  if (cached !== undefined) return cached
  const pending = state.moduleFetches.get(name)
  if (pending) return await pending
  const request = loadNotebookModule(state, name, resolver)
  state.moduleFetches.set(name, request)
  try {
    const module = await request
    state.moduleCache.set(name, module)
    return module
  } finally {
    state.moduleFetches.delete(name)
  }
}

export async function loadNotebookModule(
  state: NotebookRuntimeState,
  name: string,
  resolver: RuntimeModuleResolver,
): Promise<NotebookModule | null> {
  if (state.importableModules !== undefined && !state.importableModules.has(name)) return null
  const url = new URL(`${name}.ipynb`, window.location.href)
  const baseDir = new URL('.', window.location.href)
  if (url.origin !== window.location.origin || !url.pathname.startsWith(baseDir.pathname)) {
    throw new Error(`blocked non-sibling notebook import: ${name}`)
  }
  const relative = url.pathname.slice(baseDir.pathname.length)
  if (relative.includes('/')) {
    throw new Error(`blocked nested notebook import: ${name}`)
  }
  const response = await fetch(url)
  if (response.status === 404) return null
  if (!response.ok) {
    throw new Error(`failed to fetch notebook import ${name}.ipynb: ${response.status}`)
  }
  const source = resolver.moduleSource(await response.text(), `${name}.ipynb`)
  return { name, sourcePath: url.pathname, source }
}

export function downloadRuntimeFile(state: NotebookRuntimeState, message: RuntimeDownload) {
  const name = downloadNotebookFile(message.filename, message.contentType, message.bytes)
  setStatus(state, `downloaded ${name}`)
  showNotebookOutputToast(`downloaded ${name}`)
}

export async function resolveAsset(
  message: RuntimeAssetRequest,
  runtimeFile: (path: string) => Promise<RuntimeFileResult | undefined>,
): Promise<RuntimeAssetResult> {
  const fallback = (error: unknown): RuntimeAssetResult => ({
    runtimeId: message.runtimeId,
    assetId: message.assetId,
    ok: false,
    status: 404,
    statusText: 'Not Found',
    contentType: 'text/plain',
    error: error instanceof Error ? error.message : String(error),
  })
  let relative = ''
  let assetUrl = ''
  try {
    const base = new URL(window.location.href)
    const baseDir = new URL('.', base)
    const url = new URL(message.url, base)
    assetUrl = url.href
    relative = url.pathname.slice(baseDir.pathname.length)
    if (
      url.origin !== window.location.origin ||
      !url.pathname.startsWith(baseDir.pathname) ||
      relative.includes('/')
    ) {
      throw new Error(`blocked non-sibling notebook asset: ${message.url}`)
    }
  } catch (error) {
    return fallback(error)
  }
  try {
    const response = await fetch(assetUrl)
    if (!response.ok) {
      throw new Error(`missing notebook asset: ${relative || message.url}`)
    }
    return {
      runtimeId: message.runtimeId,
      assetId: message.assetId,
      ok: true,
      status: response.status,
      statusText: response.statusText,
      contentType: response.headers.get('content-type') ?? 'application/octet-stream',
      bytes: await response.arrayBuffer(),
    }
  } catch (error) {
    return (
      (await resolveRuntimeFileAsset(message, relative || message.url, runtimeFile)) ??
      fallback(error)
    )
  }
}

export async function resolveRuntimeFileAsset(
  message: RuntimeAssetRequest,
  path: string,
  runtimeFile: (path: string) => Promise<RuntimeFileResult | undefined>,
): Promise<RuntimeAssetResult | undefined> {
  const result = await runtimeFile(path)
  if (!result) return undefined
  return {
    runtimeId: message.runtimeId,
    assetId: message.assetId,
    ok: result.ok,
    status: result.status,
    statusText: result.statusText,
    contentType: result.contentType,
    bytes: result.bytes,
    error: result.error,
  }
}
