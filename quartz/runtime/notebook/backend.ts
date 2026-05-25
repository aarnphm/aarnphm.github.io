import type { Extension } from '@codemirror/state'
import type { LspBridge } from '../lsp/bridge'
import type { NotebookRuntimeAssets } from './assets'
import type { Kernel, NotebookModule } from './kernel'
import type {
  RuntimeAssetRequest,
  RuntimeAssetResult,
  RuntimeDownload,
  RuntimeFileResult,
} from './kernel'

export type CanExecuteResult =
  | { readonly ok: true }
  | { readonly ok: false; readonly reason: string }

export type KernelFactoryOptions = {
  readonly runtimeId: string
  readonly sourcePath: string
  readonly indexUrl?: string
  readonly workerUrl?: string | URL
  readonly resolveAsset?: (
    request: RuntimeAssetRequest,
    runtimeFile: (path: string) => Promise<RuntimeFileResult | undefined>,
  ) => Promise<RuntimeAssetResult>
  readonly download?: (download: RuntimeDownload) => void
  readonly status?: (text: string) => void
}

export type RuntimeModuleResolver = {
  readonly importNames: (source: string) => readonly string[]
  readonly moduleSource: (raw: string, sourcePath: string) => NotebookModule['source']
}

type LanguageBackendBase = {
  readonly name: string
  readonly fileExts: readonly string[]
  readonly aliases: readonly string[]
  readonly shellMagics: readonly string[]
  readonly workerAssetKey?: keyof NotebookRuntimeAssets
  readonly defaultIndexUrl?: string
  readonly canExecute: (source: string) => CanExecuteResult
  readonly editor?: {
    readonly languageExtension?: () => Promise<Extension>
    readonly lspBridge?: () => Promise<LspBridge>
  }
  readonly formatter?: () => Promise<(source: string) => Promise<string>>
}

export type LanguageBackend = LanguageBackendBase & {
  readonly kernelFactory: (opts: KernelFactoryOptions) => Promise<Kernel>
  readonly moduleResolver?: RuntimeModuleResolver
}

export type ExecutableLanguageBackend = LanguageBackend

const registry = new Map<string, LanguageBackend>()
const shellMagicIndex = new Map<string, LanguageBackend>()

function normalize(language: string): string {
  return language.trim().toLowerCase()
}

export function registerBackend(backend: LanguageBackend): void {
  const keys = new Set<string>([backend.name, ...backend.aliases, ...backend.fileExts])
  for (const key of keys) registry.set(normalize(key), backend)
  for (const magic of backend.shellMagics) shellMagicIndex.set(normalize(magic), backend)
}

export function backendFor(language: string): LanguageBackend | undefined {
  return registry.get(normalize(language))
}

export function backendForShellMagic(magic: string): LanguageBackend | undefined {
  return shellMagicIndex.get(normalize(magic))
}

export function unregisterBackend(name: string): void {
  const target = backendFor(name)
  if (!target) return
  for (const [key, value] of registry.entries()) if (value === target) registry.delete(key)
  for (const [key, value] of shellMagicIndex.entries())
    if (value === target) shellMagicIndex.delete(key)
}

export function listBackends(): readonly LanguageBackend[] {
  return Array.from(new Set(registry.values()))
}
