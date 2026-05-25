import type { CellId } from '../../util/notebook/types'
import type { Kernel, RuntimeEvent } from '../notebook/kernel'
import { createHaskellPlaygroundKernel } from '../haskell/playground-kernel'
import {
  registerBackend,
  type CanExecuteResult,
  type ExecutableLanguageBackend,
} from '../notebook/backend'
import { createRustPlaygroundKernel } from '../rust/playground-kernel'

type NativeLanguageSpec = {
  readonly name: string
  readonly fileExts: readonly string[]
  readonly aliases: readonly string[]
  readonly shellMagics: readonly string[]
}

class UnavailableNativeKernel implements Kernel {
  readonly language: string

  constructor(
    language: string,
    private readonly reason: string,
  ) {
    this.language = language
  }

  async init(): Promise<void> {}

  async *execute(cellId: CellId, _source: string): AsyncIterable<RuntimeEvent> {
    yield { type: 'started', cellId }
    yield {
      type: 'output',
      cellId,
      output: {
        type: 'error',
        ename: 'UnsupportedRuntimeFeature',
        evalue: this.reason,
        traceback: this.reason,
      },
    }
    yield { type: 'done', cellId, executionCount: null, failed: true }
  }

  interrupt(): void {}

  async reset(): Promise<void> {}

  async dispose(): Promise<void> {}
}

function nativeRuntimeReason(language: string): string {
  return `${language} notebook cells need a native compiler runtime, which is unavailable in this browser sandbox. Use a server-backed notebook runtime for this cell.`
}

function nativeCanExecute(language: string): CanExecuteResult {
  return { ok: false, reason: nativeRuntimeReason(language) }
}

function rustCanExecute(source: string): CanExecuteResult {
  return source.trim().length === 0
    ? { ok: false, reason: 'rust notebook cells need source code to execute.' }
    : { ok: true }
}

function haskellCanExecute(source: string): CanExecuteResult {
  return source.trim().length === 0
    ? { ok: false, reason: 'haskell notebook cells need source code to execute.' }
    : { ok: true }
}

function nativeBackend(spec: NativeLanguageSpec): ExecutableLanguageBackend {
  return {
    ...spec,
    kernelFactory: async () =>
      new UnavailableNativeKernel(spec.name, nativeRuntimeReason(spec.name)),
    canExecute: () => nativeCanExecute(spec.name),
  }
}

export const rustBackend: ExecutableLanguageBackend = {
  name: 'rust',
  fileExts: ['.rs'],
  aliases: ['rust', 'rs'],
  shellMagics: ['rust-shell', 'rs-shell'],
  kernelFactory: async () => createRustPlaygroundKernel(),
  canExecute: rustCanExecute,
  editor: {
    languageExtension: async () => {
      const mod = await import('@codemirror/lang-rust')
      return mod.rust()
    },
  },
}

export const mojoBackend = nativeBackend({
  name: 'mojo',
  fileExts: ['.mojo'],
  aliases: ['mojo'],
  shellMagics: ['mojo-shell'],
})

export const haskellBackend: ExecutableLanguageBackend = {
  name: 'haskell',
  fileExts: ['.hs', '.lhs'],
  aliases: ['haskell', 'hs', 'ghc', 'runghc'],
  shellMagics: ['haskell-shell', 'hs-shell'],
  kernelFactory: async () => createHaskellPlaygroundKernel(),
  canExecute: haskellCanExecute,
}

export const ocamlBackend = nativeBackend({
  name: 'ocaml',
  fileExts: ['.ml', '.mli'],
  aliases: ['ocaml', 'ml', 'utop'],
  shellMagics: ['ocaml-shell', 'ml-shell'],
})

registerBackend(rustBackend)
registerBackend(mojoBackend)
registerBackend(haskellBackend)
registerBackend(ocamlBackend)
