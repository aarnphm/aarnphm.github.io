import type { CellId } from '../../util/notebook/types'
import type { Kernel, RuntimeEvent } from '../notebook/kernel'
import {
  registerBackend,
  type CanExecuteResult,
  type ExecutableLanguageBackend,
} from '../notebook/backend'

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

function nativeBackend(spec: NativeLanguageSpec): ExecutableLanguageBackend {
  return {
    ...spec,
    kernelFactory: async () =>
      new UnavailableNativeKernel(spec.name, nativeRuntimeReason(spec.name)),
    canExecute: () => nativeCanExecute(spec.name),
  }
}

export const rustBackend = nativeBackend({
  name: 'rust',
  fileExts: ['.rs'],
  aliases: ['rust', 'rs'],
  shellMagics: ['rust', 'rust-shell', 'rs-shell'],
})

export const mojoBackend = nativeBackend({
  name: 'mojo',
  fileExts: ['.mojo'],
  aliases: ['mojo'],
  shellMagics: ['mojo', 'mojo-shell'],
})

export const haskellBackend = nativeBackend({
  name: 'haskell',
  fileExts: ['.hs', '.lhs'],
  aliases: ['haskell', 'hs', 'ghc', 'runghc'],
  shellMagics: ['haskell', 'hs', 'haskell-shell', 'ghc', 'runghc'],
})

export const ocamlBackend = nativeBackend({
  name: 'ocaml',
  fileExts: ['.ml', '.mli'],
  aliases: ['ocaml', 'ml', 'utop'],
  shellMagics: ['ocaml', 'ml', 'ocaml-shell', 'utop'],
})

registerBackend(rustBackend)
registerBackend(mojoBackend)
registerBackend(haskellBackend)
registerBackend(ocamlBackend)
