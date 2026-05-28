import {
  registerBackend,
  type ExecutableLanguageBackend,
  type KernelFactoryOptions,
} from '../notebook/backend'
import {
  nativeRuntimeCanExecute,
  nativeRuntimePackKernelFactory,
  type NativeRuntimeLanguage,
} from './runtime-pack'

type NativeLanguageSpec = {
  readonly name: NativeRuntimeLanguage
  readonly fileExts: readonly string[]
  readonly aliases: readonly string[]
  readonly shellMagics: readonly string[]
  readonly preload?: boolean
}

function nativeBackend(spec: NativeLanguageSpec): ExecutableLanguageBackend {
  return {
    ...spec,
    workerAssetKey: 'nativeRuntimeManifestUrl',
    kernelFactory: async (opts: KernelFactoryOptions) =>
      nativeRuntimePackKernelFactory(spec.name, opts),
    canExecute: nativeRuntimeCanExecute,
  }
}

export const rustBackend = nativeBackend({
  name: 'rust',
  fileExts: ['.rs'],
  aliases: ['rust', 'rs'],
  shellMagics: ['rust-shell', 'rs-shell'],
})

export const cBackend = nativeBackend({
  name: 'c',
  fileExts: ['.c', '.h'],
  aliases: ['c'],
  shellMagics: ['c-shell'],
  preload: false,
})

export const cppBackend = nativeBackend({
  name: 'cpp',
  fileExts: ['.cc', '.cpp', '.cxx', '.hh', '.hpp', '.hxx'],
  aliases: ['cpp', 'c++', 'cxx'],
  shellMagics: ['cpp-shell', 'c++-shell', 'cxx-shell'],
  preload: false,
})

export const mojoBackend = nativeBackend({
  name: 'mojo',
  fileExts: ['.mojo'],
  aliases: ['mojo'],
  shellMagics: ['mojo-shell'],
})

export const haskellBackend = nativeBackend({
  name: 'haskell',
  fileExts: ['.hs', '.lhs'],
  aliases: ['haskell', 'hs', 'ghc', 'runghc'],
  shellMagics: ['haskell-shell', 'hs-shell'],
})

export const ocamlBackend = nativeBackend({
  name: 'ocaml',
  fileExts: ['.ml', '.mli'],
  aliases: ['ocaml', 'ml', 'utop'],
  shellMagics: ['ocaml-shell', 'ml-shell'],
})

export const goBackend = nativeBackend({
  name: 'go',
  fileExts: ['.go'],
  aliases: ['go', 'golang'],
  shellMagics: ['go-shell', 'golang-shell'],
})

export const wasmBackend = nativeBackend({
  name: 'wasm',
  fileExts: ['.wasm', '.wat'],
  aliases: ['wasm', 'wat', 'webassembly'],
  shellMagics: ['wasm-shell', 'wat-shell'],
})

registerBackend(rustBackend)
registerBackend(cBackend)
registerBackend(cppBackend)
registerBackend(mojoBackend)
registerBackend(haskellBackend)
registerBackend(ocamlBackend)
registerBackend(goBackend)
registerBackend(wasmBackend)
