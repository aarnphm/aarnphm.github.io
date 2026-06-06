import {
  registerBackend,
  type ExecutableLanguageBackend,
  type KernelFactoryOptions,
} from '../notebook/backend'
import { createLeanKernel } from './kernel'

export const leanBackend: ExecutableLanguageBackend = {
  name: 'lean',
  fileExts: ['.lean'],
  aliases: ['lean', 'lean4'],
  shellMagics: ['lean-shell'],
  preload: false,
  canExecute: () => ({ ok: true }),
  editor: {
    languageExtension: async () => {
      const { codemirrorCodeLanguage } = await import('../../util/codemirror-language')
      return codemirrorCodeLanguage('lean')
    },
  },
  kernelFactory: async (opts: KernelFactoryOptions) => createLeanKernel(opts),
}

registerBackend(leanBackend)
