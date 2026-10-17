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
  defaultIndexUrl: '/api/lean/verify',
  preload: false,
  canExecute: () => ({ ok: true }),
  editor: {
    languageExtension: async () => {
      const { leanLanguageExtension } = await import('../../util/codemirror-lean')
      return leanLanguageExtension()
    },
    lspBridge: async () => {
      const { leanLspBridge } = await import('../lsp/lean')
      return leanLspBridge
    },
  },
  kernelFactory: async (opts: KernelFactoryOptions) => createLeanKernel(opts),
}

registerBackend(leanBackend)
