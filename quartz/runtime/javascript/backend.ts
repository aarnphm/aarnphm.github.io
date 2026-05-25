import {
  registerBackend,
  type CanExecuteResult,
  type ExecutableLanguageBackend,
  type KernelFactoryOptions,
} from '../notebook/backend'
import { createBrowserJavaScriptKernel } from './browser-kernel'
import { unsupportedJavaScriptRuntimeReason } from './can-execute'

function javascriptCanExecute(source: string): CanExecuteResult {
  const reason = unsupportedJavaScriptRuntimeReason(source)
  return reason ? { ok: false, reason } : { ok: true }
}

async function javascriptKernelFactory(opts: KernelFactoryOptions) {
  return createBrowserJavaScriptKernel(opts)
}

export const javascriptBackend: ExecutableLanguageBackend = {
  name: 'javascript',
  fileExts: ['.js', '.mjs', '.cjs'],
  aliases: ['javascript', 'js', 'node', 'nodejs', 'ijavascript', 'ecmascript'],
  shellMagics: ['javascript', 'js', 'javascript-shell', 'js-shell'],
  workerAssetKey: 'javascriptWorkerUrl',
  kernelFactory: javascriptKernelFactory,
  canExecute: javascriptCanExecute,
  editor: {
    languageExtension: async () => {
      const mod = await import('@codemirror/lang-javascript')
      return mod.javascript()
    },
  },
}

registerBackend(javascriptBackend)
