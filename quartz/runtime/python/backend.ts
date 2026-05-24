import {
  registerBackend,
  type CanExecuteResult,
  type ExecutableLanguageBackend,
  type KernelFactoryOptions,
} from '../notebook/backend'
import { unsupportedNotebookRuntimeReason } from './can-execute'
import { extractPythonImports } from './imports'
import { notebookRuntimeModuleSource } from './module-source'
import { createPyodideKernel } from './pyodide-kernel'

export const pyodideIndexUrl = 'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/'

function pythonCanExecute(source: string): CanExecuteResult {
  const reason = unsupportedNotebookRuntimeReason(source)
  return reason ? { ok: false, reason } : { ok: true }
}

async function pythonKernelFactory(opts: KernelFactoryOptions) {
  return createPyodideKernel(opts)
}

export const pythonBackend: ExecutableLanguageBackend = {
  name: 'python',
  fileExts: ['.py', '.ipynb'],
  aliases: ['python', 'py', 'ipython', 'python3'],
  shellMagics: ['python-shell', 'py-shell'],
  defaultIndexUrl: pyodideIndexUrl,
  kernelFactory: pythonKernelFactory,
  moduleResolver: { importNames: extractPythonImports, moduleSource: notebookRuntimeModuleSource },
  canExecute: pythonCanExecute,
  editor: {
    languageExtension: async () => {
      const mod = await import('@codemirror/lang-python')
      return mod.python()
    },
    lspBridge: async () => {
      const mod = await import('../lsp/pyright')
      return mod.pyrightLspBridge
    },
  },
}

registerBackend(pythonBackend)
