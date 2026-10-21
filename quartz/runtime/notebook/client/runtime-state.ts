import type { NotebookKernelStatus } from '../../../util/notebook-kernel-events'
import type { CellId } from '../../../util/notebook/types'
import type { NotebookCodeEditor } from '../../editor/code-editor'
import type { NotebookLspConfig } from '../../lsp/pyright'
import type { NotebookRuntimeAssetConfig } from '../assets'
import type { Kernel, NotebookModule } from '../kernel'

export type RuntimeCell = {
  id: CellId
  source: string
  language: string
  displayLanguage?: string
  executionIndex: number | null
}

export type KernelPresence = {
  readonly status: NotebookKernelStatus
  readonly statusDetail?: string
}

export type SourceControls = {
  frame: HTMLElement
  editor: NotebookCodeEditor | undefined
  editorHost: HTMLElement
  figure: HTMLElement | undefined
  status: HTMLElement
  vimButton: HTMLButtonElement
  editButton: HTMLButtonElement
  saveButton: HTMLButtonElement
  revertButton: HTMLButtonElement
  source: string
  storedSource: string | undefined
  renderedSource: string
  dirty: boolean
}

export type RuntimePayload = {
  id: string
  sourcePath: string
  language: string
  indexUrl?: string
  cells: RuntimeCell[]
  toolbar?: boolean
  debug?: boolean
  vimMode?: boolean
  importableModules?: string[]
}

export type NotebookRuntimeState = {
  payload: RuntimePayload
  root: HTMLElement
  cellRoot: Document | HTMLElement
  assets: NotebookRuntimeAssetConfig
  kernels: Map<string, Kernel>
  kernelRequests: Map<string, Promise<Kernel>>
  kernelEpochs: Map<string, number>
  kernelPresences: Map<string, KernelPresence>
  runningKernel: Kernel | undefined
  runningBackendName: string | undefined
  moduleCache: Map<string, NotebookModule | null>
  moduleFetches: Map<string, Promise<NotebookModule | null>>
  importableModules: Set<string> | undefined
  savedOutputs: Map<string, HTMLElement[]>
  sourceControls: Map<string, SourceControls>
  executionCounter: number
  running: boolean
  stopped: boolean
  runningCellId: string | undefined
  debug: boolean
  vimMode: boolean
  activeCellId: string | undefined
  editPrefixTimeout: number | undefined
  runtimePreload: Promise<void> | undefined
  runtimeEpoch: number
  disposed: boolean
  loadNotebookCodeEditor: () => Promise<{
    createNotebookCodeEditor: (config: {
      parent: HTMLElement
      initialContent?: string
      language?: string
      vimMode?: boolean
      onEdited?: () => void
      onChange?: (content: string) => void
      onSubmit?: () => void
      onSubmitAndAdvance?: () => void
      onSave?: () => void
      onCancel?: () => void
      lsp?: NotebookLspConfig
    }) => Promise<NotebookCodeEditor>
    renderNotebookHighlightedLines: (
      source: string,
      languageName: string | undefined,
    ) => Promise<HTMLElement[]>
  }>
}
