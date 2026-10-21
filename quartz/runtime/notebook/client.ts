import type { NotebookLspConfig } from '../lsp/pyright'
import type {
  KernelPresence,
  NotebookRuntimeState,
  RuntimeCell,
  RuntimePayload,
  SourceControls,
} from './client/runtime-state'
import type { Kernel, NotebookModule } from './kernel'
import {
  notebookKernelCommandEvent,
  notebookKernelRequestEvent,
  notebookKernelRunAllEvent,
} from '../../util/notebook-kernel-events'
import { notebookLocalSourcesClearedEvent } from '../../util/notebook-source-events'
import { configureNotebookRuntimeAssets, type NotebookRuntimeAssetConfig } from './assets'
import {
  armEditPrefix,
  claimNotebookKey,
  clearCellSelection,
  clearEditPrefix,
  commandCell,
  notebookRunAndAdvanceKey,
  notebookRunKey,
  queryCell,
  selectCell,
  siteShortcutLayerActive,
  targetOwnsKeyboard,
} from './client/cell-navigation'
import {
  destroyRuntime,
  handleKernelCommand,
  handleKernelRequest,
  handleKernelRunAll,
  reset,
  runAll,
  runCell,
  scheduleRuntimePreload,
  setExecutionLabel,
  stop,
} from './client/kernel-execution'
import { syncStaticOutputTabs } from './client/output-tabs'
import {
  parseRuntimeJson,
  readRuntimePayload,
  readStoredNotebookVimMode,
  readStoredVimMode,
  runtimeCellEditorLanguage,
} from './client/payload-and-warmup'
import {
  addCleanup,
  ensureCellControls,
  ensureToolbar,
  syncToolbarToggles,
  toggleDebug,
  toggleVimMode,
} from './client/runtime-chrome'
import {
  closeSourceEditor,
  decorateSourceEditor,
  handleLocalSourcesCleared,
  runCellAndAdvance,
  showSourceEditor,
} from './client/source-editor'
import { backendFor } from './registry'
export {
  nextNotebookCellId,
  notebookRunAndAdvanceKey,
  notebookRunKey,
} from './client/cell-navigation'
export {
  notebookRuntimeKernelLanguages,
  notebookRuntimePreloadLanguages,
} from './client/payload-and-warmup'

export async function warmNotebookRuntimeEditorAssets(
  data: readonly string[],
  assets: NotebookRuntimeAssetConfig = {},
): Promise<void> {
  configureNotebookRuntimeAssets(assets)
  const languages = new Set<string>()
  const lspConfigs: NotebookLspConfig[] = []
  let lsp = false
  let vimMode = false

  for (const text of data) {
    const payload = readRuntimePayload(parseRuntimeJson(text))
    if (!payload) continue
    languages.add(payload.language)
    vimMode ||= payload.vimMode ?? readStoredNotebookVimMode()
    let lspCell: RuntimeCell | undefined
    for (const cell of payload.cells) {
      const editorLanguage = runtimeCellEditorLanguage(cell)
      languages.add(editorLanguage)
      const cellLsp = backendFor(editorLanguage)?.editor?.lspBridge !== undefined
      lsp ||= cellLsp
      lspCell ??= cellLsp ? cell : undefined
    }
    if (lspCell) {
      const language = runtimeCellEditorLanguage(lspCell)
      lspConfigs.push({
        enabled: true,
        runtimeId: payload.id,
        sourcePath: payload.sourcePath,
        cellId: lspCell.id,
        language,
        cells: () =>
          payload.cells.map(cell => ({
            id: cell.id,
            source: cell.source,
            language: runtimeCellEditorLanguage(cell),
            executionIndex: cell.executionIndex,
          })),
      })
    }
  }

  if (languages.size === 0) languages.add('python')
  const { warmNotebookCodeEditorAssets } = await import('../editor/code-editor')
  await warmNotebookCodeEditorAssets({ languages: [...languages], lsp, vimMode })
  if (lspConfigs.length > 0) {
    const { warmNotebookLspRuntime } = await import('../lsp/pyright')
    await Promise.all(lspConfigs.map(config => warmNotebookLspRuntime(config)))
  }
}

function createNotebookRuntime(
  root: HTMLElement,
  payload: RuntimePayload,
  assets: NotebookRuntimeAssetConfig,
): NotebookRuntimeState {
  const state: NotebookRuntimeState = {
    payload: payload,
    root: root,
    cellRoot: root.closest('article') ?? root.parentElement ?? document,
    assets: assets,
    kernels: new Map<string, Kernel>(),
    kernelRequests: new Map<string, Promise<Kernel>>(),
    kernelEpochs: new Map<string, number>(),
    kernelPresences: new Map<string, KernelPresence>(),
    runningKernel: undefined,
    runningBackendName: undefined,
    moduleCache: new Map<string, NotebookModule | null>(),
    moduleFetches: new Map<string, Promise<NotebookModule | null>>(),
    importableModules: payload.importableModules ? new Set(payload.importableModules) : undefined,
    savedOutputs: new Map<string, HTMLElement[]>(),
    sourceControls: new Map<string, SourceControls>(),
    executionCounter: Math.max(0, ...payload.cells.map(cell => cell.executionIndex ?? 0)),
    running: false,
    stopped: false,
    runningCellId: undefined,
    debug: payload.debug ?? false,
    vimMode: payload.vimMode ?? readStoredVimMode(),
    activeCellId: undefined,
    editPrefixTimeout: undefined,
    runtimePreload: undefined,
    runtimeEpoch: 0,
    disposed: false,
    loadNotebookCodeEditor: () => import('../editor/code-editor'),
  }
  return state
}

function mount(state: NotebookRuntimeState) {
  const onRunAll = () => void runAll(state)
  const onStop = () => stop(state)
  const onReset = () => reset(state)
  const onToggleDebug = () => toggleDebug(state)
  const onToggleVimMode = () => toggleVimMode(state)
  const onHandleNotebookKeydown = (event: KeyboardEvent) => handleNotebookKeydown(state, event)
  const onHandleKernelRequest = (event: Event) => handleKernelRequest(state, event)
  const onHandleKernelCommand = (event: Event) => handleKernelCommand(state, event)
  const onHandleKernelRunAll = () => handleKernelRunAll(state)
  const onHandleLocalSourcesCleared = () => handleLocalSourcesCleared(state)
  if (state.payload.toolbar === false) {
    state.root.querySelector<HTMLElement>('[data-notebook-runtime-toolbar]')?.remove()
  } else {
    ensureToolbar(state)
  }
  decorateCells(state)
  const runAllBinding = state.root.querySelector<HTMLButtonElement>('[data-notebook-run-all]')
  const stopBinding = state.root.querySelector<HTMLButtonElement>('[data-notebook-stop]')
  const resetBinding = state.root.querySelector<HTMLButtonElement>('[data-notebook-reset]')
  const debug = state.root.querySelector<HTMLButtonElement>('[data-notebook-debug]')
  const vim = state.root.querySelector<HTMLButtonElement>('[data-notebook-vim-mode]')
  runAllBinding?.addEventListener('click', onRunAll)
  stopBinding?.addEventListener('click', onStop)
  resetBinding?.addEventListener('click', onReset)
  debug?.addEventListener('click', onToggleDebug)
  vim?.addEventListener('click', onToggleVimMode)
  document.addEventListener('keydown', onHandleNotebookKeydown, true)
  document.addEventListener(notebookKernelRequestEvent, onHandleKernelRequest)
  document.addEventListener(notebookKernelCommandEvent, onHandleKernelCommand)
  document.addEventListener(notebookKernelRunAllEvent, onHandleKernelRunAll)
  document.addEventListener(notebookLocalSourcesClearedEvent, onHandleLocalSourcesCleared)
  addCleanup(() => {
    runAllBinding?.removeEventListener('click', onRunAll)
    stopBinding?.removeEventListener('click', onStop)
    resetBinding?.removeEventListener('click', onReset)
    debug?.removeEventListener('click', onToggleDebug)
    vim?.removeEventListener('click', onToggleVimMode)
    document.removeEventListener('keydown', onHandleNotebookKeydown, true)
    document.removeEventListener(notebookKernelRequestEvent, onHandleKernelRequest)
    document.removeEventListener(notebookKernelCommandEvent, onHandleKernelCommand)
    document.removeEventListener(notebookKernelRunAllEvent, onHandleKernelRunAll)
    document.removeEventListener(notebookLocalSourcesClearedEvent, onHandleLocalSourcesCleared)
    clearEditPrefix(state)
    state.disposed = true
    destroyRuntime(state)
  })
  syncToolbarToggles(state)
  scheduleRuntimePreload(state)
}

function enterEditMode(state: NotebookRuntimeState, cell: RuntimeCell) {
  clearEditPrefix(state)
  selectCell(state, cell.id)
  void showSourceEditor(state, cell, true)
}

function handleNotebookKeydown(state: NotebookRuntimeState, event: KeyboardEvent) {
  if (event.defaultPrevented || siteShortcutLayerActive() || targetOwnsKeyboard(event.target))
    return
  const cell = commandCell(state, event.target)
  if (!cell) return
  if (notebookRunAndAdvanceKey(event)) {
    clearEditPrefix(state)
    selectCell(state, cell.id)
    claimNotebookKey(event)
    void runCellAndAdvance(state, cell)
    return
  }
  if (notebookRunKey(event)) {
    clearEditPrefix(state)
    selectCell(state, cell.id)
    claimNotebookKey(event)
    void runCell(state, cell)
    return
  }
  if (event.ctrlKey || event.metaKey || event.altKey) {
    clearEditPrefix(state)
    return
  }
  if (state.editPrefixTimeout !== undefined && event.key !== 'e') clearEditPrefix(state)
  if (event.key === 'e') {
    claimNotebookKey(event)
    if (state.editPrefixTimeout !== undefined) {
      enterEditMode(state, cell)
      return
    }
    armEditPrefix(state)
    return
  }
  if (event.key === 'Enter' || event.key === 'i') {
    claimNotebookKey(event)
    enterEditMode(state, cell)
    return
  }
  if (event.key !== 'Escape') return
  const controls = state.sourceControls.get(cell.id)
  if (controls && !controls.editorHost.hidden) {
    claimNotebookKey(event)
    void closeSourceEditor(state, cell)
    return
  }
  if (state.activeCellId !== undefined) {
    claimNotebookKey(event)
    clearCellSelection(state)
  }
}

function decorateCells(state: NotebookRuntimeState) {
  for (const cell of state.payload.cells) {
    const controls = queryCell<HTMLElement>(state, `[data-notebook-cell="${CSS.escape(cell.id)}"]`)
    if (!controls) continue
    ensureCellControls(cell, controls)
    setExecutionLabel(state, cell.id, cell.executionIndex)
    const existingFrame = controls.closest<HTMLElement>('[data-notebook-cell-frame]')
    if (existingFrame) {
      syncStaticOutputTabs(existingFrame, cell.id)
      decorateSourceEditor(state, cell, existingFrame)
      continue
    }
    const frame = document.createElement('div')
    frame.className = 'notebook-code-cell'
    frame.dataset.notebookCellFrame = cell.id
    controls.before(frame)
    frame.append(controls)
    let sibling = frame.nextElementSibling
    while (sibling instanceof HTMLElement) {
      if (sibling.matches('[data-notebook-cell]')) break
      if (
        sibling.matches(
          'figure[data-rehype-pretty-code-figure], .notebook-output, [data-notebook-output]',
        )
      ) {
        const next = sibling.nextElementSibling
        frame.append(sibling)
        sibling = next
        continue
      }
      break
    }
    syncStaticOutputTabs(frame, cell.id)
    decorateSourceEditor(state, cell, frame)
  }
}

export function mountNotebookRuntime(
  root: HTMLElement,
  text: string,
  assets: NotebookRuntimeAssetConfig = {},
) {
  if (root.dataset.runtimeMounted === 'true') return
  configureNotebookRuntimeAssets(assets)
  const parsed = parseRuntimeJson(text)
  if (parsed === undefined) return
  const payload = readRuntimePayload(parsed)
  if (!payload) return
  root.dataset.runtimeMounted = 'true'
  mount(createNotebookRuntime(root, payload, assets))
}
