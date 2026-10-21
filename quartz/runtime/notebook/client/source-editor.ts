import type { NotebookRuntimeState, RuntimeCell, SourceControls } from './runtime-state'
import { backendFor } from '../registry'
import { focusCell, nextCell, selectCell, targetOwnsKeyboard } from './cell-navigation'
import { runCell, stop } from './kernel-execution'
import { runtimeCellEditorLanguage, setNotebookIconButton } from './payload-and-warmup'
import { addCleanup, setStatus, toggleVimMode } from './runtime-chrome'
import {
  clearStoredSource,
  notebookLanguageBadgeElement,
  readStoredSource,
  renderedFigureSource,
  replaceRenderedCellSource,
  sourceForCell,
  sourceSnapshotForCell,
  writeStoredSource,
} from './source-text'

export function handleLocalSourcesCleared(state: NotebookRuntimeState) {
  void clearLocalSourcesFromRuntime(state)
}

export async function clearLocalSourcesFromRuntime(state: NotebookRuntimeState) {
  for (const cell of state.payload.cells) {
    const controls = state.sourceControls.get(cell.id)
    clearStoredSource(state, cell)
    if (!controls) continue
    const preserveEditorSource = !controls.editorHost.hidden && controls.dirty
    if (preserveEditorSource) {
      syncSourceControls(state, cell)
      continue
    }
    controls.source = cell.source
    controls.dirty = false
    controls.editor?.setValue(cell.source)
    await replaceRenderedCellSource(state, cell, controls, cell.source)
    syncSourceControls(state, cell)
  }
}

export async function runCellAndAdvance(state: NotebookRuntimeState, cell: RuntimeCell) {
  if (state.running) return
  if (!(await saveEditorSource(state, cell, false))) return
  if (state.running) return
  const running = runCell(state, cell)
  focusCell(state, nextCell(state, cell) ?? cell)
  void running
}

export function decorateSourceEditor(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
  frame: HTMLElement,
) {
  if (state.sourceControls.has(cell.id)) return
  if (!frame.hasAttribute('tabindex')) frame.tabIndex = -1
  const selectSource = (event: Event) => {
    selectCell(state, cell.id)
    if (
      (event.type === 'pointerdown' || event.type === 'click') &&
      !targetOwnsKeyboard(event.target)
    ) {
      frame.focus({ preventScroll: true })
    }
  }
  frame.addEventListener('pointerdown', selectSource, true)
  frame.addEventListener('click', selectSource)
  frame.addEventListener('focusin', selectSource)
  const figure =
    frame.querySelector<HTMLElement>('figure[data-rehype-pretty-code-figure]') ?? undefined
  const actions =
    frame.querySelector<HTMLElement>(`[data-notebook-cell-actions="${CSS.escape(cell.id)}"]`) ??
    document.createElement('div')
  actions.className = 'notebook-cell-actions'
  actions.dataset.notebookCellActions = cell.id
  const editorLanguage = runtimeCellEditorLanguage(cell)
  const languageBadge =
    actions.querySelector<HTMLElement>('.notebook-language-badge[data-notebook-language]') ??
    notebookLanguageBadgeElement(editorLanguage)
  if (!frame.dataset.notebookLanguage && languageBadge.dataset.notebookLanguage) {
    frame.dataset.notebookLanguage = languageBadge.dataset.notebookLanguage
  }
  const runButton =
    frame.querySelector<HTMLButtonElement>(`[data-notebook-run-cell="${CSS.escape(cell.id)}"]`) ??
    document.createElement('button')
  runButton.type = 'button'
  runButton.dataset.notebookRunCell = cell.id
  setNotebookIconButton(runButton, 'run', `Run ${cell.id}`)
  const runSource = () => {
    if (state.running && state.runningCellId === cell.id) {
      stop(state)
      return
    }
    void runCell(state, cell)
  }
  runButton.addEventListener('click', runSource)
  const vimButton =
    frame.querySelector<HTMLButtonElement>(`[data-notebook-vim-cell="${CSS.escape(cell.id)}"]`) ??
    document.createElement('button')
  vimButton.type = 'button'
  vimButton.dataset.notebookVimCell = cell.id
  setNotebookIconButton(vimButton, 'vim', 'Enable Vim mode')
  const toggleCellVimMode = () => {
    toggleVimMode(state)
  }
  vimButton.addEventListener('click', toggleCellVimMode)
  const editorHost =
    frame.querySelector<HTMLElement>(`[data-notebook-source-editor="${CSS.escape(cell.id)}"]`) ??
    document.createElement('div')
  editorHost.className = 'notebook-source-editor'
  editorHost.dataset.notebookSourceEditor = cell.id
  editorHost.hidden = true
  const editButton =
    frame.querySelector<HTMLButtonElement>(`[data-notebook-edit-cell="${CSS.escape(cell.id)}"]`) ??
    document.createElement('button')
  editButton.type = 'button'
  editButton.dataset.notebookEditCell = cell.id
  setNotebookIconButton(editButton, 'edit', `Edit ${cell.id}`)
  const editSource = () => {
    void showSourceEditor(state, cell, true)
  }
  editButton.addEventListener('click', editSource)
  const saveButton =
    frame.querySelector<HTMLButtonElement>(`[data-notebook-save-cell="${CSS.escape(cell.id)}"]`) ??
    document.createElement('button')
  saveButton.type = 'button'
  saveButton.dataset.notebookSaveCell = cell.id
  setNotebookIconButton(saveButton, 'save', `Save ${cell.id} locally`)
  saveButton.hidden = true
  const saveSource = () => {
    void saveEditorSource(state, cell)
  }
  saveButton.addEventListener('click', saveSource)
  const revertButton =
    frame.querySelector<HTMLButtonElement>(
      `[data-notebook-revert-cell="${CSS.escape(cell.id)}"]`,
    ) ?? document.createElement('button')
  revertButton.type = 'button'
  revertButton.dataset.notebookRevertCell = cell.id
  setNotebookIconButton(revertButton, 'revert', `Revert ${cell.id} local edit`)
  revertButton.hidden = true
  const revertSource = () => {
    void revertEditorSource(state, cell)
  }
  revertButton.addEventListener('click', revertSource)
  const status =
    frame.querySelector<HTMLElement>(
      `[data-notebook-local-source-status="${CSS.escape(cell.id)}"]`,
    ) ?? document.createElement('span')
  status.className = 'notebook-local-source-status'
  status.dataset.notebookLocalSourceStatus = cell.id
  status.hidden = true
  actions.replaceChildren(
    languageBadge,
    vimButton,
    runButton,
    editButton,
    saveButton,
    revertButton,
    status,
  )
  if (!actions.isConnected) frame.append(actions)
  if (figure) {
    if (!editorHost.isConnected) figure.before(editorHost)
  } else {
    if (!editorHost.isConnected) frame.append(editorHost)
  }
  const stored = readStoredSource(state, cell)
  const renderedSource = figure ? renderedFigureSource(figure) : undefined
  const baseSource =
    renderedSource !== undefined && renderedSource.trimEnd() !== cell.source.trimEnd()
      ? renderedSource
      : cell.source
  cell.source = baseSource
  state.sourceControls.set(cell.id, {
    frame,
    editor: undefined,
    editorHost,
    figure,
    status,
    vimButton,
    editButton,
    saveButton,
    revertButton,
    source: stored ?? baseSource,
    storedSource: stored,
    renderedSource: renderedSource ?? baseSource,
    dirty: false,
  })
  const controls = state.sourceControls.get(cell.id)
  if (!controls) return
  if (stored !== undefined && stored === cell.source) {
    clearStoredSource(state, cell)
  } else if (stored !== undefined && figure) {
    void replaceRenderedCellSource(state, cell, controls, stored)
  }
  syncSourceControls(state, cell)
  addCleanup(() => {
    runButton.removeEventListener('click', runSource)
    vimButton.removeEventListener('click', toggleCellVimMode)
    editButton.removeEventListener('click', editSource)
    saveButton.removeEventListener('click', saveSource)
    revertButton.removeEventListener('click', revertSource)
    frame.removeEventListener('pointerdown', selectSource, true)
    frame.removeEventListener('click', selectSource)
    frame.removeEventListener('focusin', selectSource)
    state.sourceControls.get(cell.id)?.editor?.destroy()
  })
}

export async function ensureSourceEditor(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
  controls: SourceControls,
) {
  if (controls.editor) return controls.editor
  const { createNotebookCodeEditor } = await state.loadNotebookCodeEditor()
  const editorLanguage = runtimeCellEditorLanguage(cell)
  controls.editor = await createNotebookCodeEditor({
    parent: controls.editorHost,
    initialContent: controls.source,
    language: editorLanguage,
    vimMode: state.vimMode,
    lsp: {
      enabled: backendFor(editorLanguage)?.editor?.lspBridge !== undefined,
      runtimeId: state.payload.id,
      sourcePath: state.payload.sourcePath,
      cellId: cell.id,
      language: editorLanguage,
      cells: () =>
        state.payload.cells.map(runtimeCell => ({
          id: runtimeCell.id,
          source: sourceSnapshotForCell(state, runtimeCell),
          language: runtimeCellEditorLanguage(runtimeCell),
          executionIndex: runtimeCell.executionIndex,
        })),
    },
    onEdited: () => {
      if (controls.dirty) return
      controls.dirty = true
      syncSourceControls(state, cell)
    },
    onSubmit: () => runCell(state, cell),
    onSubmitAndAdvance: () => runCellAndAdvance(state, cell),
    onSave: () => {
      void saveEditorSource(state, cell)
    },
    onCancel: () => {
      void closeSourceEditor(state, cell)
    },
  })
  return controls.editor
}

export async function showSourceEditor(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
  visible: boolean,
  restoreFocus = true,
) {
  const controls = state.sourceControls.get(cell.id)
  if (!controls) return
  if (visible) {
    const editor = await ensureSourceEditor(state, cell, controls)
    const source = sourceForCell(state, cell)
    editor.setValue(source)
    controls.source = source
    controls.dirty = false
  }
  controls.editorHost.hidden = !visible
  if (controls.figure) controls.figure.hidden = visible
  controls.frame.toggleAttribute('data-notebook-editing', visible)
  syncSourceControls(state, cell)
  if (visible) {
    controls.editor?.focus()
  } else if (restoreFocus) {
    selectCell(state, cell.id)
    controls.frame.focus({ preventScroll: true })
  }
}

export async function closeSourceEditor(state: NotebookRuntimeState, cell: RuntimeCell) {
  const controls = state.sourceControls.get(cell.id)
  if (!controls) return
  if (controls.editor) {
    controls.source = controls.editor.getValue()
  }
  await replaceRenderedCellSource(state, cell, controls, controls.source)
  void showSourceEditor(state, cell, false)
}

export async function saveEditorSource(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
  restoreFocus = true,
): Promise<boolean> {
  const controls = state.sourceControls.get(cell.id)
  if (!controls) return true
  const source = sourceForCell(state, cell)
  if (source === cell.source) {
    controls.source = cell.source
    controls.dirty = false
    clearStoredSource(state, cell)
    await replaceRenderedCellSource(state, cell, controls, cell.source)
    await showSourceEditor(state, cell, false, restoreFocus)
    setStatus(state, 'idle')
    return true
  }
  if (!writeStoredSource(state, cell, source)) {
    setStatus(state, 'local save failed')
    return false
  }
  controls.source = source
  controls.dirty = false
  await replaceRenderedCellSource(state, cell, controls, source)
  await showSourceEditor(state, cell, false, restoreFocus)
  setStatus(state, 'saved local edit')
  return true
}

export async function revertEditorSource(state: NotebookRuntimeState, cell: RuntimeCell) {
  const controls = state.sourceControls.get(cell.id)
  if (!controls) return
  controls.source = cell.source
  controls.dirty = false
  controls.editor?.setValue(cell.source)
  clearStoredSource(state, cell)
  await replaceRenderedCellSource(state, cell, controls, cell.source)
  void showSourceEditor(state, cell, false)
  setStatus(state, 'idle')
}

export function syncSourceControls(state: NotebookRuntimeState, cell: RuntimeCell) {
  const controls = state.sourceControls.get(cell.id)
  if (!controls) return
  const stored = controls.storedSource
  const editing = !controls.editorHost.hidden
  const edited = editing && controls.dirty
  const hasStoredSource = stored !== undefined
  controls.status.hidden = !hasStoredSource && !edited
  controls.status.dataset.notebookSourceState = edited ? 'edited' : 'local'
  controls.status.title = edited ? 'edited local source' : 'saved local source'
  controls.status.setAttribute('aria-label', edited ? 'edited local source' : 'saved local source')
  controls.editButton.hidden = editing
  controls.vimButton.hidden = !editing
  controls.saveButton.hidden = !editing
  controls.revertButton.hidden = !(editing || hasStoredSource)
  controls.frame.toggleAttribute('data-notebook-local-source', hasStoredSource)
}
