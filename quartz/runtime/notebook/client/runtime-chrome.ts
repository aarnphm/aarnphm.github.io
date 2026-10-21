import type { NotebookRuntimeState, RuntimeCell } from './runtime-state'
import { queryCells } from './cell-navigation'
import {
  setNotebookIconButton,
  setNotebookToggleButtonState,
  setNotebookVimButtonState,
  writeStoredVimMode,
} from './payload-and-warmup'

export function ensureToolbar(state: NotebookRuntimeState) {
  const toolbar =
    state.root.querySelector<HTMLElement>('[data-notebook-runtime-toolbar]') ??
    document.createElement('div')
  toolbar.className = 'notebook-runtime-toolbar'
  toolbar.dataset.notebookRuntimeToolbar = ''
  toolbar.setAttribute('role', 'toolbar')
  toolbar.setAttribute('aria-label', 'Notebook runtime')
  let status = toolbar.querySelector<HTMLElement>('[data-notebook-status]')
  if (!status) {
    status = document.createElement('span')
  }
  status.className = 'notebook-runtime-status'
  status.dataset.notebookStatus = ''
  status.setAttribute('aria-live', 'polite')
  if (!status.textContent) status.textContent = 'idle'
  const ensureButton = (
    selector: string,
    setup: (button: HTMLButtonElement) => void,
  ): HTMLButtonElement => {
    const existing = toolbar.querySelector<HTMLButtonElement>(selector)
    const button = existing ?? document.createElement('button')
    setup(button)
    if (!existing) toolbar.insertBefore(button, status.isConnected ? status : null)
    return button
  }
  ensureButton('[data-notebook-run-all]', button => {
    button.type = 'button'
    button.dataset.notebookRunAll = ''
    button.classList.add('notebook-toolbar-button')
    setNotebookIconButton(button, 'run', 'Run all')
  })
  ensureButton('[data-notebook-stop]', button => {
    button.type = 'button'
    button.dataset.notebookStop = ''
    button.disabled = true
    button.classList.add('notebook-toolbar-button')
    setNotebookIconButton(button, 'stop', 'Stop execution')
  })
  ensureButton('[data-notebook-reset]', button => {
    button.type = 'button'
    button.dataset.notebookReset = ''
    button.classList.add('notebook-toolbar-button')
    setNotebookIconButton(button, 'reset', 'Reset runtime')
  })
  ensureButton('[data-notebook-debug]', button => {
    button.type = 'button'
    button.dataset.notebookDebug = ''
    button.classList.add('notebook-toolbar-button')
    setNotebookIconButton(button, 'debug', 'Enable debug output')
  })
  ensureButton('[data-notebook-vim-mode]', button => {
    button.type = 'button'
    button.dataset.notebookVimMode = ''
    button.classList.add('notebook-toolbar-button')
    setNotebookIconButton(button, 'vim', 'Enable Vim mode')
  })
  if (!status.isConnected) toolbar.append(status)
  if (!toolbar.isConnected) state.root.append(toolbar)
}

export function syncToolbarToggles(state: NotebookRuntimeState) {
  const debug = state.root.querySelector<HTMLButtonElement>('[data-notebook-debug]')
  if (debug) {
    setNotebookToggleButtonState(
      debug,
      state.debug,
      state.debug ? 'Disable debug output' : 'Enable debug output',
    )
  }
  const vim = state.root.querySelector<HTMLButtonElement>('[data-notebook-vim-mode]')
  if (vim) setNotebookVimButtonState(vim, state.vimMode)
  for (const button of state.cellRoot.querySelectorAll<HTMLButtonElement>(
    '[data-notebook-vim-cell]',
  )) {
    setNotebookVimButtonState(button, state.vimMode)
  }
}

export function toggleDebug(state: NotebookRuntimeState) {
  state.debug = !state.debug
  syncToolbarToggles(state)
}

export function toggleVimMode(state: NotebookRuntimeState) {
  state.vimMode = !state.vimMode
  writeStoredVimMode(state.vimMode)
  syncToolbarToggles(state)
  for (const controls of state.sourceControls.values()) {
    void controls.editor?.setVimMode(state.vimMode).catch(error => {
      setStatus(state, error instanceof Error ? error.message : 'failed to toggle vim mode')
    })
  }
}

export function addCleanup(callback: () => void) {
  const runtimeWindow = window as Window & { addCleanup?: (callback: () => void) => void }
  if (typeof runtimeWindow.addCleanup === 'function') {
    runtimeWindow.addCleanup(callback)
  } else {
    window.addEventListener('beforeunload', callback, { once: true })
  }
}

export function ensureCellControls(cell: RuntimeCell, controls: HTMLElement) {
  if (
    !controls.querySelector<HTMLElement>(`[data-notebook-execution-label="${CSS.escape(cell.id)}"]`)
  ) {
    const label = document.createElement('span')
    label.className = 'notebook-execution-prompt'
    label.dataset.notebookExecutionLabel = cell.id
    label.setAttribute('aria-live', 'polite')
    controls.append(label)
  }
}

export function setStatus(state: NotebookRuntimeState, value: string) {
  const status = state.root.querySelector<HTMLElement>('[data-notebook-status]')
  if (status) status.textContent = value
}

export function setRunningControls(state: NotebookRuntimeState, running: boolean) {
  state.root
    .querySelector<HTMLButtonElement>('[data-notebook-stop]')
    ?.toggleAttribute('disabled', !running)
  state.root
    .querySelector<HTMLButtonElement>('[data-notebook-run-all]')
    ?.toggleAttribute('disabled', running)
  queryCells<HTMLButtonElement>(state, '[data-notebook-run-cell]').forEach(button => {
    const cellId = button.dataset.notebookRunCell
    const active = running && cellId === state.runningCellId
    button.toggleAttribute('disabled', running && !active)
    if (running && active && cellId) {
      setNotebookIconButton(button, 'stop', `Interrupt ${cellId}`)
    } else if (cellId) {
      setNotebookIconButton(button, 'run', `Run ${cellId}`)
    }
  })
}
