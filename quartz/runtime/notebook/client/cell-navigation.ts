import type { NotebookRuntimeState, RuntimeCell } from './runtime-state'
import {
  activeNotebookCellFrame,
  clearActiveNotebookCellFrames,
  notebookCellFrameFromElement,
  notebookCellFrameId,
  notebookCellFrameIsActive,
  selectNotebookCellFrame,
} from '../../../util/notebook-active-cell'

type NotebookRunKeyEvent = Pick<
  KeyboardEvent,
  'key' | 'metaKey' | 'ctrlKey' | 'shiftKey' | 'altKey'
>

export function notebookRunAndAdvanceKey(event: NotebookRunKeyEvent): boolean {
  return (
    event.key === 'Enter' && event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey
  )
}

export function notebookRunKey(event: NotebookRunKeyEvent): boolean {
  return event.key === 'Enter' && (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
}

export function nextNotebookCellId(
  cells: readonly { readonly id: string }[],
  cellId: string,
): string | undefined {
  const index = cells.findIndex(cell => cell.id === cellId)
  return index === -1 ? undefined : cells[index + 1]?.id
}

export function queryCell<T extends Element>(
  state: NotebookRuntimeState,
  selector: string,
): T | null {
  return state.cellRoot.querySelector<T>(selector)
}

export function queryCells<T extends Element>(
  state: NotebookRuntimeState,
  selector: string,
): NodeListOf<T> {
  return state.cellRoot.querySelectorAll<T>(selector)
}

export function cellById(
  state: NotebookRuntimeState,
  cellId: string | undefined,
): RuntimeCell | undefined {
  if (cellId === undefined) return undefined
  return state.payload.cells.find(cell => cell.id === cellId)
}

export function cellFrame(state: NotebookRuntimeState, cellId: string): HTMLElement | null {
  return queryCell<HTMLElement>(state, `[data-notebook-cell-frame="${CSS.escape(cellId)}"]`)
}

export function cellFromFrame(
  state: NotebookRuntimeState,
  frame: HTMLElement | undefined,
): RuntimeCell | undefined {
  if (!frame || !state.cellRoot.contains(frame)) return undefined
  return cellById(state, notebookCellFrameId(frame))
}

export function selectCell(state: NotebookRuntimeState, cellId: string) {
  const frame = cellFrame(state, cellId)
  if (!frame) return
  selectNotebookCellFrame(frame)
  state.activeCellId = cellId
}

export function clearCellSelection(state: NotebookRuntimeState) {
  if (state.activeCellId === undefined) return
  clearActiveNotebookCellFrames(document)
  state.activeCellId = undefined
}

export function cellFromTarget(
  state: NotebookRuntimeState,
  target: EventTarget | null,
): RuntimeCell | undefined {
  if (!(target instanceof Element)) return undefined
  return cellFromFrame(state, notebookCellFrameFromElement(target))
}

export function activeCellFromDocument(
  state: NotebookRuntimeState,
  frame = activeNotebookCellFrame(document),
): RuntimeCell | undefined {
  if (!frame) return undefined
  const cell = cellFromFrame(state, frame)
  state.activeCellId = cell?.id
  return cell
}

export function activeCellFromCache(state: NotebookRuntimeState): RuntimeCell | undefined {
  if (state.activeCellId === undefined) return undefined
  const frame = cellFrame(state, state.activeCellId)
  if (!frame || !notebookCellFrameIsActive(frame)) {
    state.activeCellId = undefined
    return undefined
  }
  return cellById(state, state.activeCellId)
}

export function commandCell(
  state: NotebookRuntimeState,
  target: EventTarget | null,
): RuntimeCell | undefined {
  const activeFrame = activeNotebookCellFrame(document)
  const activeCell = activeCellFromDocument(state, activeFrame)
  if (activeFrame) return activeCell
  return cellFromTarget(state, target) ?? activeCellFromCache(state)
}

export function targetOwnsKeyboard(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false
  if (target.closest('.cm-editor')) return true
  if (target.closest('input, textarea, select, button, a[href], [contenteditable]')) return true
  return false
}

export function siteShortcutLayerActive(): boolean {
  const palette = document.getElementById('palette-container')
  if (palette?.classList.contains('active')) return true
  const shortcuts = document.getElementById('shortcut-container')
  if (shortcuts?.classList.contains('active')) return true
  const search = document.querySelector('.search .search-container.active')
  if (search) return true
  return document.querySelector<HTMLDialogElement>('dialog.headings-modal')?.open ?? false
}

export function claimNotebookKey(event: KeyboardEvent) {
  event.preventDefault()
  event.stopImmediatePropagation()
}

export function clearEditPrefix(state: NotebookRuntimeState) {
  if (state.editPrefixTimeout !== undefined) window.clearTimeout(state.editPrefixTimeout)
  state.editPrefixTimeout = undefined
}

export function armEditPrefix(state: NotebookRuntimeState) {
  clearEditPrefix(state)
  state.editPrefixTimeout = window.setTimeout(() => {
    state.editPrefixTimeout = undefined
  }, 800)
}

export function nextCell(state: NotebookRuntimeState, cell: RuntimeCell): RuntimeCell | undefined {
  return cellById(state, nextNotebookCellId(state.payload.cells, cell.id))
}

export function focusCell(state: NotebookRuntimeState, cell: RuntimeCell) {
  selectCell(state, cell.id)
  cellFrame(state, cell.id)?.focus()
}
