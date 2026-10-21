import type { CellId } from '../../../util/notebook/types'
import type { RuntimeCell, RuntimePayload } from './runtime-state'
import { notebookIconSvg, type NotebookIcon } from '../../../util/notebook/render/icons'
import { isRecord, readString } from '../../../util/type-guards'
import { backendFor } from '../registry'

type NotebookRuntimePreloadPayload = {
  readonly language: string
  readonly cells: readonly { readonly language: string }[]
}

export const notebookRuntimeVimModeKey = 'quartz:notebook-vim-mode'

export function setNotebookIconButton(
  button: HTMLButtonElement,
  icon: NotebookIcon,
  label: string,
) {
  button.classList.add('notebook-icon-button')
  button.setAttribute('aria-label', label)
  button.title = label
  button.dataset.notebookTooltip = label
  button.textContent = ''
  button.insertAdjacentHTML('afterbegin', notebookIconSvg[icon])
}

export function setNotebookToggleButtonState(
  button: HTMLButtonElement,
  enabled: boolean,
  label: string,
) {
  button.setAttribute('aria-pressed', String(enabled))
  button.setAttribute('aria-label', label)
  button.title = label
  button.dataset.notebookTooltip = label
}

export function setNotebookVimButtonState(button: HTMLButtonElement, enabled: boolean) {
  const label = enabled ? 'Disable Vim mode' : 'Enable Vim mode'
  setNotebookToggleButtonState(button, enabled, label)
}

export function readStoredNotebookVimMode(): boolean {
  try {
    return window.localStorage.getItem(notebookRuntimeVimModeKey) === 'true'
  } catch {
    return false
  }
}

export function readRuntimeCell(value: unknown): RuntimeCell | undefined {
  if (!isRecord(value)) return undefined
  const id = readString(value, 'id')
  const source = readString(value, 'source')
  const language = readString(value, 'language')
  const displayLanguage = readString(value, 'displayLanguage')
  const executionIndex = value.executionIndex
  if (!isRuntimeCellId(id) || source === undefined || !language) return undefined
  if (typeof executionIndex === 'number' || executionIndex === null) {
    return displayLanguage
      ? { id, source, language, displayLanguage, executionIndex }
      : { id, source, language, executionIndex }
  }
}

export function readBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

export function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined
  const strings = value.filter((item): item is string => typeof item === 'string')
  if (strings.length !== value.length) return undefined
  return strings
}

export function isRuntimeCellId(value: unknown): value is CellId {
  return typeof value === 'string' && value.length > 0
}

export function readRuntimePayload(value: unknown): RuntimePayload | undefined {
  if (!isRecord(value)) return undefined
  const id = readString(value, 'id')
  const sourcePath = readString(value, 'sourcePath')
  const language = readString(value, 'language')
  const indexUrl = readString(value, 'indexUrl')
  if (!id || !sourcePath || !language || !Array.isArray(value.cells)) {
    return undefined
  }
  const cells = value.cells.map(readRuntimeCell)
  if (cells.some(cell => cell === undefined)) return undefined
  const payload: RuntimePayload = {
    id,
    sourcePath,
    language,
    cells: cells.filter(cell => cell !== undefined),
  }
  if (indexUrl !== undefined) payload.indexUrl = indexUrl
  const toolbar = readBoolean(value.toolbar)
  if (toolbar !== undefined) payload.toolbar = toolbar
  const debug = readBoolean(value.debug)
  if (debug !== undefined) payload.debug = debug
  const vimMode = readBoolean(value.vimMode)
  if (vimMode !== undefined) payload.vimMode = vimMode
  if (value.importableModules !== undefined) {
    const importableModules = readStringArray(value.importableModules)
    if (importableModules === undefined) return undefined
    payload.importableModules = importableModules
  }
  return payload
}

export function runtimeCellEditorLanguage(cell: RuntimeCell): string {
  return cell.displayLanguage ?? cell.language
}

export function notebookRuntimePreloadLanguages(payload: NotebookRuntimePreloadPayload): string[] {
  return notebookRuntimeKernelLanguages(payload).filter(language => {
    const backend = backendFor(language)
    return backend?.preload !== false
  })
}

export function notebookRuntimeKernelLanguages(payload: NotebookRuntimePreloadPayload): string[] {
  const languages: string[] = []
  const seen = new Set<string>()
  for (const cell of payload.cells) {
    const backend = backendFor(cell.language)
    if (!backend || seen.has(backend.name)) continue
    seen.add(backend.name)
    languages.push(backend.name)
  }
  return languages
}

export function decodeRuntimeCodePoint(codePoint: number, fallback: string): string {
  if (!Number.isInteger(codePoint) || codePoint < 0 || codePoint > 0x10ffff) return fallback
  try {
    return String.fromCodePoint(codePoint)
  } catch {
    return fallback
  }
}

export function decodeRuntimeHtmlEntities(text: string): string {
  return text.replace(
    /&(?:#(\d+)|#x([0-9a-fA-F]+)|quot|apos|amp|lt|gt);/g,
    (entity: string, decimal?: string, hex?: string) => {
      if (decimal !== undefined) {
        return decodeRuntimeCodePoint(Number.parseInt(decimal, 10), entity)
      }
      if (hex !== undefined) {
        return decodeRuntimeCodePoint(Number.parseInt(hex, 16), entity)
      }
      if (entity === '&quot;') return '"'
      if (entity === '&apos;') return "'"
      if (entity === '&amp;') return '&'
      if (entity === '&lt;') return '<'
      if (entity === '&gt;') return '>'
      return entity
    },
  )
}

export function parseRuntimeJson(text: string): unknown | undefined {
  try {
    return JSON.parse(text)
  } catch {}

  const decoded = decodeRuntimeHtmlEntities(text)
  if (decoded === text) return undefined

  try {
    return JSON.parse(decoded)
  } catch {
    return undefined
  }
}

export function scheduleNotebookRuntimePreload(callback: () => void) {
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(callback, { timeout: 1500 })
    return
  }
  window.setTimeout(callback, 750)
}

export function notebookRuntimeWasDisposed(error: unknown): boolean {
  return error instanceof Error && error.message === 'notebook runtime was disposed'
}

export function readStoredVimMode(): boolean {
  return readStoredNotebookVimMode()
}

export function writeStoredVimMode(enabled: boolean) {
  try {
    window.localStorage.setItem(notebookRuntimeVimModeKey, enabled ? 'true' : 'false')
  } catch {}
}
