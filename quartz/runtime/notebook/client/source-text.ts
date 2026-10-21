import type { NotebookRuntimeState, RuntimeCell, SourceControls } from './runtime-state'
import { notebookCellLanguageBadge } from '../../../util/notebook/cell-html'
import { notebookRuntimeLocalSourceKey } from '../types'
import { runtimeCellEditorLanguage } from './payload-and-warmup'

export function notebookLanguageBadgeElement(language: string): HTMLElement {
  const template = document.createElement('template')
  template.innerHTML = notebookCellLanguageBadge(language)
  const badge = template.content.firstElementChild
  if (badge instanceof HTMLElement) return badge

  const fallback = document.createElement('span')
  fallback.className = 'notebook-language-badge notebook-language-badge-code'
  fallback.dataset.notebookLanguage = 'code'
  fallback.setAttribute('aria-label', 'Code cell')
  fallback.title = 'Code cell'

  const icon = document.createElement('span')
  icon.className = 'notebook-language-icon'
  icon.setAttribute('aria-hidden', 'true')
  icon.textContent = 'code'

  fallback.append(icon)
  return fallback
}

export function replaceRenderedSource(
  figure: HTMLElement,
  source: string,
  highlightedLines: HTMLElement[] | undefined,
) {
  const pre = figure.querySelector('pre')
  if (!pre) {
    figure.textContent = source
    return
  }
  const code = pre.querySelector('code')
  if (!code) {
    pre.textContent = source
    return
  }

  code.dataset.clipboard = source
  code.replaceChildren()
  const lines = source.split(/\r?\n/)
  if (highlightedLines && highlightedLines.length === lines.length) {
    highlightedLines.forEach((line, index) => {
      line.dataset.line = ''
      code.append(line)
      if (index < lines.length - 1) code.append(document.createTextNode('\n'))
    })
    return
  }
  lines.forEach((line, index) => {
    const span = document.createElement('span')
    span.dataset.line = ''
    span.textContent = line
    code.append(span)
    if (index < lines.length - 1) code.append(document.createTextNode('\n'))
  })
}

export function renderedFigureSource(figure: HTMLElement): string | undefined {
  const lines = Array.from(figure.querySelectorAll<HTMLElement>('pre code [data-line]'))
  if (lines.length > 0) return lines.map(line => line.textContent ?? '').join('\n')
  const code = figure.querySelector('pre code')
  return code?.textContent ?? undefined
}

export function sourceForCell(state: NotebookRuntimeState, cell: RuntimeCell): string {
  const controls = state.sourceControls.get(cell.id)
  if (controls && !controls.editorHost.hidden && controls.editor) return controls.editor.getValue()
  if (controls) return controls.source
  return readStoredSource(state, cell) ?? cell.source
}

export function sourceSnapshotForCell(state: NotebookRuntimeState, cell: RuntimeCell): string {
  const controls = state.sourceControls.get(cell.id)
  if (controls) return controls.source
  return readStoredSource(state, cell) ?? cell.source
}

export function sourceStorageKey(state: NotebookRuntimeState, cell: RuntimeCell): string {
  return notebookRuntimeLocalSourceKey(state.payload.sourcePath, cell.id)
}

export function readStoredSource(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
): string | undefined {
  try {
    const value = window.localStorage.getItem(sourceStorageKey(state, cell))
    return value === null ? undefined : value
  } catch {
    return undefined
  }
}

export function writeStoredSource(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
  source: string,
): boolean {
  try {
    window.localStorage.setItem(sourceStorageKey(state, cell), source)
    const controls = state.sourceControls.get(cell.id)
    if (controls) controls.storedSource = source
    return true
  } catch {
    return false
  }
}

export function clearStoredSource(state: NotebookRuntimeState, cell: RuntimeCell) {
  try {
    window.localStorage.removeItem(sourceStorageKey(state, cell))
  } catch {}
  const controls = state.sourceControls.get(cell.id)
  if (controls) controls.storedSource = undefined
}

export async function highlightedSourceLines(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
  controls: SourceControls,
  source: string,
): Promise<HTMLElement[] | undefined> {
  const lineCount = source.split(/\r?\n/).length
  if (!controls.figure) return undefined
  try {
    const { renderNotebookHighlightedLines } = await state.loadNotebookCodeEditor()
    const highlightedLines = await renderNotebookHighlightedLines(
      source,
      runtimeCellEditorLanguage(cell),
    )
    return highlightedLines.length === lineCount ? highlightedLines : undefined
  } catch {
    return undefined
  }
}

export async function replaceRenderedCellSource(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
  controls: SourceControls,
  source: string,
) {
  if (!controls.figure || controls.renderedSource === source) return
  const highlightedLines = await highlightedSourceLines(state, cell, controls, source)
  replaceRenderedSource(controls.figure, source, highlightedLines)
  controls.renderedSource = source
}
