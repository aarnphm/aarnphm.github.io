import DOMPurify from 'dompurify'
import type {
  NotebookKernelCommandDetail,
  NotebookKernelRequestDetail,
  NotebookKernelSnapshot,
  NotebookKernelStatus,
} from '../../util/notebook-kernel-events'
import type { CellId } from '../../util/notebook/types'
import type { NotebookCodeEditor } from '../editor/code-editor'
import type { NotebookLspConfig } from '../lsp/pyright'
import type { NotebookRuntimeAssetConfig } from './assets'
import type { ExecutableLanguageBackend, RuntimeModuleResolver } from './backend'
import type {
  Kernel,
  KernelOutput,
  NotebookModule,
  RuntimeAssetRequest,
  RuntimeAssetResult,
  RuntimeDownload,
  RuntimeEvent,
  RuntimeFileResult,
} from './kernel'
import {
  activeNotebookCellFrame,
  clearActiveNotebookCellFrames,
  notebookCellFrameFromElement,
  notebookCellFrameId,
  notebookCellFrameIsActive,
  selectNotebookCellFrame,
} from '../../util/notebook-active-cell'
import {
  notebookKernelCommandEvent,
  notebookKernelRequestEvent,
  notebookKernelRunAllEvent,
} from '../../util/notebook-kernel-events'
import { notebookLocalSourcesClearedEvent } from '../../util/notebook-source-events'
import { notebookCellLanguageBadge } from '../../util/notebook/cell-html'
import { renderOutputHtml } from '../../util/notebook/render/output-to-hast'
import { supportsEagerRuntimePreload } from '../../util/runtime-preload'
import { isRecord, readString } from '../../util/type-guards'
import { configureNotebookRuntimeAssets } from './assets'
import { backendFor } from './registry'
import {
  notebookRuntimeLocalSourceKey,
  notebookSuccessOutputLabel,
  type NotebookRuntimeDebugOutput,
  type NotebookRuntimeOutput,
} from './types'

type RuntimeCell = {
  id: CellId
  source: string
  language: string
  displayLanguage?: string
  executionIndex: number | null
}

type RuntimeErrorOutput = Extract<NotebookRuntimeOutput, { type: 'error' }>

type RuntimeStreamOutput = Extract<NotebookRuntimeOutput, { type: 'stream' }>

type RuntimeDebugOutput = NonNullable<RuntimeErrorOutput['debug']>

type CellRunResult = { failed: boolean }

type KernelPresence = { readonly status: NotebookKernelStatus; readonly statusDetail?: string }

type SourceControls = {
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

type RuntimePayload = {
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

export type NotebookRuntimePreloadPayload = {
  readonly language: string
  readonly cells: readonly { readonly language: string }[]
}

export type NotebookRunKeyEvent = Pick<
  KeyboardEvent,
  'key' | 'metaKey' | 'ctrlKey' | 'shiftKey' | 'altKey'
>

import { notebookIconSvg, type NotebookIcon } from '../../util/notebook/render/icons'

const notebookRuntimeVimModeKey = 'quartz:notebook-vim-mode'

function setNotebookIconButton(button: HTMLButtonElement, icon: NotebookIcon, label: string) {
  button.classList.add('notebook-icon-button')
  button.setAttribute('aria-label', label)
  button.title = label
  button.dataset.notebookTooltip = label
  button.textContent = ''
  button.insertAdjacentHTML('afterbegin', notebookIconSvg[icon])
}

function setNotebookToggleButtonState(button: HTMLButtonElement, enabled: boolean, label: string) {
  button.setAttribute('aria-pressed', String(enabled))
  button.setAttribute('aria-label', label)
  button.title = label
  button.dataset.notebookTooltip = label
}

function setNotebookVimButtonState(button: HTMLButtonElement, enabled: boolean) {
  const label = enabled ? 'Disable Vim mode' : 'Enable Vim mode'
  setNotebookToggleButtonState(button, enabled, label)
}

function notebookLanguageBadgeElement(language: string): HTMLElement {
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

function readStoredNotebookVimMode(): boolean {
  try {
    return window.localStorage.getItem(notebookRuntimeVimModeKey) === 'true'
  } catch {
    return false
  }
}

function readRuntimeCell(value: unknown): RuntimeCell | undefined {
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

function readBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined
  const strings = value.filter((item): item is string => typeof item === 'string')
  if (strings.length !== value.length) return undefined
  return strings
}

function isRuntimeCellId(value: unknown): value is CellId {
  return typeof value === 'string' && value.length > 0
}

function readRuntimePayload(value: unknown): RuntimePayload | undefined {
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

function runtimeCellEditorLanguage(cell: RuntimeCell): string {
  return cell.displayLanguage ?? cell.language
}

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

function decodeRuntimeCodePoint(codePoint: number, fallback: string): string {
  if (!Number.isInteger(codePoint) || codePoint < 0 || codePoint > 0x10ffff) return fallback
  try {
    return String.fromCodePoint(codePoint)
  } catch {
    return fallback
  }
}

function decodeRuntimeHtmlEntities(text: string): string {
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

function parseRuntimeJson(text: string): unknown | undefined {
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

function replaceRenderedSource(
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

function renderedFigureSource(figure: HTMLElement): string | undefined {
  const lines = Array.from(figure.querySelectorAll<HTMLElement>('pre code [data-line]'))
  if (lines.length > 0) return lines.map(line => line.textContent ?? '').join('\n')
  const code = figure.querySelector('pre code')
  return code?.textContent ?? undefined
}

function outputClassToken(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, '-') || 'output'
}

function debugOutputText(debug: NotebookRuntimeDebugOutput): string {
  return [
    ['phase', debug.phase],
    ['cell', debug.cellId],
    ['error', debug.errorName],
    ['message', debug.errorMessage],
    ['stack', debug.stack],
  ]
    .filter((entry): entry is [string, string] => entry[1] !== undefined && entry[1].length > 0)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n')
}

function createPreOutput(
  classes: string[],
  label: string,
  value: string,
  options: { trimEnd?: boolean } = {},
): HTMLPreElement {
  const pre = document.createElement('pre')
  pre.classList.add(...classes)
  pre.dataset.outputName = label
  const samp = document.createElement('samp')
  samp.textContent = options.trimEnd === false ? value : value.replace(/\s+$/, '')
  pre.appendChild(samp)
  return pre
}

function lastNotebookOutput(target: HTMLElement): HTMLElement | undefined {
  const direct = target.lastElementChild
  if (direct instanceof HTMLElement && direct.classList.contains('notebook-output')) return direct
  const tabbed = target.querySelectorAll<HTMLElement>(
    ':scope > [data-notebook-output-tabs] > .notebook-output-panels > .notebook-output-panel-frame > .notebook-output-panel > .notebook-output',
  )
  return tabbed[tabbed.length - 1]
}

function appendStreamOutput(target: HTMLElement, output: RuntimeStreamOutput) {
  const previous = lastNotebookOutput(target)
  if (
    previous instanceof HTMLPreElement &&
    previous.classList.contains('notebook-output-stream') &&
    previous.dataset.outputName === output.name
  ) {
    const sample = previous.querySelector('samp')
    if (sample) {
      sample.textContent += output.text
      previous.scrollTop = previous.scrollHeight
      syncOutputScrollHints(target)
      return
    }
  }
  target.appendChild(
    createPreOutput(
      [
        'notebook-output',
        'notebook-output-stream',
        `notebook-output-stream-${outputClassToken(output.name)}`,
      ],
      output.name,
      output.text,
      { trimEnd: false },
    ),
  )
}

function createHtmlOutput(html: string): HTMLDivElement {
  const output = document.createElement('div')
  output.classList.add('notebook-output', 'notebook-output-html')
  output.dataset.outputName = 'display'
  output.append(
    DOMPurify.sanitize(html, {
      FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'link', 'meta', 'base'],
      FORBID_ATTR: ['srcdoc'],
      ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto|tel|cid):|[^a-z]|[a-z+.-]+(?:[^a-z+.:-]|$))/i,
      RETURN_DOM_FRAGMENT: true,
    }),
  )
  return output
}

function createSuccessOutput(): HTMLDivElement {
  const output = document.createElement('div')
  output.classList.add('notebook-output', 'notebook-output-success')
  output.dataset.outputName = notebookSuccessOutputLabel
  return output
}

function outputLabel(output: HTMLElement): string {
  return output.dataset.outputName?.trim() || 'output'
}

function outputTabId(label: string): string {
  return outputClassToken(label.toLowerCase())
}

function collectOutputElements(target: HTMLElement): HTMLElement[] {
  const direct = Array.from(target.children).filter(
    (child): child is HTMLElement =>
      child instanceof HTMLElement && child.classList.contains('notebook-output'),
  )
  const tabbed = Array.from(
    target.querySelectorAll<HTMLElement>(
      ':scope > [data-notebook-output-tabs] > .notebook-output-panels > .notebook-output-panel-frame > .notebook-output-panel > .notebook-output',
    ),
  )
  return [...tabbed, ...direct]
}

function hasRenderedOutput(target: HTMLElement): boolean {
  return collectOutputElements(target).length > 0
}

const outputScrollHintTargets = new WeakSet<HTMLElement>()

function syncOutputScrollHint(output: HTMLElement) {
  const slack = 1
  const container = output.closest<HTMLElement>('[data-notebook-output-tabs]')
  const expanded = container?.hasAttribute('data-notebook-output-expanded') ?? false
  const scrollable = !expanded && output.scrollHeight - output.clientHeight > slack
  output.toggleAttribute('data-notebook-scrollable', scrollable)
  output.toggleAttribute('data-notebook-scroll-before', scrollable && output.scrollTop > slack)
  output.toggleAttribute(
    'data-notebook-scroll-after',
    scrollable && output.scrollTop + output.clientHeight < output.scrollHeight - slack,
  )
  if (container) syncOutputExpandControls(container)
}

function syncOutputScrollHints(root: HTMLElement) {
  const outputs = root.matches('.notebook-output-panel, pre.notebook-output-stream')
    ? [root]
    : Array.from(
        root.querySelectorAll<HTMLElement>('.notebook-output-panel, pre.notebook-output-stream'),
      )
  for (const output of outputs) {
    if (!outputScrollHintTargets.has(output)) {
      output.addEventListener('scroll', () => syncOutputScrollHint(output), { passive: true })
      output.addEventListener('load', () => syncOutputScrollHint(output), { capture: true })
      outputScrollHintTargets.add(output)
    }
    requestAnimationFrame(() => syncOutputScrollHint(output))
  }
}

function showNotebookOutputToast(message: string) {
  const event: CustomEventMap['toast'] = new CustomEvent('toast', {
    detail: { message, containerId: 'notebook-output-toast-container' },
  })
  document.dispatchEvent(event)
}

function downloadFileName(filename: string): string {
  const name = filename.replaceAll('\\', '/').split('/').filter(Boolean).at(-1)?.trim()
  return name && name !== '.' && name !== '..' ? name : 'notebook-file'
}

function downloadNotebookFile(filename: string, contentType: string, bytes: ArrayBuffer): string {
  const name = downloadFileName(filename)
  const url = URL.createObjectURL(new Blob([bytes], { type: contentType }))
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.style.display = 'none'
  document.body.append(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000)
  return name
}

function outputCopyText(outputs: HTMLElement[]): string {
  return outputs
    .map(output => output.textContent ?? '')
    .join('\n')
    .trimEnd()
}

function selectedOutputTabId(container: HTMLElement): string | undefined {
  return container.querySelector<HTMLElement>('[data-notebook-output-tab][aria-selected="true"]')
    ?.dataset.notebookOutputTab
}

function activeOutputPanel(container: HTMLElement): HTMLElement | undefined {
  const selectedId = selectedOutputTabId(container)
  if (!selectedId) return undefined
  for (const frame of container.querySelectorAll<HTMLElement>('[data-notebook-output-panel]')) {
    if (frame.dataset.notebookOutputPanel !== selectedId) continue
    return frame.querySelector<HTMLElement>(':scope > .notebook-output-panel') ?? undefined
  }
}

function outputPanelCollapsedHeight(container: HTMLElement): number {
  const height = Number.parseFloat(
    window.getComputedStyle(container).getPropertyValue('--notebook-output-panel-height'),
  )
  return Number.isFinite(height) && height > 0 ? height : 150
}

function syncOutputExpandButton(button: HTMLButtonElement, expandable: boolean, expanded: boolean) {
  const label = expanded ? 'Collapse output' : 'Expand output'
  button.toggleAttribute('data-notebook-output-expandable', expandable)
  button.setAttribute('aria-expanded', String(expanded))
  button.setAttribute('aria-label', label)
  button.title = label
}

function syncOutputExpandControls(container: HTMLElement) {
  const activeId = selectedOutputTabId(container)
  const panel = activeOutputPanel(container)
  const expandable =
    panel !== undefined && panel.scrollHeight - outputPanelCollapsedHeight(container) > 1
  if (!expandable) container.removeAttribute('data-notebook-output-expanded')
  const expanded = expandable && container.hasAttribute('data-notebook-output-expanded')
  container.toggleAttribute('data-notebook-output-expandable', expandable)
  for (const button of container.querySelectorAll<HTMLButtonElement>(
    '[data-notebook-output-expand-action]',
  )) {
    const active = activeId !== undefined && button.dataset.notebookOutputExpandAction === activeId
    syncOutputExpandButton(button, active && expandable, active && expanded)
  }
}

function createOutputExpandButton(container: HTMLElement, groupId: string): HTMLButtonElement {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'notebook-output-expand-button'
  button.classList.add('notebook-icon-button')
  button.dataset.notebookOutputAction = groupId
  button.dataset.notebookOutputExpandAction = groupId
  syncOutputExpandButton(button, false, false)
  const expandIcon = document.createElement('span')
  expandIcon.className = 'notebook-output-expand-icon'
  expandIcon.insertAdjacentHTML('afterbegin', notebookIconSvg.expand)
  button.append(expandIcon)
  button.addEventListener('click', event => {
    event.preventDefault()
    event.stopPropagation()
    const expanded = !container.hasAttribute('data-notebook-output-expanded')
    container.toggleAttribute('data-notebook-output-expanded', expanded)
    syncOutputExpandControls(container)
    syncOutputScrollHints(container)
  })
  return button
}

function createOutputCopyButton(outputs: HTMLElement[]): HTMLButtonElement {
  const button = document.createElement('button')
  let resetCopiedState: number | undefined
  button.type = 'button'
  button.className = 'notebook-output-copy-button'
  button.classList.add('notebook-icon-button')
  button.setAttribute('aria-label', 'Copy output')
  button.title = 'Copy output'
  const copyIcon = document.createElement('span')
  copyIcon.className = 'notebook-output-copy-icon'
  copyIcon.insertAdjacentHTML('afterbegin', notebookIconSvg.copy)
  const checkIcon = document.createElement('span')
  checkIcon.className = 'notebook-output-check-icon'
  checkIcon.insertAdjacentHTML('afterbegin', notebookIconSvg.check)
  button.append(copyIcon, checkIcon)
  button.addEventListener('click', async event => {
    event.preventDefault()
    event.stopPropagation()
    const text = outputCopyText(outputs)
    if (text.length === 0) {
      showNotebookOutputToast('nothing to copy')
      return
    }
    try {
      await navigator.clipboard.writeText(text)
      if (resetCopiedState !== undefined) window.clearTimeout(resetCopiedState)
      button.classList.add('notebook-output-copy-button-copied')
      resetCopiedState = window.setTimeout(() => {
        button.classList.remove('notebook-output-copy-button-copied')
      }, 2000)
      showNotebookOutputToast('copied output')
    } catch {
      button.classList.remove('notebook-output-copy-button-copied')
      showNotebookOutputToast('copy failed')
    }
  })
  return button
}

function selectOutputTab(container: HTMLElement, activeId: string | undefined, focusId = activeId) {
  container.toggleAttribute('data-notebook-output-collapsed', activeId === undefined)
  const fallbackFocusId =
    focusId ??
    container.querySelector<HTMLButtonElement>('[data-notebook-output-tab]')?.dataset
      .notebookOutputTab
  for (const tab of container.querySelectorAll<HTMLButtonElement>('[data-notebook-output-tab]')) {
    if (tab.disabled) {
      tab.setAttribute('aria-selected', 'false')
      tab.removeAttribute('aria-expanded')
      tab.tabIndex = -1
      continue
    }
    const active = activeId !== undefined && tab.dataset.notebookOutputTab === activeId
    tab.setAttribute('aria-selected', String(active))
    tab.setAttribute('aria-expanded', String(active))
    tab.tabIndex = tab.dataset.notebookOutputTab === fallbackFocusId ? 0 : -1
  }
  for (const panel of container.querySelectorAll<HTMLElement>('[data-notebook-output-panel]')) {
    panel.hidden = activeId === undefined || panel.dataset.notebookOutputPanel !== activeId
  }
  for (const action of container.querySelectorAll<HTMLElement>('[data-notebook-output-action]')) {
    action.hidden = activeId === undefined || action.dataset.notebookOutputAction !== activeId
  }
  if (activeId === undefined) container.removeAttribute('data-notebook-output-expanded')
  syncOutputExpandControls(container)
  if (activeId !== undefined) syncOutputScrollHints(container)
}

function syncOutputTabs(target: HTMLElement, cellId = target.dataset.notebookOutput ?? 'cell') {
  const outputs = collectOutputElements(target)
  if (outputs.length === 0) {
    target.removeAttribute('data-notebook-output-tabbed')
    return
  }

  const groups = new Map<string, { id: string; label: string; outputs: HTMLElement[] }>()
  for (const output of outputs) {
    const label = outputLabel(output)
    const id = outputTabId(label)
    const group = groups.get(id) ?? { id, label, outputs: [] }
    group.outputs.push(output)
    groups.set(id, group)
  }
  const previousContainer = target.querySelector<HTMLElement>(
    ':scope > [data-notebook-output-tabs]',
  )
  const previousActive = previousContainer?.querySelector<HTMLElement>(
    '[data-notebook-output-tab][aria-selected="true"]',
  )?.dataset.notebookOutputTab
  const wasCollapsed = previousContainer?.hasAttribute('data-notebook-output-collapsed') ?? true
  const orderedGroups = [...groups.values()]
  const defaultActiveId = orderedGroups.find(
    group => group.label !== notebookSuccessOutputLabel,
  )?.id
  const selectableGroups = orderedGroups.filter(group => group.label !== notebookSuccessOutputLabel)
  const activeId =
    previousContainer === null
      ? defaultActiveId
      : !wasCollapsed && previousActive && orderedGroups.some(group => group.id === previousActive)
        ? previousActive
        : undefined

  const container = document.createElement('div')
  container.className = 'notebook-output-tabs'
  container.dataset.notebookOutputTabs = ''
  const tablist = document.createElement('div')
  tablist.className = 'notebook-output-tablist'
  tablist.setAttribute('role', 'tablist')
  tablist.setAttribute('aria-orientation', 'horizontal')
  const actions = document.createElement('div')
  actions.className = 'notebook-output-actions'
  const panels = document.createElement('div')
  panels.className = 'notebook-output-panels'
  const outputId = outputClassToken(cellId)

  orderedGroups.forEach((group, index) => {
    const tabId = `notebook-output-${outputId}-${index}-tab`
    const panelId = `notebook-output-${outputId}-${index}-panel`
    const tab = document.createElement('button')
    tab.type = 'button'
    tab.className = 'notebook-output-tab'
    tab.dataset.notebookOutputTab = group.id
    tab.id = tabId
    tab.textContent = group.label
    tab.setAttribute('role', 'tab')
    tab.setAttribute('aria-controls', panelId)
    tab.disabled = group.label === notebookSuccessOutputLabel
    tab.toggleAttribute('data-notebook-output-disabled', tab.disabled)
    if (tab.disabled) {
      tab.setAttribute('aria-disabled', 'true')
    }
    tab.addEventListener('click', () => {
      if (tab.disabled) return
      const active = tab.getAttribute('aria-selected') === 'true'
      selectOutputTab(container, active ? undefined : group.id, group.id)
    })
    tab.addEventListener('keydown', event => {
      if (tab.disabled) return
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        const active = tab.getAttribute('aria-selected') === 'true'
        selectOutputTab(container, active ? undefined : group.id, group.id)
        return
      }
      if (
        event.key !== 'ArrowDown' &&
        event.key !== 'ArrowRight' &&
        event.key !== 'ArrowUp' &&
        event.key !== 'ArrowLeft'
      ) {
        return
      }
      event.preventDefault()
      const delta = event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1 : -1
      const currentIndex = selectableGroups.findIndex(
        selectableGroup => selectableGroup.id === group.id,
      )
      if (currentIndex === -1) return
      const nextGroup =
        selectableGroups[(currentIndex + delta + selectableGroups.length) % selectableGroups.length]
      if (!nextGroup) return
      const nextTab = tablist.querySelector<HTMLButtonElement>(
        `[data-notebook-output-tab="${nextGroup.id}"]`,
      )
      if (!nextTab) return
      if (container.hasAttribute('data-notebook-output-collapsed')) {
        selectOutputTab(container, undefined, nextGroup.id)
      } else {
        selectOutputTab(container, nextGroup.id)
      }
      nextTab.focus()
    })
    tablist.append(tab)

    const panelFrame = document.createElement('div')
    panelFrame.className = 'notebook-output-panel-frame'
    panelFrame.dataset.notebookOutputPanel = group.id
    panelFrame.id = panelId
    panelFrame.setAttribute('role', 'tabpanel')
    panelFrame.setAttribute('aria-labelledby', tabId)
    const panel = document.createElement('div')
    panel.className = 'notebook-output-panel'
    panel.append(...group.outputs)
    panelFrame.append(panel)
    actions.append(createOutputExpandButton(container, group.id))
    const copyButton = createOutputCopyButton(group.outputs)
    copyButton.dataset.notebookOutputAction = group.id
    actions.append(copyButton)
    panels.append(panelFrame)
  })

  container.append(tablist, actions, panels)
  target.dataset.notebookOutputTabbed = ''
  target.replaceChildren(container)
  selectOutputTab(container, activeId)
}

function syncStaticOutputTabs(frame: HTMLElement, cellId: string) {
  const existing = frame.querySelector<HTMLElement>(':scope > [data-notebook-static-output]')
  if (existing) {
    syncOutputTabs(existing, cellId)
    return
  }
  const outputs = Array.from(frame.children).filter(
    (child): child is HTMLElement =>
      child instanceof HTMLElement && child.classList.contains('notebook-output'),
  )
  if (outputs.length === 0) return
  const container = document.createElement('div')
  container.className = 'notebook-static-output'
  container.dataset.notebookStaticOutput = cellId
  outputs[0]?.before(container)
  container.append(...outputs)
  syncOutputTabs(container, cellId)
}

function scheduleNotebookRuntimePreload(callback: () => void) {
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(callback, { timeout: 1500 })
    return
  }
  window.setTimeout(callback, 750)
}

function notebookRuntimeWasDisposed(error: unknown): boolean {
  return error instanceof Error && error.message === 'notebook runtime was disposed'
}

function appendRuntimeOutput(
  target: HTMLElement,
  output: NotebookRuntimeOutput,
  options: { debug?: boolean } = {},
) {
  if (output.type === 'stream') {
    appendStreamOutput(target, output)
    syncOutputTabs(target)
    return
  }

  if (output.type === 'error') {
    const text = output.traceback || [output.ename, output.evalue].filter(Boolean).join(': ')
    target.appendChild(createPreOutput(['notebook-output', 'notebook-output-error'], 'error', text))
    if (options.debug === true && output.debug) {
      target.appendChild(
        createPreOutput(
          ['notebook-output', 'notebook-output-debug'],
          'debug',
          debugOutputText(output.debug),
        ),
      )
    }
    syncOutputTabs(target)
    return
  }

  if (output.type === 'html') {
    target.appendChild(createHtmlOutput(output.html))
    syncOutputTabs(target)
    return
  }

  if (output.type === 'json') {
    target.appendChild(
      createPreOutput(
        ['notebook-output', 'notebook-output-text', 'notebook-output-json'],
        'json',
        output.text,
      ),
    )
    syncOutputTabs(target)
    return
  }

  if (output.type === 'success') {
    target.appendChild(createSuccessOutput())
    syncOutputTabs(target)
    return
  }

  target.appendChild(
    createPreOutput(['notebook-output', 'notebook-output-text'], 'result', output.text),
  )
  syncOutputTabs(target)
}

class NotebookRuntime {
  private payload: RuntimePayload
  private root: HTMLElement
  private cellRoot: Document | HTMLElement
  private assets: NotebookRuntimeAssetConfig
  private kernels = new Map<string, Kernel>()
  private kernelRequests = new Map<string, Promise<Kernel>>()
  private kernelEpochs = new Map<string, number>()
  private kernelPresences = new Map<string, KernelPresence>()
  private runningKernel: Kernel | undefined
  private runningBackendName: string | undefined
  private moduleCache = new Map<string, NotebookModule | null>()
  private moduleFetches = new Map<string, Promise<NotebookModule | null>>()
  private importableModules: Set<string> | undefined
  private savedOutputs = new Map<string, HTMLElement[]>()
  private sourceControls = new Map<string, SourceControls>()
  private executionCounter: number
  private running = false
  private stopped = false
  private runningCellId: string | undefined
  private debug = false
  private vimMode: boolean
  private activeCellId: string | undefined
  private editPrefixTimeout: number | undefined
  private runtimePreload: Promise<void> | undefined
  private runtimeEpoch = 0
  private disposed = false

  constructor(root: HTMLElement, payload: RuntimePayload, assets: NotebookRuntimeAssetConfig) {
    this.root = root
    this.cellRoot = root.closest('article') ?? root.parentElement ?? document
    this.assets = assets
    this.payload = payload
    this.importableModules = payload.importableModules
      ? new Set(payload.importableModules)
      : undefined
    this.executionCounter = Math.max(0, ...payload.cells.map(cell => cell.executionIndex ?? 0))
    this.debug = this.payload.debug ?? false
    this.vimMode = this.payload.vimMode ?? this.readStoredVimMode()
  }

  mount() {
    if (this.payload.toolbar === false) {
      this.root.querySelector<HTMLElement>('[data-notebook-runtime-toolbar]')?.remove()
    } else {
      this.ensureToolbar()
    }
    this.decorateCells()
    const runAll = this.root.querySelector<HTMLButtonElement>('[data-notebook-run-all]')
    const stop = this.root.querySelector<HTMLButtonElement>('[data-notebook-stop]')
    const reset = this.root.querySelector<HTMLButtonElement>('[data-notebook-reset]')
    const debug = this.root.querySelector<HTMLButtonElement>('[data-notebook-debug]')
    const vim = this.root.querySelector<HTMLButtonElement>('[data-notebook-vim-mode]')
    runAll?.addEventListener('click', this.runAll)
    stop?.addEventListener('click', this.stop)
    reset?.addEventListener('click', this.reset)
    debug?.addEventListener('click', this.toggleDebug)
    vim?.addEventListener('click', this.toggleVimMode)
    document.addEventListener('keydown', this.handleNotebookKeydown, true)
    document.addEventListener(notebookKernelRequestEvent, this.handleKernelRequest)
    document.addEventListener(notebookKernelCommandEvent, this.handleKernelCommand)
    document.addEventListener(notebookKernelRunAllEvent, this.handleKernelRunAll)
    document.addEventListener(notebookLocalSourcesClearedEvent, this.handleLocalSourcesCleared)
    this.addCleanup(() => {
      runAll?.removeEventListener('click', this.runAll)
      stop?.removeEventListener('click', this.stop)
      reset?.removeEventListener('click', this.reset)
      debug?.removeEventListener('click', this.toggleDebug)
      vim?.removeEventListener('click', this.toggleVimMode)
      document.removeEventListener('keydown', this.handleNotebookKeydown, true)
      document.removeEventListener(notebookKernelRequestEvent, this.handleKernelRequest)
      document.removeEventListener(notebookKernelCommandEvent, this.handleKernelCommand)
      document.removeEventListener(notebookKernelRunAllEvent, this.handleKernelRunAll)
      document.removeEventListener(notebookLocalSourcesClearedEvent, this.handleLocalSourcesCleared)
      this.clearEditPrefix()
      this.disposed = true
      this.destroyRuntime()
    })
    this.syncToolbarToggles()
    this.scheduleRuntimePreload()
  }

  private queryCell<T extends Element>(selector: string): T | null {
    return this.cellRoot.querySelector<T>(selector)
  }

  private queryCells<T extends Element>(selector: string): NodeListOf<T> {
    return this.cellRoot.querySelectorAll<T>(selector)
  }

  private cellById(cellId: string | undefined): RuntimeCell | undefined {
    if (cellId === undefined) return undefined
    return this.payload.cells.find(cell => cell.id === cellId)
  }

  private cellFrame(cellId: string): HTMLElement | null {
    return this.queryCell<HTMLElement>(`[data-notebook-cell-frame="${CSS.escape(cellId)}"]`)
  }

  private cellFromFrame(frame: HTMLElement | undefined): RuntimeCell | undefined {
    if (!frame || !this.cellRoot.contains(frame)) return undefined
    return this.cellById(notebookCellFrameId(frame))
  }

  private selectCell(cellId: string) {
    const frame = this.cellFrame(cellId)
    if (!frame) return
    selectNotebookCellFrame(frame)
    this.activeCellId = cellId
  }

  private clearCellSelection() {
    if (this.activeCellId === undefined) return
    clearActiveNotebookCellFrames(document)
    this.activeCellId = undefined
  }

  private cellFromTarget(target: EventTarget | null): RuntimeCell | undefined {
    if (!(target instanceof Element)) return undefined
    return this.cellFromFrame(notebookCellFrameFromElement(target))
  }

  private activeCellFromDocument(
    frame = activeNotebookCellFrame(document),
  ): RuntimeCell | undefined {
    if (!frame) return undefined
    const cell = this.cellFromFrame(frame)
    this.activeCellId = cell?.id
    return cell
  }

  private activeCellFromCache(): RuntimeCell | undefined {
    if (this.activeCellId === undefined) return undefined
    const frame = this.cellFrame(this.activeCellId)
    if (!frame || !notebookCellFrameIsActive(frame)) {
      this.activeCellId = undefined
      return undefined
    }
    return this.cellById(this.activeCellId)
  }

  private commandCell(target: EventTarget | null): RuntimeCell | undefined {
    const activeFrame = activeNotebookCellFrame(document)
    const activeCell = this.activeCellFromDocument(activeFrame)
    if (activeFrame) return activeCell
    return this.cellFromTarget(target) ?? this.activeCellFromCache()
  }

  private targetOwnsKeyboard(target: EventTarget | null): boolean {
    if (!(target instanceof Element)) return false
    if (target.closest('.cm-editor')) return true
    if (target.closest('input, textarea, select, button, a[href], [contenteditable]')) return true
    return false
  }

  private siteShortcutLayerActive(): boolean {
    const palette = document.getElementById('palette-container')
    if (palette?.classList.contains('active')) return true
    const shortcuts = document.getElementById('shortcut-container')
    if (shortcuts?.classList.contains('active')) return true
    const search = document.querySelector('.search .search-container.active')
    if (search) return true
    const headings = document.querySelector<HTMLElement>('.headings-modal-container')
    return headings?.style.display === 'flex'
  }

  private claimNotebookKey(event: KeyboardEvent) {
    event.preventDefault()
    event.stopImmediatePropagation()
  }

  private enterEditMode(cell: RuntimeCell) {
    this.clearEditPrefix()
    this.selectCell(cell.id)
    void this.showSourceEditor(cell, true)
  }

  private clearEditPrefix() {
    if (this.editPrefixTimeout !== undefined) window.clearTimeout(this.editPrefixTimeout)
    this.editPrefixTimeout = undefined
  }

  private armEditPrefix() {
    this.clearEditPrefix()
    this.editPrefixTimeout = window.setTimeout(() => {
      this.editPrefixTimeout = undefined
    }, 800)
  }

  private handleNotebookKeydown = (event: KeyboardEvent) => {
    if (
      event.defaultPrevented ||
      this.siteShortcutLayerActive() ||
      this.targetOwnsKeyboard(event.target)
    )
      return
    const cell = this.commandCell(event.target)
    if (!cell) return

    if (notebookRunAndAdvanceKey(event)) {
      this.clearEditPrefix()
      this.selectCell(cell.id)
      this.claimNotebookKey(event)
      void this.runCellAndAdvance(cell)
      return
    }

    if (notebookRunKey(event)) {
      this.clearEditPrefix()
      this.selectCell(cell.id)
      this.claimNotebookKey(event)
      void this.runCell(cell)
      return
    }

    if (event.ctrlKey || event.metaKey || event.altKey) {
      this.clearEditPrefix()
      return
    }

    if (this.editPrefixTimeout !== undefined && event.key !== 'e') this.clearEditPrefix()

    if (event.key === 'e') {
      this.claimNotebookKey(event)
      if (this.editPrefixTimeout !== undefined) {
        this.enterEditMode(cell)
        return
      }
      this.armEditPrefix()
      return
    }

    if (event.key === 'Enter' || event.key === 'i') {
      this.claimNotebookKey(event)
      this.enterEditMode(cell)
      return
    }

    if (event.key !== 'Escape') return
    const controls = this.sourceControls.get(cell.id)
    if (controls && !controls.editorHost.hidden) {
      this.claimNotebookKey(event)
      void this.closeSourceEditor(cell)
      return
    }
    if (this.activeCellId !== undefined) {
      this.claimNotebookKey(event)
      this.clearCellSelection()
    }
  }

  private ensureToolbar() {
    const toolbar =
      this.root.querySelector<HTMLElement>('[data-notebook-runtime-toolbar]') ??
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
    if (!toolbar.isConnected) this.root.append(toolbar)
  }

  private readStoredVimMode(): boolean {
    return readStoredNotebookVimMode()
  }

  private writeStoredVimMode(enabled: boolean) {
    try {
      window.localStorage.setItem(notebookRuntimeVimModeKey, enabled ? 'true' : 'false')
    } catch {}
  }

  private syncToolbarToggles() {
    const debug = this.root.querySelector<HTMLButtonElement>('[data-notebook-debug]')
    if (debug) {
      setNotebookToggleButtonState(
        debug,
        this.debug,
        this.debug ? 'Disable debug output' : 'Enable debug output',
      )
    }
    const vim = this.root.querySelector<HTMLButtonElement>('[data-notebook-vim-mode]')
    if (vim) setNotebookVimButtonState(vim, this.vimMode)
    for (const button of this.cellRoot.querySelectorAll<HTMLButtonElement>(
      '[data-notebook-vim-cell]',
    )) {
      setNotebookVimButtonState(button, this.vimMode)
    }
  }

  private toggleDebug = () => {
    this.debug = !this.debug
    this.syncToolbarToggles()
  }

  private toggleVimMode = () => {
    this.vimMode = !this.vimMode
    this.writeStoredVimMode(this.vimMode)
    this.syncToolbarToggles()
    for (const controls of this.sourceControls.values()) {
      void controls.editor?.setVimMode(this.vimMode).catch(error => {
        this.setStatus(error instanceof Error ? error.message : 'failed to toggle vim mode')
      })
    }
  }

  private addCleanup(callback: () => void) {
    const runtimeWindow = window as Window & { addCleanup?: (callback: () => void) => void }
    if (typeof runtimeWindow.addCleanup === 'function') {
      runtimeWindow.addCleanup(callback)
    } else {
      window.addEventListener('beforeunload', callback, { once: true })
    }
  }

  private handleKernelRequest = (event: Event) => {
    const detail = (event as CustomEvent<NotebookKernelRequestDetail>).detail
    if (!detail || typeof detail.respond !== 'function') return
    for (const snapshot of this.kernelSnapshots()) detail.respond(snapshot)
  }

  private handleKernelCommand = (event: Event) => {
    const detail = (event as CustomEvent<NotebookKernelCommandDetail>).detail
    if (!detail || detail.runtimeId !== this.payload.id) return
    void this.applyKernelCommand(detail).catch(error => {
      this.setStatus(error instanceof Error ? error.message : 'kernel command failed')
    })
  }

  private handleKernelRunAll = () => {
    void this.runAll()
  }

  private handleLocalSourcesCleared = () => {
    void this.clearLocalSourcesFromRuntime()
  }

  private async clearLocalSourcesFromRuntime() {
    for (const cell of this.payload.cells) {
      const controls = this.sourceControls.get(cell.id)
      this.clearStoredSource(cell)
      if (!controls) continue
      const preserveEditorSource = !controls.editorHost.hidden && controls.dirty
      if (preserveEditorSource) {
        this.syncSourceControls(cell)
        continue
      }
      controls.source = cell.source
      controls.dirty = false
      controls.editor?.setValue(cell.source)
      await this.replaceRenderedCellSource(cell, controls, cell.source)
      this.syncSourceControls(cell)
    }
  }

  private kernelSnapshots(): NotebookKernelSnapshot[] {
    return notebookRuntimeKernelLanguages(this.payload).map(language => {
      const backend = backendFor(language)
      const backendName = backend?.name ?? language
      const presence = this.kernelPresence(backendName)
      const snapshot: NotebookKernelSnapshot = {
        runtimeId: this.payload.id,
        sourcePath: this.payload.sourcePath,
        language: backendName,
        status: presence.status,
      }
      if (snapshot.status === 'running' && this.runningCellId) {
        return { ...snapshot, runningCellId: this.runningCellId, statusDetail: this.runningCellId }
      }
      if (presence.statusDetail !== undefined) {
        return { ...snapshot, statusDetail: presence.statusDetail }
      }
      return snapshot
    })
  }

  private kernelPresence(backendName: string): KernelPresence {
    const kernel = this.kernels.get(backendName)
    const presence = this.kernelPresences.get(backendName)
    if (kernel && this.running && this.runningKernel === kernel) {
      if (presence?.status === 'interrupting') return presence
      return { status: 'running', statusDetail: this.runningCellId }
    }
    if (this.kernelRequests.has(backendName)) return { status: 'warming' }
    if (presence) return presence
    return { status: kernel ? 'ready' : 'available' }
  }

  private setKernelPresence(
    backendName: string,
    status: NotebookKernelStatus,
    statusDetail?: string,
  ) {
    this.kernelPresences.set(
      backendName,
      statusDetail === undefined ? { status } : { status, statusDetail },
    )
  }

  private async applyKernelCommand(detail: NotebookKernelCommandDetail): Promise<void> {
    const backend = backendFor(detail.language)
    if (!backend) return
    if (detail.command === 'interrupt') {
      this.interruptBackendKernel(backend)
      return
    }
    this.disposeBackendKernel(backend)
    if (detail.command === 'kill') {
      this.setKernelPresence(backend.name, 'killed')
      this.setStatus(`killed ${backend.name}`)
      return
    }
    this.setKernelPresence(backend.name, 'warming')
    this.setStatus(`restarting ${backend.name}`)
    try {
      await this.ensureBackendKernel(backend)
      this.setKernelPresence(backend.name, 'ready')
      if (!this.running) this.setStatus(`${backend.name} ready`)
    } catch (error) {
      this.setKernelPresence(backend.name, 'failed')
      this.setStatus(error instanceof Error ? error.message : `${backend.name} restart failed`)
    }
  }

  private interruptBackendKernel(backend: ExecutableLanguageBackend) {
    const kernel = this.kernels.get(backend.name)
    if (kernel && this.running && this.runningKernel === kernel) {
      this.stopped = true
      this.setKernelPresence(
        backend.name,
        'interrupting',
        this.runningCellId === undefined ? undefined : this.runningCellId,
      )
      kernel.interrupt()
      this.setStatus(`interrupting ${backend.name}`)
      return
    }
    this.setStatus(`${backend.name} kernel is idle`)
  }

  private runAll = async () => {
    if (this.running) return
    this.stopped = false
    for (const cell of this.payload.cells) {
      if (this.stopped) break
      const succeeded = await this.runCell(cell)
      if (!succeeded) {
        if (!this.stopped) {
          this.stopped = true
          this.setStatus(`stopped after ${cell.id}`)
        }
        break
      }
    }
  }

  private nextCell(cell: RuntimeCell): RuntimeCell | undefined {
    return this.cellById(nextNotebookCellId(this.payload.cells, cell.id))
  }

  private focusCell(cell: RuntimeCell) {
    this.selectCell(cell.id)
    this.cellFrame(cell.id)?.focus()
  }

  private async runCellAndAdvance(cell: RuntimeCell) {
    if (this.running) return
    if (!(await this.saveEditorSource(cell, false))) return
    if (this.running) return
    const running = this.runCell(cell)
    this.focusCell(this.nextCell(cell) ?? cell)
    void running
  }

  private runCell = async (cell: RuntimeCell): Promise<boolean> => {
    if (this.running) return false
    this.stopped = false
    const source = this.sourceForCell(cell)
    const executionCount = this.nextExecutionCount(cell.id)
    this.running = true
    this.runningCellId = cell.id
    this.setStatus(`running ${cell.id}`)
    this.setExecutionLabel(cell.id, '*')
    this.setRunningControls(true)
    this.clearOutput(cell.id)
    try {
      const backend = backendFor(cell.language)
      const check = backend?.canExecute(source) ?? {
        ok: false as const,
        reason: `${cell.language} notebook cells are not executable in this runtime.`,
      }
      if (!check.ok) {
        if (backend) this.setKernelPresence(backend.name, 'failed', cell.id)
        this.renderOutput(cell.id, {
          type: 'error',
          ename: 'UnsupportedRuntimeFeature',
          evalue: check.reason,
          traceback: check.reason,
        })
        return false
      }
      const result = await this.runKernelCell(cell, source)
      if (!result.failed) this.renderSuccessOutput(cell.id)
      return !result.failed
    } catch (error) {
      if (this.stopped) return false
      const text = error instanceof Error ? error.message : String(error)
      const output: RuntimeErrorOutput = {
        type: 'error',
        ename: 'RuntimeError',
        evalue: text,
        traceback: text,
      }
      if (this.debug) output.debug = this.debugOutput('client-runtime', cell.id, error)
      this.renderOutput(cell.id, output)
      return false
    } finally {
      this.running = false
      this.runningCellId = undefined
      this.setExecutionLabel(cell.id, executionCount)
      this.setRunningControls(false)
      this.setStatus(this.stopped ? 'stopped' : 'idle')
    }
  }

  private stop = () => {
    if (this.running && this.runningKernel) {
      this.stopped = true
      if (this.runningBackendName) {
        this.setKernelPresence(
          this.runningBackendName,
          'interrupting',
          this.runningCellId === undefined ? undefined : this.runningCellId,
        )
      }
      this.runningKernel.interrupt()
      this.setStatus('interrupting')
      return
    }
    this.stopped = true
    this.destroyRuntime()
    this.running = false
    this.runningCellId = undefined
    for (const language of notebookRuntimeKernelLanguages(this.payload)) {
      const backend = backendFor(language)
      this.setKernelPresence(backend?.name ?? language, 'stopped')
    }
    this.setRunningControls(false)
    this.setStatus('stopped')
  }

  private reset = () => {
    this.stopped = false
    this.destroyRuntime()
    this.kernelPresences.clear()
    this.running = false
    this.runningCellId = undefined
    this.payload.cells.forEach(cell => {
      this.clearOutput(cell.id)
      this.restoreSavedOutput(cell.id)
      this.setExecutionLabel(cell.id, cell.executionIndex)
    })
    this.executionCounter = Math.max(0, ...this.payload.cells.map(cell => cell.executionIndex ?? 0))
    this.setRunningControls(false)
    this.setStatus('idle')
    this.scheduleRuntimePreload()
  }

  private decorateCells() {
    for (const cell of this.payload.cells) {
      const controls = this.queryCell<HTMLElement>(`[data-notebook-cell="${CSS.escape(cell.id)}"]`)
      if (!controls) continue
      this.ensureCellControls(cell, controls)
      this.setExecutionLabel(cell.id, cell.executionIndex)
      const existingFrame = controls.closest<HTMLElement>('[data-notebook-cell-frame]')
      if (existingFrame) {
        syncStaticOutputTabs(existingFrame, cell.id)
        this.decorateSourceEditor(cell, existingFrame)
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
      this.decorateSourceEditor(cell, frame)
    }
  }

  private ensureCellControls(cell: RuntimeCell, controls: HTMLElement) {
    if (
      !controls.querySelector<HTMLElement>(
        `[data-notebook-execution-label="${CSS.escape(cell.id)}"]`,
      )
    ) {
      const label = document.createElement('span')
      label.className = 'notebook-execution-prompt'
      label.dataset.notebookExecutionLabel = cell.id
      label.setAttribute('aria-live', 'polite')
      controls.append(label)
    }
  }

  private decorateSourceEditor(cell: RuntimeCell, frame: HTMLElement) {
    if (this.sourceControls.has(cell.id)) return
    if (!frame.hasAttribute('tabindex')) frame.tabIndex = -1
    const selectSource = (event: Event) => {
      this.selectCell(cell.id)
      if (
        (event.type === 'pointerdown' || event.type === 'click') &&
        !this.targetOwnsKeyboard(event.target)
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
      if (this.running && this.runningCellId === cell.id) {
        this.stop()
        return
      }
      void this.runCell(cell)
    }
    runButton.addEventListener('click', runSource)

    const vimButton =
      frame.querySelector<HTMLButtonElement>(`[data-notebook-vim-cell="${CSS.escape(cell.id)}"]`) ??
      document.createElement('button')
    vimButton.type = 'button'
    vimButton.dataset.notebookVimCell = cell.id
    setNotebookIconButton(vimButton, 'vim', 'Enable Vim mode')
    const toggleCellVimMode = () => {
      this.toggleVimMode()
    }
    vimButton.addEventListener('click', toggleCellVimMode)

    const editorHost =
      frame.querySelector<HTMLElement>(`[data-notebook-source-editor="${CSS.escape(cell.id)}"]`) ??
      document.createElement('div')
    editorHost.className = 'notebook-source-editor'
    editorHost.dataset.notebookSourceEditor = cell.id
    editorHost.hidden = true

    const editButton =
      frame.querySelector<HTMLButtonElement>(
        `[data-notebook-edit-cell="${CSS.escape(cell.id)}"]`,
      ) ?? document.createElement('button')
    editButton.type = 'button'
    editButton.dataset.notebookEditCell = cell.id
    setNotebookIconButton(editButton, 'edit', `Edit ${cell.id}`)
    const editSource = () => {
      void this.showSourceEditor(cell, true)
    }
    editButton.addEventListener('click', editSource)

    const saveButton =
      frame.querySelector<HTMLButtonElement>(
        `[data-notebook-save-cell="${CSS.escape(cell.id)}"]`,
      ) ?? document.createElement('button')
    saveButton.type = 'button'
    saveButton.dataset.notebookSaveCell = cell.id
    setNotebookIconButton(saveButton, 'save', `Save ${cell.id} locally`)
    saveButton.hidden = true
    const saveSource = () => {
      void this.saveEditorSource(cell)
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
      void this.revertEditorSource(cell)
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
    const stored = this.readStoredSource(cell)
    const renderedSource = figure ? renderedFigureSource(figure) : undefined
    const baseSource =
      renderedSource !== undefined && renderedSource.trimEnd() !== cell.source.trimEnd()
        ? renderedSource
        : cell.source
    cell.source = baseSource
    this.sourceControls.set(cell.id, {
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
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return

    if (stored !== undefined && stored === cell.source) {
      this.clearStoredSource(cell)
    } else if (stored !== undefined && figure) {
      void this.replaceRenderedCellSource(cell, controls, stored)
    }
    this.syncSourceControls(cell)

    this.addCleanup(() => {
      runButton.removeEventListener('click', runSource)
      vimButton.removeEventListener('click', toggleCellVimMode)
      editButton.removeEventListener('click', editSource)
      saveButton.removeEventListener('click', saveSource)
      revertButton.removeEventListener('click', revertSource)
      frame.removeEventListener('pointerdown', selectSource, true)
      frame.removeEventListener('click', selectSource)
      frame.removeEventListener('focusin', selectSource)
      this.sourceControls.get(cell.id)?.editor?.destroy()
    })
  }

  private sourceForCell(cell: RuntimeCell): string {
    const controls = this.sourceControls.get(cell.id)
    if (controls && !controls.editorHost.hidden && controls.editor)
      return controls.editor.getValue()
    if (controls) return controls.source
    return this.readStoredSource(cell) ?? cell.source
  }

  private sourceSnapshotForCell(cell: RuntimeCell): string {
    const controls = this.sourceControls.get(cell.id)
    if (controls) return controls.source
    return this.readStoredSource(cell) ?? cell.source
  }

  private sourceStorageKey(cell: RuntimeCell): string {
    return notebookRuntimeLocalSourceKey(this.payload.sourcePath, cell.id)
  }

  private readStoredSource(cell: RuntimeCell): string | undefined {
    try {
      const value = window.localStorage.getItem(this.sourceStorageKey(cell))
      return value === null ? undefined : value
    } catch {
      return undefined
    }
  }

  private writeStoredSource(cell: RuntimeCell, source: string): boolean {
    try {
      window.localStorage.setItem(this.sourceStorageKey(cell), source)
      const controls = this.sourceControls.get(cell.id)
      if (controls) controls.storedSource = source
      return true
    } catch {
      return false
    }
  }

  private clearStoredSource(cell: RuntimeCell) {
    try {
      window.localStorage.removeItem(this.sourceStorageKey(cell))
    } catch {}
    const controls = this.sourceControls.get(cell.id)
    if (controls) controls.storedSource = undefined
  }

  private async ensureSourceEditor(cell: RuntimeCell, controls: SourceControls) {
    if (controls.editor) return controls.editor
    const { createNotebookCodeEditor } = await import('../editor/code-editor')
    const editorLanguage = runtimeCellEditorLanguage(cell)
    controls.editor = await createNotebookCodeEditor({
      parent: controls.editorHost,
      initialContent: controls.source,
      language: editorLanguage,
      vimMode: this.vimMode,
      lsp: {
        enabled: backendFor(editorLanguage)?.editor?.lspBridge !== undefined,
        runtimeId: this.payload.id,
        sourcePath: this.payload.sourcePath,
        cellId: cell.id,
        language: editorLanguage,
        cells: () =>
          this.payload.cells.map(runtimeCell => ({
            id: runtimeCell.id,
            source: this.sourceSnapshotForCell(runtimeCell),
            language: runtimeCellEditorLanguage(runtimeCell),
            executionIndex: runtimeCell.executionIndex,
          })),
      },
      onEdited: () => {
        if (controls.dirty) return
        controls.dirty = true
        this.syncSourceControls(cell)
      },
      onSubmit: () => this.runCell(cell),
      onSubmitAndAdvance: () => this.runCellAndAdvance(cell),
      onSave: () => {
        void this.saveEditorSource(cell)
      },
      onCancel: () => {
        void this.closeSourceEditor(cell)
      },
    })
    return controls.editor
  }

  private async showSourceEditor(cell: RuntimeCell, visible: boolean, restoreFocus = true) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    if (visible) {
      const editor = await this.ensureSourceEditor(cell, controls)
      const source = this.sourceForCell(cell)
      editor.setValue(source)
      controls.source = source
      controls.dirty = false
    }
    controls.editorHost.hidden = !visible
    if (controls.figure) controls.figure.hidden = visible
    controls.frame.toggleAttribute('data-notebook-editing', visible)
    this.syncSourceControls(cell)
    if (visible) {
      controls.editor?.focus()
    } else if (restoreFocus) {
      this.selectCell(cell.id)
      controls.frame.focus({ preventScroll: true })
    }
  }

  private async highlightedSourceLines(
    cell: RuntimeCell,
    controls: SourceControls,
    source: string,
  ): Promise<HTMLElement[] | undefined> {
    const lineCount = source.split(/\r?\n/).length
    if (!controls.figure) return undefined
    try {
      const { renderNotebookHighlightedLines } = await import('../editor/code-editor')
      const highlightedLines = await renderNotebookHighlightedLines(
        source,
        runtimeCellEditorLanguage(cell),
      )
      return highlightedLines.length === lineCount ? highlightedLines : undefined
    } catch {
      return undefined
    }
  }

  private async replaceRenderedCellSource(
    cell: RuntimeCell,
    controls: SourceControls,
    source: string,
  ) {
    if (!controls.figure || controls.renderedSource === source) return
    const highlightedLines = await this.highlightedSourceLines(cell, controls, source)
    replaceRenderedSource(controls.figure, source, highlightedLines)
    controls.renderedSource = source
  }

  private async closeSourceEditor(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    if (controls.editor) {
      controls.source = controls.editor.getValue()
    }
    await this.replaceRenderedCellSource(cell, controls, controls.source)
    void this.showSourceEditor(cell, false)
  }

  private async saveEditorSource(cell: RuntimeCell, restoreFocus = true): Promise<boolean> {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return true
    const source = this.sourceForCell(cell)
    if (source === cell.source) {
      controls.source = cell.source
      controls.dirty = false
      this.clearStoredSource(cell)
      await this.replaceRenderedCellSource(cell, controls, cell.source)
      await this.showSourceEditor(cell, false, restoreFocus)
      this.setStatus('idle')
      return true
    }
    if (!this.writeStoredSource(cell, source)) {
      this.setStatus('local save failed')
      return false
    }
    controls.source = source
    controls.dirty = false
    await this.replaceRenderedCellSource(cell, controls, source)
    await this.showSourceEditor(cell, false, restoreFocus)
    this.setStatus('saved local edit')
    return true
  }

  private async revertEditorSource(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    controls.source = cell.source
    controls.dirty = false
    controls.editor?.setValue(cell.source)
    this.clearStoredSource(cell)
    await this.replaceRenderedCellSource(cell, controls, cell.source)
    void this.showSourceEditor(cell, false)
    this.setStatus('idle')
  }

  private syncSourceControls(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    const stored = controls.storedSource
    const editing = !controls.editorHost.hidden
    const edited = editing && controls.dirty
    const hasStoredSource = stored !== undefined
    controls.status.hidden = !hasStoredSource && !edited
    controls.status.dataset.notebookSourceState = edited ? 'edited' : 'local'
    controls.status.title = edited ? 'edited local source' : 'saved local source'
    controls.status.setAttribute(
      'aria-label',
      edited ? 'edited local source' : 'saved local source',
    )
    controls.editButton.hidden = editing
    controls.vimButton.hidden = !editing
    controls.saveButton.hidden = !editing
    controls.revertButton.hidden = !(editing || hasStoredSource)
    controls.frame.toggleAttribute('data-notebook-local-source', hasStoredSource)
  }

  private nextExecutionCount(cellId: string): number {
    this.executionCounter += 1
    const controls = this.queryCell<HTMLElement>(`[data-notebook-cell="${CSS.escape(cellId)}"]`)
    if (controls) controls.dataset.notebookExecutionCount = String(this.executionCounter)
    return this.executionCounter
  }

  private setExecutionLabel(cellId: string, count: number | '*' | null) {
    const label = this.queryCell<HTMLElement>(
      `[data-notebook-execution-label="${CSS.escape(cellId)}"]`,
    )
    if (!label) return
    label.textContent = count === null ? 'In [ ]:' : `In [${count}]:`
  }

  private async ensureKernel(
    cell: RuntimeCell,
  ): Promise<{ kernel: Kernel; backend: ExecutableLanguageBackend }> {
    const backend = backendFor(cell.language)
    if (!backend) {
      throw new Error(`${cell.language} notebook cells are not executable here.`)
    }
    return { kernel: await this.ensureBackendKernel(backend), backend }
  }

  private async ensureBackendKernel(backend: ExecutableLanguageBackend): Promise<Kernel> {
    const existing = this.kernels.get(backend.name)
    if (existing) return existing
    const pending = this.kernelRequests.get(backend.name)
    if (pending) return await pending
    const epoch = this.runtimeEpoch
    const kernelEpoch = this.kernelEpoch(backend.name)
    const request = this.createBackendKernel(backend, epoch, kernelEpoch)
    this.kernelRequests.set(backend.name, request)
    try {
      return await request
    } finally {
      if (this.kernelRequests.get(backend.name) === request)
        this.kernelRequests.delete(backend.name)
    }
  }

  private async createBackendKernel(
    backend: ExecutableLanguageBackend,
    epoch: number,
    kernelEpoch: number,
  ): Promise<Kernel> {
    const kernel = await backend.kernelFactory({
      runtimeId: this.payload.id,
      sourcePath: this.payload.sourcePath,
      indexUrl: this.payload.indexUrl || backend.defaultIndexUrl,
      workerUrl: backend.workerAssetKey
        ? this.assets[backend.workerAssetKey]
        : this.assets.workerUrl,
      resolveAsset: (request, runtimeFile) => this.resolveAsset(request, runtimeFile),
      status: text => {
        if (
          this.disposed ||
          this.runtimeEpoch !== epoch ||
          this.kernelEpoch(backend.name) !== kernelEpoch
        ) {
          return
        }
        this.setKernelPresence(backend.name, 'warming', text)
        this.setStatus(text)
      },
    })
    try {
      await kernel.init({
        signal: new AbortController().signal,
        indexUrl: this.payload.indexUrl || backend.defaultIndexUrl,
      })
    } catch (error) {
      await kernel.dispose()
      throw error
    }
    if (
      this.disposed ||
      this.runtimeEpoch !== epoch ||
      this.kernelEpoch(backend.name) !== kernelEpoch
    ) {
      await kernel.dispose()
      throw new Error('notebook runtime was disposed')
    }
    this.kernels.set(backend.name, kernel)
    this.setKernelPresence(backend.name, 'ready')
    return kernel
  }

  private kernelEpoch(backendName: string): number {
    return this.kernelEpochs.get(backendName) ?? 0
  }

  private invalidateBackendKernel(backendName: string) {
    this.kernelEpochs.set(backendName, this.kernelEpoch(backendName) + 1)
  }

  private disposeBackendKernel(backend: ExecutableLanguageBackend) {
    this.invalidateBackendKernel(backend.name)
    this.kernelRequests.delete(backend.name)
    const kernel = this.kernels.get(backend.name)
    this.kernels.delete(backend.name)
    if (kernel && this.running && this.runningKernel === kernel) {
      this.stopped = true
      this.setKernelPresence(
        backend.name,
        'interrupting',
        this.runningCellId === undefined ? undefined : this.runningCellId,
      )
      this.setStatus(`killing ${backend.name}`)
    }
    void kernel?.dispose()
  }

  private async runKernelCell(cell: RuntimeCell, source: string): Promise<CellRunResult> {
    const { kernel, backend } = await this.ensureKernel(cell)
    const modules = await this.notebookModulesFor(source, backend)
    let failed = false
    let threw = false
    let interrupted = false
    this.runningKernel = kernel
    this.runningBackendName = backend.name
    this.setKernelPresence(backend.name, 'running', cell.id)
    try {
      for await (const event of kernel.execute(cell.id, source, { debug: this.debug, modules })) {
        if (event.type === 'interrupted') interrupted = true
        failed = this.handleRuntimeEvent(event) || failed
      }
    } catch (error) {
      threw = true
      this.setKernelPresence(backend.name, 'failed', cell.id)
      throw error
    } finally {
      if (this.runningKernel === kernel) this.runningKernel = undefined
      if (this.runningBackendName === backend.name) this.runningBackendName = undefined
      const presence = this.kernelPresences.get(backend.name)
      if (!threw && presence?.status !== 'killed') {
        this.setKernelPresence(
          backend.name,
          interrupted || this.stopped ? 'interrupted' : failed ? 'failed' : 'ready',
          failed || interrupted || this.stopped ? cell.id : undefined,
        )
      }
    }
    return { failed }
  }

  private scheduleRuntimePreload() {
    if (this.runtimePreload) return
    if (!supportsEagerRuntimePreload()) return
    scheduleNotebookRuntimePreload(() => {
      if (this.disposed) return
      if (this.running) {
        this.runtimePreload = undefined
        this.scheduleRuntimePreload()
        return
      }
      this.runtimePreload = this.preloadRuntimeLanguages()
    })
  }

  private async preloadRuntimeLanguages(): Promise<void> {
    for (const language of notebookRuntimePreloadLanguages(this.payload)) {
      if (this.disposed || this.running) {
        this.runtimePreload = undefined
        this.scheduleRuntimePreload()
        return
      }
      const backend = backendFor(language)
      if (!backend) continue
      try {
        await this.ensureBackendKernel(backend)
      } catch (error) {
        if (this.disposed || notebookRuntimeWasDisposed(error)) return
        console.warn(`failed to preload ${backend.name} notebook runtime`, error)
      }
    }
  }

  private handleRuntimeEvent(event: RuntimeEvent): boolean {
    if (event.type === 'started') return false
    if (event.type === 'stream') {
      this.renderOutput(event.cellId, { type: 'stream', name: event.name, text: event.text })
      return false
    }
    if (event.type === 'output') {
      if (
        'type' in event.output &&
        event.output.type === 'error' &&
        event.output.ename === 'UnsupportedRuntimeFeature'
      ) {
        this.restoreSavedOutput(event.cellId)
      }
      this.renderKernelOutput(event.cellId, event.output)
      return false
    }
    if (event.type === 'error') {
      this.renderKernelOutput(event.cellId, event.output)
      return true
    }
    if (event.type === 'download') {
      this.downloadRuntimeFile(event.download)
      return false
    }
    if (event.type === 'status') {
      this.setStatus(event.text)
      return false
    }
    if (event.type === 'interrupted') {
      return true
    }
    if (event.type === 'done') {
      return event.failed === true
    }
    return false
  }

  private debugOutput(phase: string, cellId: string, error: unknown): RuntimeDebugOutput {
    const debug: RuntimeDebugOutput = {
      phase,
      cellId,
      errorMessage: error instanceof Error ? error.message : String(error),
    }
    if (error instanceof Error) {
      debug.errorName = error.name
      if (error.stack !== undefined) debug.stack = error.stack
    }
    return debug
  }

  private async notebookModulesFor(
    source: string,
    backend: ExecutableLanguageBackend,
  ): Promise<NotebookModule[]> {
    const resolver = backend.moduleResolver
    if (!resolver) return []
    const seen = new Set<string>()
    const modules = new Map<string, NotebookModule>()
    await this.collectNotebookModules(source, resolver, seen, modules)
    return [...modules.values()]
  }

  private async collectNotebookModules(
    source: string,
    resolver: RuntimeModuleResolver,
    seen: Set<string>,
    modules: Map<string, NotebookModule>,
  ): Promise<void> {
    for (const name of resolver.importNames(source)) {
      if (seen.has(name)) continue
      seen.add(name)
      const module = await this.fetchNotebookModule(name, resolver)
      if (!module) continue
      modules.set(name, module)
      await this.collectNotebookModules(module.source, resolver, seen, modules)
    }
  }

  private async fetchNotebookModule(
    name: string,
    resolver: RuntimeModuleResolver,
  ): Promise<NotebookModule | null> {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) return null
    const cached = this.moduleCache.get(name)
    if (cached !== undefined) return cached
    const pending = this.moduleFetches.get(name)
    if (pending) return await pending
    const request = this.loadNotebookModule(name, resolver)
    this.moduleFetches.set(name, request)
    try {
      const module = await request
      this.moduleCache.set(name, module)
      return module
    } finally {
      this.moduleFetches.delete(name)
    }
  }

  private async loadNotebookModule(
    name: string,
    resolver: RuntimeModuleResolver,
  ): Promise<NotebookModule | null> {
    if (this.importableModules !== undefined && !this.importableModules.has(name)) return null
    const url = new URL(`${name}.ipynb`, window.location.href)
    const baseDir = new URL('.', window.location.href)
    if (url.origin !== window.location.origin || !url.pathname.startsWith(baseDir.pathname)) {
      throw new Error(`blocked non-sibling notebook import: ${name}`)
    }
    const relative = url.pathname.slice(baseDir.pathname.length)
    if (relative.includes('/')) {
      throw new Error(`blocked nested notebook import: ${name}`)
    }
    const response = await fetch(url)
    if (response.status === 404) return null
    if (!response.ok) {
      throw new Error(`failed to fetch notebook import ${name}.ipynb: ${response.status}`)
    }
    const source = resolver.moduleSource(await response.text(), `${name}.ipynb`)
    return { name, sourcePath: url.pathname, source }
  }

  private downloadRuntimeFile(message: RuntimeDownload) {
    const name = downloadNotebookFile(message.filename, message.contentType, message.bytes)
    this.setStatus(`downloaded ${name}`)
    showNotebookOutputToast(`downloaded ${name}`)
  }

  private async resolveAsset(
    message: RuntimeAssetRequest,
    runtimeFile: (path: string) => Promise<RuntimeFileResult | undefined>,
  ): Promise<RuntimeAssetResult> {
    const fallback = (error: unknown): RuntimeAssetResult => ({
      runtimeId: message.runtimeId,
      assetId: message.assetId,
      ok: false,
      status: 404,
      statusText: 'Not Found',
      contentType: 'text/plain',
      error: error instanceof Error ? error.message : String(error),
    })

    let relative = ''
    let assetUrl = ''
    try {
      const base = new URL(window.location.href)
      const baseDir = new URL('.', base)
      const url = new URL(message.url, base)
      assetUrl = url.href
      relative = url.pathname.slice(baseDir.pathname.length)
      if (
        url.origin !== window.location.origin ||
        !url.pathname.startsWith(baseDir.pathname) ||
        relative.includes('/')
      ) {
        throw new Error(`blocked non-sibling notebook asset: ${message.url}`)
      }
    } catch (error) {
      return fallback(error)
    }

    try {
      const response = await fetch(assetUrl)
      if (!response.ok) {
        throw new Error(`missing notebook asset: ${relative || message.url}`)
      }
      return {
        runtimeId: message.runtimeId,
        assetId: message.assetId,
        ok: true,
        status: response.status,
        statusText: response.statusText,
        contentType: response.headers.get('content-type') ?? 'application/octet-stream',
        bytes: await response.arrayBuffer(),
      }
    } catch (error) {
      return (
        (await this.resolveRuntimeFileAsset(message, relative || message.url, runtimeFile)) ??
        fallback(error)
      )
    }
  }

  private async resolveRuntimeFileAsset(
    message: RuntimeAssetRequest,
    path: string,
    runtimeFile: (path: string) => Promise<RuntimeFileResult | undefined>,
  ): Promise<RuntimeAssetResult | undefined> {
    const result = await runtimeFile(path)
    if (!result) return undefined
    return {
      runtimeId: message.runtimeId,
      assetId: message.assetId,
      ok: result.ok,
      status: result.status,
      statusText: result.statusText,
      contentType: result.contentType,
      bytes: result.bytes,
      error: result.error,
    }
  }

  private renderOutput(cellId: string, output: NotebookRuntimeOutput) {
    const target = this.queryCell<HTMLElement>(`[data-notebook-output="${CSS.escape(cellId)}"]`)
    if (!target) return
    this.hideSavedOutput(cellId)
    target.hidden = false
    appendRuntimeOutput(target, output, { debug: this.debug })
  }

  private renderKernelOutput(cellId: string, output: KernelOutput) {
    if ('kind' in output) {
      const target = this.queryCell<HTMLElement>(`[data-notebook-output="${CSS.escape(cellId)}"]`)
      if (!target) return
      this.hideSavedOutput(cellId)
      target.hidden = false
      target.insertAdjacentHTML('beforeend', renderOutputHtml(output))
      syncOutputTabs(target)
      return
    }
    this.renderOutput(cellId, output)
  }

  private renderSuccessOutput(cellId: string) {
    const target = this.queryCell<HTMLElement>(`[data-notebook-output="${CSS.escape(cellId)}"]`)
    if (!target || hasRenderedOutput(target)) return
    this.renderOutput(cellId, { type: 'success' })
  }

  private clearOutput(cellId: string) {
    const target = this.queryCell<HTMLElement>(`[data-notebook-output="${CSS.escape(cellId)}"]`)
    if (!target) return
    this.hideSavedOutput(cellId)
    target.replaceChildren()
    target.hidden = true
    target.removeAttribute('data-notebook-output-tabbed')
  }

  private hideSavedOutput(cellId: string) {
    const existing = this.savedOutputs.get(cellId)
    if (existing) {
      existing.forEach(element => {
        element.hidden = true
      })
      return
    }
    const target = this.queryCell<HTMLElement>(`[data-notebook-output="${CSS.escape(cellId)}"]`)
    if (!target) return
    const saved: HTMLElement[] = []
    let sibling = target.previousElementSibling
    while (sibling instanceof HTMLElement) {
      if (sibling.matches('figure, [data-notebook-cell]')) break
      if (
        sibling.classList.contains('notebook-output') ||
        sibling.hasAttribute('data-notebook-static-output')
      ) {
        saved.unshift(sibling)
      }
      sibling = sibling.previousElementSibling
    }
    if (saved.length === 0) return
    this.savedOutputs.set(cellId, saved)
    saved.forEach(element => {
      element.hidden = true
    })
  }

  private restoreSavedOutput(cellId: string) {
    const saved = this.savedOutputs.get(cellId)
    if (!saved) return
    saved.forEach(element => {
      element.hidden = false
    })
    this.savedOutputs.delete(cellId)
  }

  private setStatus(value: string) {
    const status = this.root.querySelector<HTMLElement>('[data-notebook-status]')
    if (status) status.textContent = value
  }

  private setRunningControls(running: boolean) {
    this.root
      .querySelector<HTMLButtonElement>('[data-notebook-stop]')
      ?.toggleAttribute('disabled', !running)
    this.root
      .querySelector<HTMLButtonElement>('[data-notebook-run-all]')
      ?.toggleAttribute('disabled', running)
    this.queryCells<HTMLButtonElement>('[data-notebook-run-cell]').forEach(button => {
      const cellId = button.dataset.notebookRunCell
      const active = running && cellId === this.runningCellId
      button.toggleAttribute('disabled', running && !active)
      if (running && active && cellId) {
        setNotebookIconButton(button, 'stop', `Interrupt ${cellId}`)
      } else if (cellId) {
        setNotebookIconButton(button, 'run', `Run ${cellId}`)
      }
    })
  }

  private destroyRuntime() {
    this.runtimeEpoch += 1
    this.runtimePreload = undefined
    const kernels = [...this.kernels.values()]
    this.kernels.clear()
    this.kernelRequests.clear()
    this.runningKernel = undefined
    this.runningBackendName = undefined
    for (const kernel of kernels) void kernel.dispose()
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
  new NotebookRuntime(root, payload, assets).mount()
}
