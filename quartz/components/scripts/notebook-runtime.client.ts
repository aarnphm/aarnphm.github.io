import DOMPurify from 'dompurify'
import type { NotebookRuntimeDebugOutput, NotebookRuntimeOutput } from '../../util/notebook-runtime'
import type { NotebookCodeEditor } from './notebook-code-editor'
import {
  notebookSuccessOutputLabel,
  notebookRuntimeImportCandidates,
  notebookRuntimeLocalSourceKey,
  notebookRuntimeModuleSource,
  unsupportedNotebookRuntimeReason,
} from '../../util/notebook-runtime'
import { isRecord, readNumber, readString } from '../../util/type-guards'

type RuntimeCell = { id: string; source: string; language: string; executionIndex: number | null }

type RuntimeModule = { name: string; sourcePath: string; source: string }

type RuntimeErrorOutput = Extract<NotebookRuntimeOutput, { type: 'error' }>

type RuntimeStreamOutput = Extract<NotebookRuntimeOutput, { type: 'stream' }>

type RuntimeDebugOutput = NonNullable<RuntimeErrorOutput['debug']>

type CellRunResult = { failed: boolean }

type SourceControls = {
  frame: HTMLElement
  editor: NotebookCodeEditor | undefined
  editorHost: HTMLElement
  figure: HTMLElement | undefined
  status: HTMLElement
  editButton: HTMLButtonElement
  saveButton: HTMLButtonElement
  revertButton: HTMLButtonElement
  source: string
  renderedSource: string
}

type RuntimePayload = {
  id: string
  sourcePath: string
  language: string
  pyodideIndexUrl: string
  cells: RuntimeCell[]
  toolbar?: boolean
  debug?: boolean
  vimMode?: boolean
  importableModules?: string[]
}

type AssetResult = {
  runtimeId: string
  assetId: string
  ok: boolean
  status: number
  statusText: string
  contentType: string
  bytes?: ArrayBuffer
  error?: string
}

type FrameMessage =
  | { type: 'ready'; runtimeId: string }
  | { type: 'output'; runtimeId: string; cellId: string; output: NotebookRuntimeOutput }
  | { type: 'done'; runtimeId: string; cellId: string; failed: boolean }
  | { type: 'asset'; runtimeId: string; cellId: string; assetId: string; url: string }
  | {
      type: 'file-result'
      runtimeId: string
      requestId: string
      ok: boolean
      status: number
      statusText: string
      contentType: string
      bytes?: ArrayBuffer
      error?: string
    }
  | { type: 'status'; runtimeId: string; text: string }

type CellWaiter = { resolve: (result: CellRunResult) => void; reject: (error: Error) => void }

type RuntimeFileWaiter = {
  resolve: (message: Extract<FrameMessage, { type: 'file-result' }>) => void
}

type NotebookIcon = 'run' | 'stop' | 'edit' | 'save' | 'revert' | 'copy' | 'check'

const notebookRuntimeVimModeKey = 'quartz:notebook-vim-mode'

const notebookIconSvg: Record<NotebookIcon, string> = {
  run: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5.5v13l10-6.5z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M7 7h10v10H7z"/></svg>',
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m4 16.5-.5 4 4-.5L19 8.5 15.5 5z"/><path d="m14 6.5 3.5 3.5"/></svg>',
  save: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v6h8V4"/><path d="M8 20v-6h8v6"/></svg>',
  revert:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>',
  copy: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 8h11v11H8z"/><path d="M5 16H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>',
  check:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m5 12 4 4L19 6"/></svg>',
}

function setNotebookIconButton(button: HTMLButtonElement, icon: NotebookIcon, label: string) {
  button.classList.add('notebook-icon-button')
  button.setAttribute('aria-label', label)
  button.title = label
  button.textContent = ''
  button.insertAdjacentHTML('afterbegin', notebookIconSvg[icon])
}

function readRuntimeOutput(value: unknown): NotebookRuntimeOutput | undefined {
  if (!isRecord(value)) return undefined
  const type = readString(value, 'type')
  if (type === 'stream') {
    const name = readString(value, 'name')
    const text = readString(value, 'text')
    if (name !== undefined && text !== undefined) return { type, name, text }
  }
  if (type === 'error') {
    const ename = readString(value, 'ename')
    const evalue = readString(value, 'evalue')
    const traceback = readString(value, 'traceback')
    if (ename !== undefined && evalue !== undefined && traceback !== undefined) {
      const debug = readRuntimeDebugOutput(value.debug)
      return debug ? { type, ename, evalue, traceback, debug } : { type, ename, evalue, traceback }
    }
  }
  if (type === 'text') {
    const text = readString(value, 'text')
    if (text !== undefined) return { type, text }
  }
  if (type === 'json') {
    const text = readString(value, 'text')
    if (text !== undefined) return { type, text }
  }
  if (type === 'html') {
    const html = readString(value, 'html')
    if (html !== undefined) return { type, html }
  }
  if (type === 'success') return { type }
}

function readRuntimeDebugOutput(value: unknown): RuntimeDebugOutput | undefined {
  if (!isRecord(value)) return undefined
  const phase = readString(value, 'phase')
  if (!phase) return undefined
  const debug: RuntimeDebugOutput = { phase }
  const cellId = readString(value, 'cellId')
  if (cellId !== undefined) debug.cellId = cellId
  const errorName = readString(value, 'errorName')
  if (errorName !== undefined) debug.errorName = errorName
  const errorMessage = readString(value, 'errorMessage')
  if (errorMessage !== undefined) debug.errorMessage = errorMessage
  const stack = readString(value, 'stack')
  if (stack !== undefined) debug.stack = stack
  return debug
}

function readFrameMessage(value: unknown): FrameMessage | undefined {
  if (!isRecord(value) || value.source !== 'quartz-notebook-runtime') return undefined
  const type = readString(value, 'type')
  const runtimeId = readString(value, 'runtimeId')
  if (!type || !runtimeId) return undefined
  if (type === 'ready') return { type, runtimeId }
  if (type === 'done') {
    const cellId = readString(value, 'cellId')
    if (cellId) return { type, runtimeId, cellId, failed: value.failed === true }
  }
  if (type === 'output') {
    const cellId = readString(value, 'cellId')
    const output = readRuntimeOutput(value.output)
    if (cellId && output) return { type, runtimeId, cellId, output }
  }
  if (type === 'status') {
    const text = readString(value, 'text')
    if (text !== undefined) return { type, runtimeId, text }
  }
  if (type === 'asset') {
    const cellId = readString(value, 'cellId')
    const assetId = readString(value, 'assetId')
    const url = readString(value, 'url')
    if (cellId && assetId && url) return { type, runtimeId, cellId, assetId, url }
  }
  if (type === 'file-result') {
    const requestId = readString(value, 'requestId')
    const status = readNumber(value, 'status')
    const statusText = readString(value, 'statusText')
    const contentType = readString(value, 'contentType')
    if (!requestId || status === undefined || !statusText || !contentType) return undefined
    const message: Extract<FrameMessage, { type: 'file-result' }> = {
      type,
      runtimeId,
      requestId,
      ok: value.ok === true,
      status,
      statusText,
      contentType,
    }
    if (value.bytes instanceof ArrayBuffer) message.bytes = value.bytes
    const error = readString(value, 'error')
    if (error !== undefined) message.error = error
    return message
  }
}

function readRuntimeCell(value: unknown): RuntimeCell | undefined {
  if (!isRecord(value)) return undefined
  const id = readString(value, 'id')
  const source = readString(value, 'source')
  const language = readString(value, 'language')
  const executionIndex = value.executionIndex
  if (!id || source === undefined || !language) return undefined
  if (typeof executionIndex === 'number' || executionIndex === null) {
    return { id, source, language, executionIndex }
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

function readRuntimePayload(value: unknown): RuntimePayload | undefined {
  if (!isRecord(value)) return undefined
  const id = readString(value, 'id')
  const sourcePath = readString(value, 'sourcePath')
  const language = readString(value, 'language')
  const pyodideIndexUrl = readString(value, 'pyodideIndexUrl')
  if (!id || !sourcePath || !language || !pyodideIndexUrl || !Array.isArray(value.cells)) {
    return undefined
  }
  const cells = value.cells.map(readRuntimeCell)
  if (cells.some(cell => cell === undefined)) return undefined
  const payload: RuntimePayload = {
    id,
    sourcePath,
    language,
    pyodideIndexUrl,
    cells: cells.filter(cell => cell !== undefined),
  }
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
  const scrollable = output.scrollHeight - output.clientHeight > slack
  output.toggleAttribute('data-notebook-scrollable', scrollable)
  output.toggleAttribute('data-notebook-scroll-before', scrollable && output.scrollTop > slack)
  output.toggleAttribute(
    'data-notebook-scroll-after',
    scrollable && output.scrollTop + output.clientHeight < output.scrollHeight - slack,
  )
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

function outputCopyText(outputs: HTMLElement[]): string {
  return outputs
    .map(output => output.textContent ?? '')
    .join('\n')
    .trimEnd()
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
    panelFrame.append(panel, createOutputCopyButton(group.outputs))
    panels.append(panelFrame)
  })

  container.append(tablist, panels)
  target.dataset.notebookOutputTabbed = ''
  target.replaceChildren(container)
  selectOutputTab(container, activeId)
}

function syncStaticOutputTabs(frame: HTMLElement, cellId: string) {
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
  private worker: Worker | undefined
  private ready: Promise<void> | undefined
  private readyResolve: (() => void) | undefined
  private waiters = new Map<string, CellWaiter>()
  private runtimeFileWaiters = new Map<string, RuntimeFileWaiter>()
  private moduleCache = new Map<string, RuntimeModule | null>()
  private moduleFetches = new Map<string, Promise<RuntimeModule | null>>()
  private importableModules: Set<string> | undefined
  private savedOutputs = new Map<string, HTMLElement[]>()
  private sourceControls = new Map<string, SourceControls>()
  private executionCounter: number
  private runtimeFileSequence = 0
  private running = false
  private stopped = false
  private runningCellId: string | undefined
  private debug = false
  private vimMode: boolean
  private activeCellId: string | undefined

  constructor(root: HTMLElement, payload: RuntimePayload) {
    this.root = root
    this.cellRoot = root.closest('article') ?? root.parentElement ?? document
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
    this.addCleanup(() => {
      runAll?.removeEventListener('click', this.runAll)
      stop?.removeEventListener('click', this.stop)
      reset?.removeEventListener('click', this.reset)
      debug?.removeEventListener('click', this.toggleDebug)
      vim?.removeEventListener('click', this.toggleVimMode)
      document.removeEventListener('keydown', this.handleNotebookKeydown, true)
      this.destroyRuntime()
    })
    this.syncToolbarToggles()
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

  private selectCell(cellId: string) {
    if (this.activeCellId !== undefined && this.activeCellId !== cellId) {
      this.cellFrame(this.activeCellId)?.removeAttribute('data-notebook-active-cell')
    }
    this.activeCellId = cellId
    this.cellFrame(cellId)?.setAttribute('data-notebook-active-cell', '')
  }

  private clearCellSelection() {
    if (this.activeCellId === undefined) return
    this.cellFrame(this.activeCellId)?.removeAttribute('data-notebook-active-cell')
    this.activeCellId = undefined
  }

  private cellFromTarget(target: EventTarget | null): RuntimeCell | undefined {
    if (!(target instanceof Element)) return undefined
    const frame = target.closest<HTMLElement>('[data-notebook-cell-frame]')
    if (!frame || !this.cellRoot.contains(frame)) return undefined
    return this.cellById(frame.dataset.notebookCellFrame)
  }

  private activeCellFromDocument(): RuntimeCell | undefined {
    const frame = this.cellRoot.querySelector<HTMLElement>(
      '[data-notebook-cell-frame][data-notebook-active-cell]',
    )
    const cell = this.cellById(frame?.dataset.notebookCellFrame)
    if (cell) this.activeCellId = cell.id
    return cell
  }

  private commandCell(target: EventTarget | null): RuntimeCell | undefined {
    return (
      this.cellFromTarget(target) ??
      this.activeCellFromDocument() ??
      this.cellById(this.activeCellId)
    )
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

  private notebookRunKey(event: KeyboardEvent): boolean {
    return (
      event.key === 'Enter' && (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
    )
  }

  private enterEditMode(cell: RuntimeCell) {
    this.selectCell(cell.id)
    void this.showSourceEditor(cell, true)
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

    if (this.notebookRunKey(event)) {
      this.selectCell(cell.id)
      this.claimNotebookKey(event)
      void this.runCell(cell)
      return
    }

    if (event.ctrlKey || event.metaKey || event.altKey) return

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
      button.textContent = 'Run all'
    })
    ensureButton('[data-notebook-stop]', button => {
      button.type = 'button'
      button.dataset.notebookStop = ''
      button.disabled = true
      button.textContent = 'Stop'
    })
    ensureButton('[data-notebook-reset]', button => {
      button.type = 'button'
      button.dataset.notebookReset = ''
      button.textContent = 'Reset runtime'
    })
    ensureButton('[data-notebook-debug]', button => {
      button.type = 'button'
      button.dataset.notebookDebug = ''
      button.setAttribute('aria-pressed', 'false')
      button.textContent = 'Debug'
    })
    ensureButton('[data-notebook-vim-mode]', button => {
      button.type = 'button'
      button.dataset.notebookVimMode = ''
      button.setAttribute('aria-pressed', 'false')
      button.textContent = 'Vim'
    })

    if (!status.isConnected) toolbar.append(status)
    if (!toolbar.isConnected) this.root.append(toolbar)
  }

  private readStoredVimMode(): boolean {
    try {
      return window.localStorage.getItem(notebookRuntimeVimModeKey) === 'true'
    } catch {
      return false
    }
  }

  private writeStoredVimMode(enabled: boolean) {
    try {
      window.localStorage.setItem(notebookRuntimeVimModeKey, enabled ? 'true' : 'false')
    } catch {}
  }

  private syncToolbarToggles() {
    const debug = this.root.querySelector<HTMLButtonElement>('[data-notebook-debug]')
    debug?.setAttribute('aria-pressed', String(this.debug))
    const vim = this.root.querySelector<HTMLButtonElement>('[data-notebook-vim-mode]')
    vim?.setAttribute('aria-pressed', String(this.vimMode))
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
      const unsupported = unsupportedNotebookRuntimeReason(source)
      if (unsupported) {
        this.renderOutput(cell.id, {
          type: 'error',
          ename: 'UnsupportedRuntimeFeature',
          evalue: unsupported,
          traceback: unsupported,
        })
        return false
      }
      await this.ensureWorker()
      const result = await this.postRun(cell, source)
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
      if (!this.stopped) this.setStatus('idle')
    }
  }

  private stop = () => {
    this.stopped = true
    this.destroyRuntime()
    this.running = false
    this.runningCellId = undefined
    this.setRunningControls(false)
    this.setStatus('stopped')
    for (const waiter of this.waiters.values()) {
      waiter.reject(new Error('runtime stopped'))
    }
    this.waiters.clear()
  }

  private reset = () => {
    this.stopped = false
    this.destroyRuntime()
    for (const waiter of this.waiters.values()) {
      waiter.reject(new Error('runtime reset'))
    }
    this.waiters.clear()
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

    actions.replaceChildren(runButton, editButton, saveButton, revertButton, status)
    if (!actions.isConnected) frame.append(actions)
    if (figure) {
      if (!editorHost.isConnected) figure.before(editorHost)
    } else {
      if (!editorHost.isConnected) frame.append(editorHost)
    }
    const stored = this.readStoredSource(cell)
    this.sourceControls.set(cell.id, {
      frame,
      editor: undefined,
      editorHost,
      figure,
      status,
      editButton,
      saveButton,
      revertButton,
      source: stored ?? cell.source,
      renderedSource: cell.source,
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
      return true
    } catch {
      return false
    }
  }

  private clearStoredSource(cell: RuntimeCell) {
    try {
      window.localStorage.removeItem(this.sourceStorageKey(cell))
    } catch {}
  }

  private async ensureSourceEditor(cell: RuntimeCell, controls: SourceControls) {
    if (controls.editor) return controls.editor
    const { createNotebookCodeEditor } = await import('./notebook-code-editor')
    controls.editor = await createNotebookCodeEditor({
      parent: controls.editorHost,
      initialContent: controls.source,
      language: cell.language,
      vimMode: this.vimMode,
      lsp: {
        enabled: true,
        runtimeId: this.payload.id,
        sourcePath: this.payload.sourcePath,
        cellId: cell.id,
        language: cell.language,
      },
      onChange: source => {
        controls.source = source
        this.syncSourceControls(cell)
      },
      onSubmit: () => this.runCell(cell),
      onSave: () => {
        void this.saveEditorSource(cell)
      },
      onCancel: () => {
        void this.closeSourceEditor(cell)
      },
    })
    return controls.editor
  }

  private async showSourceEditor(cell: RuntimeCell, visible: boolean) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    if (visible) {
      const editor = await this.ensureSourceEditor(cell, controls)
      editor.setValue(this.sourceForCell(cell))
      controls.source = editor.getValue()
    }
    controls.editorHost.hidden = !visible
    if (controls.figure) controls.figure.hidden = visible
    controls.frame.toggleAttribute('data-notebook-editing', visible)
    this.syncSourceControls(cell)
    if (visible) {
      controls.editor?.focus()
    } else {
      this.selectCell(cell.id)
      controls.frame.focus({ preventScroll: true })
    }
  }

  private async highlightedSourceLines(
    cell: RuntimeCell,
    controls: SourceControls,
    source: string,
    preferredLines?: HTMLElement[],
  ): Promise<HTMLElement[] | undefined> {
    if (preferredLines && preferredLines.length === source.split(/\r?\n/).length)
      return preferredLines
    if (!controls.figure) return undefined
    try {
      const { renderNotebookHighlightedLines } = await import('./notebook-code-editor')
      return await renderNotebookHighlightedLines(source, cell.language)
    } catch {
      return undefined
    }
  }

  private async replaceRenderedCellSource(
    cell: RuntimeCell,
    controls: SourceControls,
    source: string,
    preferredLines?: HTMLElement[],
  ) {
    if (!controls.figure || controls.renderedSource === source) return
    const highlightedLines = await this.highlightedSourceLines(
      cell,
      controls,
      source,
      preferredLines,
    )
    replaceRenderedSource(controls.figure, source, highlightedLines)
    controls.renderedSource = source
  }

  private async closeSourceEditor(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    let highlightedLines: HTMLElement[] | undefined
    if (controls.editor) {
      controls.source = controls.editor.getValue()
      highlightedLines = controls.editor.highlightedLines()
    }
    await this.replaceRenderedCellSource(cell, controls, controls.source, highlightedLines)
    void this.showSourceEditor(cell, false)
  }

  private async saveEditorSource(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    const source = this.sourceForCell(cell)
    const highlightedLines = controls.editor?.highlightedLines()
    if (source === cell.source) {
      controls.source = cell.source
      this.clearStoredSource(cell)
      await this.replaceRenderedCellSource(cell, controls, cell.source, highlightedLines)
      void this.showSourceEditor(cell, false)
      this.setStatus('idle')
      return
    }
    if (!this.writeStoredSource(cell, source)) {
      this.setStatus('local save failed')
      return
    }
    controls.source = source
    await this.replaceRenderedCellSource(cell, controls, source, highlightedLines)
    void this.showSourceEditor(cell, false)
    this.setStatus('saved local edit')
  }

  private async revertEditorSource(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    controls.source = cell.source
    controls.editor?.setValue(cell.source)
    const highlightedLines = controls.editor?.highlightedLines()
    this.clearStoredSource(cell)
    await this.replaceRenderedCellSource(cell, controls, cell.source, highlightedLines)
    void this.showSourceEditor(cell, false)
    this.setStatus('idle')
  }

  private syncSourceControls(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    const stored = this.readStoredSource(cell)
    const editing = !controls.editorHost.hidden
    const edited = editing && this.sourceForCell(cell) !== (stored ?? cell.source)
    const hasStoredSource = stored !== undefined
    controls.status.hidden = !hasStoredSource && !edited
    controls.status.dataset.notebookSourceState = edited ? 'edited' : 'local'
    controls.status.title = edited ? 'edited local source' : 'saved local source'
    controls.status.setAttribute(
      'aria-label',
      edited ? 'edited local source' : 'saved local source',
    )
    controls.editButton.hidden = editing
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

  private ensureWorker(): Promise<void> {
    if (this.ready) return this.ready
    this.ready = new Promise(resolve => {
      this.readyResolve = resolve
    })
    const worker = new Worker(new URL('notebook-runtime.worker.js', import.meta.url), {
      type: 'module',
    })
    worker.addEventListener('message', this.onWorkerMessage)
    worker.addEventListener('error', this.onWorkerError)
    worker.postMessage({
      source: 'quartz-notebook-runtime',
      type: 'init',
      runtimeId: this.payload.id,
    })
    this.worker = worker
    return this.ready
  }

  private async postRun(cell: RuntimeCell, source: string): Promise<CellRunResult> {
    const modules = await this.notebookModulesFor(source)
    return new Promise((resolve, reject) => {
      this.waiters.set(cell.id, { resolve, reject })
      this.worker?.postMessage({
        source: 'quartz-notebook-runtime',
        type: 'run',
        runtimeId: this.payload.id,
        cellId: cell.id,
        code: source,
        pyodideIndexUrl: this.payload.pyodideIndexUrl,
        debug: this.debug,
        modules,
      })
    })
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

  private async notebookModulesFor(source: string): Promise<RuntimeModule[]> {
    const seen = new Set<string>()
    const modules = new Map<string, RuntimeModule>()
    await this.collectNotebookModules(source, seen, modules)
    return [...modules.values()]
  }

  private async collectNotebookModules(
    source: string,
    seen: Set<string>,
    modules: Map<string, RuntimeModule>,
  ): Promise<void> {
    for (const name of notebookRuntimeImportCandidates(source)) {
      if (seen.has(name)) continue
      seen.add(name)
      const module = await this.fetchNotebookModule(name)
      if (!module) continue
      modules.set(name, module)
      await this.collectNotebookModules(module.source, seen, modules)
    }
  }

  private async fetchNotebookModule(name: string): Promise<RuntimeModule | null> {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) return null
    const cached = this.moduleCache.get(name)
    if (cached !== undefined) return cached
    const pending = this.moduleFetches.get(name)
    if (pending) return await pending
    const request = this.loadNotebookModule(name)
    this.moduleFetches.set(name, request)
    try {
      const module = await request
      this.moduleCache.set(name, module)
      return module
    } finally {
      this.moduleFetches.delete(name)
    }
  }

  private async loadNotebookModule(name: string): Promise<RuntimeModule | null> {
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
    const source = notebookRuntimeModuleSource(await response.text(), `${name}.ipynb`)
    return { name, sourcePath: url.pathname, source }
  }

  private onWorkerError = (event: ErrorEvent) => {
    this.readyResolve?.()
    this.readyResolve = undefined
    for (const waiter of this.waiters.values()) {
      waiter.reject(event.error instanceof Error ? event.error : new Error(event.message))
    }
    this.waiters.clear()
  }

  private onWorkerMessage = (event: MessageEvent<unknown>) => {
    const message = readFrameMessage(event.data)
    if (!message || message.runtimeId !== this.payload.id) return
    this.handleRuntimeMessage(message, this.worker)
  }

  private handleRuntimeMessage(message: FrameMessage, target: Worker | undefined) {
    if (message.type === 'ready') {
      this.readyResolve?.()
      this.readyResolve = undefined
    } else if (message.type === 'output') {
      if (message.output.type === 'error' && message.output.ename === 'UnsupportedRuntimeFeature') {
        this.restoreSavedOutput(message.cellId)
      }
      this.renderOutput(message.cellId, message.output)
    } else if (message.type === 'done') {
      const waiter = this.waiters.get(message.cellId)
      if (!waiter) return
      this.waiters.delete(message.cellId)
      waiter.resolve({ failed: message.failed })
    } else if (message.type === 'asset') {
      void this.fetchAsset(message, target)
    } else if (message.type === 'file-result') {
      if (!(target instanceof Worker)) return
      const waiter = this.runtimeFileWaiters.get(message.requestId)
      if (!waiter) return
      this.runtimeFileWaiters.delete(message.requestId)
      waiter.resolve(message)
    } else if (message.type === 'status') {
      this.setStatus(message.text)
    }
  }

  private async fetchAsset(
    message: Extract<FrameMessage, { type: 'asset' }>,
    target: Worker | undefined,
  ) {
    const result = await this.resolveAsset(message)
    if (!target) return
    const transfer = result.bytes ? [result.bytes] : []
    const payload = { source: 'quartz-notebook-runtime', type: 'asset-result', ...result }
    target.postMessage(payload, transfer)
  }

  private async resolveAsset(
    message: Extract<FrameMessage, { type: 'asset' }>,
  ): Promise<AssetResult> {
    const fallback = (error: unknown): AssetResult => ({
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
        (await this.resolveRuntimeFileAsset(message, relative || message.url)) ?? fallback(error)
      )
    }
  }

  private resolveRuntimeFileAsset(
    message: Extract<FrameMessage, { type: 'asset' }>,
    path: string,
  ): Promise<AssetResult | undefined> {
    if (!this.worker) return Promise.resolve(undefined)
    const requestId = `runtime-file-${++this.runtimeFileSequence}`
    return new Promise(resolve => {
      const timeout = window.setTimeout(() => {
        this.runtimeFileWaiters.delete(requestId)
        resolve(undefined)
      }, 10_000)
      this.runtimeFileWaiters.set(requestId, {
        resolve: result => {
          window.clearTimeout(timeout)
          resolve({
            runtimeId: message.runtimeId,
            assetId: message.assetId,
            ok: result.ok,
            status: result.status,
            statusText: result.statusText,
            contentType: result.contentType,
            bytes: result.bytes,
            error: result.error,
          })
        },
      })
      this.worker?.postMessage({
        source: 'quartz-notebook-runtime',
        type: 'file',
        runtimeId: this.payload.id,
        requestId,
        path,
      })
    })
  }

  private renderOutput(cellId: string, output: NotebookRuntimeOutput) {
    const target = this.queryCell<HTMLElement>(`[data-notebook-output="${CSS.escape(cellId)}"]`)
    if (!target) return
    this.hideSavedOutput(cellId)
    target.hidden = false
    appendRuntimeOutput(target, output, { debug: this.debug })
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
    this.worker?.removeEventListener('message', this.onWorkerMessage)
    this.worker?.removeEventListener('error', this.onWorkerError)
    this.worker?.terminate()
    this.worker = undefined
    this.runtimeFileWaiters.clear()
    this.ready = undefined
    this.readyResolve = undefined
  }
}

export function mountNotebookRuntime(root: HTMLElement, text: string) {
  if (root.dataset.runtimeMounted === 'true') return
  const parsed = parseRuntimeJson(text)
  if (parsed === undefined) return
  const payload = readRuntimePayload(parsed)
  if (!payload) return
  root.dataset.runtimeMounted = 'true'
  new NotebookRuntime(root, payload).mount()
}
