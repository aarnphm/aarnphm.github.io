import DOMPurify from 'dompurify'
import type { KernelOutput } from '../kernel'
import type { NotebookRuntimeState } from './runtime-state'
import { renderOutputHtml } from '../../../util/notebook/render/output-to-hast'
import {
  notebookSuccessOutputLabel,
  type NotebookRuntimeDebugOutput,
  type NotebookRuntimeOutput,
} from '../types'
import { queryCell } from './cell-navigation'
import {
  hasRenderedOutput,
  outputClassToken,
  outputTabId,
  syncOutputScrollHints,
  syncOutputTabs,
} from './output-tabs'

export type RuntimeErrorOutput = Extract<NotebookRuntimeOutput, { type: 'error' }>

export type RuntimeStreamOutput = Extract<NotebookRuntimeOutput, { type: 'stream' }>

export type RuntimeDebugOutput = NonNullable<RuntimeErrorOutput['debug']>

export function debugOutputText(debug: NotebookRuntimeDebugOutput): string {
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

export function createPreOutput(
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

export function lastNotebookOutput(target: HTMLElement): HTMLElement | undefined {
  const direct = target.lastElementChild
  if (direct instanceof HTMLElement && direct.classList.contains('notebook-output')) return direct
  const tabbed = target.querySelectorAll<HTMLElement>(
    ':scope > [data-notebook-output-tabs] > .notebook-output-panels > .notebook-output-panel-frame > .notebook-output-panel > .notebook-output',
  )
  return tabbed[tabbed.length - 1]
}

export function appendTextInPlace(target: HTMLElement, text: string) {
  const firstChild = target.firstChild
  if (
    firstChild &&
    firstChild.nodeType === 3 &&
    firstChild === target.lastChild &&
    typeof (firstChild as Text).appendData === 'function'
  ) {
    ;(firstChild as Text).appendData(text)
    return
  }
  target.textContent = (target.textContent ?? '') + text
}

export function findOutputPanel(container: HTMLElement, groupId: string): HTMLElement | undefined {
  for (const frame of container.querySelectorAll<HTMLElement>(
    ':scope > .notebook-output-panels > .notebook-output-panel-frame[data-notebook-output-panel]',
  )) {
    if (frame.dataset.notebookOutputPanel !== groupId) continue
    return frame.querySelector<HTMLElement>(':scope > .notebook-output-panel') ?? undefined
  }
}

export function appendStreamOutput(target: HTMLElement, output: RuntimeStreamOutput): boolean {
  const container = target.querySelector<HTMLElement>(':scope > [data-notebook-output-tabs]')
  const hasDirectOutputs = Array.from(target.children).some(
    child => child instanceof HTMLElement && child.classList.contains('notebook-output'),
  )
  if (container && !hasDirectOutputs) {
    const label = output.name.trim() || 'output'
    const groupId = outputTabId(label)
    const panel = findOutputPanel(container, groupId)
    if (panel) {
      const lastInPanel = panel.lastElementChild
      if (
        lastInPanel instanceof HTMLPreElement &&
        lastInPanel.classList.contains('notebook-output-stream') &&
        lastInPanel.dataset.outputName === output.name
      ) {
        const sample = lastInPanel.querySelector<HTMLElement>('samp') ?? lastInPanel
        appendTextInPlace(sample, output.text)
        lastInPanel.scrollTop = lastInPanel.scrollHeight
        panel.scrollTop = panel.scrollHeight
        syncOutputScrollHints(panel)
        return true
      }
    }
  }

  const previous = lastNotebookOutput(target)
  if (
    previous instanceof HTMLPreElement &&
    previous.classList.contains('notebook-output-stream') &&
    previous.dataset.outputName === output.name
  ) {
    const sample = previous.querySelector<HTMLElement>('samp')
    if (sample) {
      appendTextInPlace(sample, output.text)
      previous.scrollTop = previous.scrollHeight
      syncOutputScrollHints(target)
      return container !== null && !hasDirectOutputs
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
  return false
}

export function createHtmlOutput(html: string): HTMLDivElement {
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

export function createSuccessOutput(): HTMLDivElement {
  const output = document.createElement('div')
  output.classList.add('notebook-output', 'notebook-output-success')
  output.dataset.outputName = notebookSuccessOutputLabel
  return output
}

export function appendRuntimeOutput(
  target: HTMLElement,
  output: NotebookRuntimeOutput,
  options: { debug?: boolean } = {},
) {
  if (output.type === 'stream') {
    if (!appendStreamOutput(target, output)) syncOutputTabs(target)
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

export function debugOutput(phase: string, cellId: string, error: unknown): RuntimeDebugOutput {
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

export function renderOutput(
  state: NotebookRuntimeState,
  cellId: string,
  output: NotebookRuntimeOutput,
) {
  const target = queryCell<HTMLElement>(state, `[data-notebook-output="${CSS.escape(cellId)}"]`)
  if (!target) return
  hideSavedOutput(state, cellId)
  target.hidden = false
  appendRuntimeOutput(target, output, { debug: state.debug })
}

export function renderKernelOutput(
  state: NotebookRuntimeState,
  cellId: string,
  output: KernelOutput,
) {
  if ('kind' in output) {
    if (output.kind === 'stream') {
      renderOutput(state, cellId, { type: 'stream', name: output.name, text: output.text })
      return
    }
    const target = queryCell<HTMLElement>(state, `[data-notebook-output="${CSS.escape(cellId)}"]`)
    if (!target) return
    hideSavedOutput(state, cellId)
    target.hidden = false
    target.insertAdjacentHTML('beforeend', renderOutputHtml(output))
    syncOutputTabs(target)
    return
  }
  renderOutput(state, cellId, output)
}

export function renderSuccessOutput(state: NotebookRuntimeState, cellId: string) {
  const target = queryCell<HTMLElement>(state, `[data-notebook-output="${CSS.escape(cellId)}"]`)
  if (!target || hasRenderedOutput(target)) return
  renderOutput(state, cellId, { type: 'success' })
}

export function clearOutput(state: NotebookRuntimeState, cellId: string) {
  const target = queryCell<HTMLElement>(state, `[data-notebook-output="${CSS.escape(cellId)}"]`)
  if (!target) return
  hideSavedOutput(state, cellId)
  target.replaceChildren()
  target.hidden = true
  target.removeAttribute('data-notebook-output-tabbed')
}

export function hideSavedOutput(state: NotebookRuntimeState, cellId: string) {
  const existing = state.savedOutputs.get(cellId)
  if (existing) {
    existing.forEach(element => {
      element.hidden = true
    })
    return
  }
  const target = queryCell<HTMLElement>(state, `[data-notebook-output="${CSS.escape(cellId)}"]`)
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
  state.savedOutputs.set(cellId, saved)
  saved.forEach(element => {
    element.hidden = true
  })
}

export function restoreSavedOutput(state: NotebookRuntimeState, cellId: string) {
  const saved = state.savedOutputs.get(cellId)
  if (!saved) return
  saved.forEach(element => {
    element.hidden = false
  })
  state.savedOutputs.delete(cellId)
}
