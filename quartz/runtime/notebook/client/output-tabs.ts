import { notebookIconSvg } from '../../../util/notebook/render/icons'
import { notebookSuccessOutputLabel } from '../types'

export function outputClassToken(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, '-') || 'output'
}

export function outputLabel(output: HTMLElement): string {
  return output.dataset.outputName?.trim() || 'output'
}

export function outputTabId(label: string): string {
  return outputClassToken(label.toLowerCase())
}

export function collectOutputElements(target: HTMLElement): HTMLElement[] {
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

export function hasRenderedOutput(target: HTMLElement): boolean {
  return collectOutputElements(target).length > 0
}

export const outputScrollHintTargets = new WeakSet<HTMLElement>()
export const pendingOutputScrollHints = new WeakSet<HTMLElement>()

export function syncOutputScrollHint(output: HTMLElement) {
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

export function syncOutputScrollHints(root: HTMLElement) {
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
    if (!pendingOutputScrollHints.has(output)) {
      pendingOutputScrollHints.add(output)
      requestAnimationFrame(() => {
        pendingOutputScrollHints.delete(output)
        syncOutputScrollHint(output)
      })
    }
  }
}

export function showNotebookOutputToast(message: string) {
  const event: CustomEventMap['toast'] = new CustomEvent('toast', {
    detail: { message, containerId: 'notebook-output-toast-container' },
  })
  document.dispatchEvent(event)
}

export function outputCopyText(outputs: HTMLElement[]): string {
  return outputs
    .map(output => output.textContent ?? '')
    .join('\n')
    .trimEnd()
}

export function selectedOutputTabId(container: HTMLElement): string | undefined {
  return container.querySelector<HTMLElement>('[data-notebook-output-tab][aria-selected="true"]')
    ?.dataset.notebookOutputTab
}

export function activeOutputPanel(container: HTMLElement): HTMLElement | undefined {
  const selectedId = selectedOutputTabId(container)
  if (!selectedId) return undefined
  for (const frame of container.querySelectorAll<HTMLElement>('[data-notebook-output-panel]')) {
    if (frame.dataset.notebookOutputPanel !== selectedId) continue
    return frame.querySelector<HTMLElement>(':scope > .notebook-output-panel') ?? undefined
  }
}

export function outputPanelCollapsedHeight(container: HTMLElement): number {
  const height = Number.parseFloat(
    window.getComputedStyle(container).getPropertyValue('--notebook-output-panel-height'),
  )
  return Number.isFinite(height) && height > 0 ? height : 150
}

export function syncOutputExpandButton(
  button: HTMLButtonElement,
  expandable: boolean,
  expanded: boolean,
) {
  const label = expanded ? 'Collapse output' : 'Expand output'
  button.toggleAttribute('data-notebook-output-expandable', expandable)
  button.setAttribute('aria-expanded', String(expanded))
  button.setAttribute('aria-label', label)
  button.title = label
}

export function syncOutputExpandControls(container: HTMLElement) {
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

export function createOutputExpandButton(
  container: HTMLElement,
  groupId: string,
): HTMLButtonElement {
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

export function createOutputCopyButton(outputs: HTMLElement[]): HTMLButtonElement {
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

export function selectOutputTab(
  container: HTMLElement,
  activeId: string | undefined,
  focusId = activeId,
) {
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

export function syncOutputTabs(
  target: HTMLElement,
  cellId = target.dataset.notebookOutput ?? 'cell',
) {
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

export function syncStaticOutputTabs(frame: HTMLElement, cellId: string) {
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
