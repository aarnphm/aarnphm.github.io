import type {
  NotebookKernelCommand,
  NotebookKernelSnapshot,
  NotebookKernelStatus,
} from '../../../util/notebook-kernel-events'
import { notebookLocalSourcesClearedEvent } from '../../../util/notebook-source-events'
import {
  setTriActivityLayout,
  TRI_ACTIVITY_LAYOUTS,
  type TriActivityLayout,
} from '../../../util/triathlon-activity-layout'
import { notifyToast } from './comment-author'
import { appendActionAux, appendHighlightedText, appendTrustedSvg } from './format'
import {
  clearNotebookLocalSources,
  dispatchNotebookKernelCommand,
  dispatchNotebookKernelRunAll,
  notebookKernelLanguageIcon,
  notebookKernelLanguageToken,
  notebookKernelSnapshots,
  notebookKernelSourceLabel,
  notebookKernelStatusLabel,
} from './notebook-kernel'
import {
  commandInputValue,
  hidePalette,
  paletteMode,
  setFocusFirstChild,
  syncPaletteHelper,
  type Action,
  type CommandLocation,
  type PaletteState,
} from './shell'

export function sizePaletteToDefaultItems(s: PaletteState) {
  const panel = s.output.parentElement
  if (!panel || !s.container.matches(':popover-open')) return
  const width = panel.clientWidth
  if (width === s.measuredPaletteWidth) return
  const measurement = document.createElement('output')
  measurement.setAttribute('aria-hidden', 'true')
  measurement.inert = true
  measurement.style.cssText = `position:absolute;visibility:hidden;pointer-events:none;flex:none;height:auto;max-height:none;overflow:visible;box-sizing:border-box;width:${width}px`
  panel.append(measurement)
  let itemsHeight = 0
  for (const actions of [s.connectorActs, s.commandActs]) {
    measurement.replaceChildren(
      ...actions.map(action =>
        (s.actionCache.get(action) ?? actionComponent(s, action)).cloneNode(true),
      ),
    )
    itemsHeight = Math.max(itemsHeight, measurement.getBoundingClientRect().height)
  }
  const inputHeight = s.bar.parentElement?.getBoundingClientRect().height ?? 0
  const height =
    itemsHeight +
    inputHeight +
    s.helper.getBoundingClientRect().height +
    panel.offsetHeight -
    panel.clientHeight
  measurement.remove()
  panel.style.setProperty('--palette-content-height', `${Math.ceil(height)}px`)
  s.measuredPaletteWidth = width
}

const TRI_ACTIVITY_LAYOUT_LABELS: Record<TriActivityLayout, string> = {
  one: 'one activity per view',
  two: 'two activities per view',
  all: 'all activities',
}
const pressKit = (tab: 'note' | 'quote' | 'links') => () =>
  document.dispatchEvent(new CustomEvent('presskitopen', { detail: { tab } }))
export function commandActions(s: PaletteState): Action[] {
  return [
    {
      name: 'press kit: note card',
      auxInnerHtml: '<kbd>gs</kbd> image',
      onClick: pressKit('note'),
    },
    {
      name: 'press kit: quote card',
      auxInnerHtml: '<kbd>↵</kbd> selection',
      onClick: pressKit('quote'),
    },
    { name: 'press kit: links', auxInnerHtml: '<kbd>↵</kbd> share', onClick: pressKit('links') },
    ...TRI_ACTIVITY_LAYOUTS.map(
      (layout): Action => ({
        name: `triathlon: ${TRI_ACTIVITY_LAYOUT_LABELS[layout]}`,
        auxInnerHtml: '<kbd>↵</kbd> activity layout',
        onClick: () => {
          setTriActivityLayout(layout)
          notifyToast(`activities: ${TRI_ACTIVITY_LAYOUT_LABELS[layout]}`)
        },
      }),
    ),
    {
      name: 'kernels: run all',
      auxInnerHtml: '<kbd>↵</kbd> current page',
      onClick: () => {
        dispatchNotebookKernelRunAll()
        notifyToast('running all notebook kernels')
      },
    },
    {
      name: 'kernels: show all',
      auxInnerHtml: '<kbd>↵</kbd> notebooks',
      keepOpen: true,
      onClick: () => {
        showKernelItems(s)
      },
    },
    {
      name: 'general: reset code segments',
      auxInnerHtml: '<kbd>↵</kbd> localStorage',
      keepOpen: true,
      onClick: () => {
        const count = clearNotebookLocalSources()
        const event: CustomEventMap['notebooklocalsourcescleared'] = new CustomEvent(
          notebookLocalSourcesClearedEvent,
          { detail: { count } },
        )
        document.dispatchEvent(event)
        notifyToast(`cleared ${count} saved code segment${count === 1 ? '' : 's'}`)
        showCommandItems(s, true)
      },
    },
  ]
}

function createStatusComponent(tone: NotebookKernelStatus, label: string | undefined) {
  const status = document.createElement('span')
  status.className = 'suggestion-status'
  status.dataset.statusTone = tone
  status.setAttribute('aria-label', label ? `${tone}: ${label}` : tone)
  if (label) status.textContent = label
  return status
}

export const createActComponent = (
  s: PaletteState,
  {
    name,
    auxInnerHtml,
    onClick,
    keepOpen,
    detail,
    titlePrefixInnerHtml,
    statusLabel,
    statusTone,
  }: Action,
  query = paletteMode.currentSearchTerm,
) => {
  const item = document.createElement('div')
  item.classList.add('suggestion-item')
  item.tabIndex = -1

  const content = document.createElement('div')
  content.classList.add('suggestion-content')
  const title = document.createElement('div')
  title.classList.add('suggestion-title')
  if (titlePrefixInnerHtml) {
    const prefix = document.createElement('span')
    prefix.className = 'suggestion-title-prefix'
    appendTrustedSvg(prefix, titlePrefixInnerHtml)
    title.append(prefix)
  }
  appendHighlightedText(title, query, name)
  content.appendChild(title)
  if (detail) {
    const subscript = document.createElement('span')
    subscript.className = 'subscript'
    subscript.textContent = detail
    content.appendChild(subscript)
  }

  const aux = document.createElement('div')
  aux.classList.add('suggestion-aux')
  if (statusTone) aux.append(createStatusComponent(statusTone, statusLabel))
  if (auxInnerHtml.length > 0) appendActionAux(aux, auxInnerHtml)
  item.append(content, aux)

  function mainOnClick(e: MouseEvent) {
    e.preventDefault()
    onClick(e)
    if (keepOpen !== true) hidePalette(s)
  }
  item.addEventListener('click', mainOnClick)
  window.addCleanup(() => item.removeEventListener('click', mainOnClick))
  return item
}

function actionComponent(s: PaletteState, action: Action) {
  if (paletteMode.currentSearchTerm) return createActComponent(s, action)
  let item = s.actionCache.get(action)
  if (!item) {
    item = createActComponent(s, action, '')
    s.actionCache.set(action, item)
  }
  item.classList.remove('focus')
  return item
}

function transitionCommandItems(s: PaletteState) {
  s.output.removeAttribute('data-palette-transition')
  void s.output.offsetHeight
  s.output.dataset.paletteTransition = ''
  window.setTimeout(() => {
    s.output.removeAttribute('data-palette-transition')
  }, 220)
}

export function getCommandItems(s: PaletteState, acts: Action[], transition = false) {
  if (s.output) {
    s.output.replaceChildren()
  }
  s.currentHover = null
  if (acts.length === 0) {
    if (s.bar.matches(':focus') && paletteMode.currentSearchTerm === '') {
      s.output.append(
        ...(paletteMode.actionType === 'command' ? s.commandActs : s.connectorActs).map(action =>
          actionComponent(s, action),
        ),
      )
    } else {
      const fallbackAction =
        paletteMode.actionType === 'command' ? s.commandActs[0] : s.connectorActs[0]
      if (fallbackAction) s.output.append(actionComponent(s, fallbackAction))
    }
  } else {
    s.output.append(...acts.map(action => actionComponent(s, action)))
  }
  setFocusFirstChild(s)
  if (transition) transitionCommandItems(s)
}

export function recordCommandLocation(s: PaletteState, location: CommandLocation) {
  s.commandHistory = s.commandHistory.slice(0, s.commandHistoryIndex + 1)
  s.commandHistory.push(location)
  s.commandHistoryIndex = s.commandHistory.length - 1
}

function renderCommandLocation(s: PaletteState, location: CommandLocation, transition = false) {
  s.queryGeneration += 1
  paletteMode.actionType = 'command'
  syncPaletteHelper(s)
  if (location.type === 'commands') {
    s.bar.value = commandInputValue()
    paletteMode.currentSearchTerm = ''
    getCommandItems(s, s.commandActs, transition)
    return
  }
  if (location.type === 'kernels') {
    renderKernelItems(s, transition)
    return
  }
  renderKernelActionItems(s, location.snapshot, transition)
}

function showCommandLocation(
  s: PaletteState,
  location: CommandLocation,
  transition = false,
  record = true,
) {
  if (record) recordCommandLocation(s, location)
  renderCommandLocation(s, location, transition)
}

export function showCommandItems(s: PaletteState, transition = false, record = true) {
  showCommandLocation(s, { type: 'commands' }, transition, record)
}

function showKernelItems(s: PaletteState, transition = true, record = true) {
  showCommandLocation(s, { type: 'kernels' }, transition, record)
}

function renderKernelItems(s: PaletteState, transition = true) {
  s.bar.value = commandInputValue('kernels')
  paletteMode.currentSearchTerm = ''
  const snapshots = notebookKernelSnapshots()
  const acts: Action[] =
    snapshots.length === 0
      ? [
          {
            name: 'no notebook kernels in this view',
            auxInnerHtml: '',
            keepOpen: true,
            onClick: () => {},
          },
        ]
      : snapshots.map(snapshot => ({
          name: notebookKernelLanguageToken(snapshot.language),
          detail: notebookKernelSourceLabel(snapshot.sourcePath),
          titlePrefixInnerHtml: notebookKernelLanguageIcon(snapshot.language),
          auxInnerHtml: '',
          keepOpen: true,
          statusLabel: notebookKernelStatusLabel(snapshot),
          statusTone: snapshot.status,
          onClick: () => showKernelActionItems(s, snapshot),
        }))
  getCommandItems(s, acts, transition)
}

function showKernelActionItems(
  s: PaletteState,
  snapshot: NotebookKernelSnapshot,
  transition = true,
  record = true,
) {
  showCommandLocation(s, { type: 'kernel-actions', snapshot }, transition, record)
}

function renderKernelActionItems(
  s: PaletteState,
  snapshot: NotebookKernelSnapshot,
  transition = true,
) {
  s.bar.value = commandInputValue(`kernels ${snapshot.language}`)
  paletteMode.currentSearchTerm = ''
  const statusLabel = notebookKernelStatusLabel(snapshot)
  const detail = statusLabel
    ? `${notebookKernelSourceLabel(snapshot.sourcePath)} - ${statusLabel}`
    : notebookKernelSourceLabel(snapshot.sourcePath)
  const action = (command: NotebookKernelCommand): Action => ({
    name: `${command}`,
    detail,
    auxInnerHtml: '',
    keepOpen: true,
    onClick: () => {
      dispatchNotebookKernelCommand(snapshot, command)
      notifyToast(`${snapshot.language} kernel ${command} requested`)
      window.setTimeout(() => showKernelItems(s, true, false), 140)
    },
  })
  getCommandItems(s, [action('interrupt'), action('restart'), action('kill')], transition)
}

export function navigateCommandHistory(s: PaletteState, delta: -1 | 1): boolean {
  if (paletteMode.actionType !== 'command') return false
  const nextIndex = s.commandHistoryIndex + delta
  if (nextIndex < 0 || nextIndex >= s.commandHistory.length) return false
  s.commandHistoryIndex = nextIndex
  renderCommandLocation(s, s.commandHistory[s.commandHistoryIndex], true)
  return true
}
