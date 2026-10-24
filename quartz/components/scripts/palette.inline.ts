import { parseCodeQuery } from '../../util/code-query'
import { getFullSlug, isFullSlug, pathToRoot } from '../../util/path'
import { parseSearchQuery } from '../../util/search-scope'
import { CodeSearchClient } from './code-search'
import { registerEscapeHandler } from './escape-handler'
import { createImageSimilarity } from './image-similarity'
import { currentNavSignal } from './nav-lifecycle'
import {
  commandActions,
  createActComponent,
  getCommandItems,
  navigateCommandHistory,
  recordCommandLocation,
  showCommandItems,
  sizePaletteToDefaultItems,
} from './palette/actions'
import { connectorActions } from './palette/connectors'
import { fetchContent } from './palette/format'
import { displayNearbyNotes, displayQuickOpenResults, querySearch } from './palette/search'
import {
  commandInputValue,
  commandSearchTerm,
  hidePalette,
  paletteMode,
  quickOpenProgress,
  quickOpenStatus,
  syncPaletteHelper,
  type Action,
  type ActionType,
  type FallbackActionType,
  type PaletteState,
} from './palette/shell'
import { PdfSemantic } from './pdf-semantic'
import { createSearchCompletions } from './search-completions'
import { beginSidePanelRequest, disposeSidePanel, getOrCreateSidePanel } from './side-panel'

let activePaletteSignal: AbortSignal | undefined

function showPalette(
  s: PaletteState,
  actionTypeNew: ActionType,
  fallbackActionTypeNew?: FallbackActionType,
) {
  s.resultPages?.dispose()
  s.resultPages = null
  s.queryGeneration += 1
  paletteMode.actionType = actionTypeNew
  paletteMode.fallbackActionType =
    actionTypeNew === 'command'
      ? (fallbackActionTypeNew ?? paletteMode.fallbackActionType)
      : actionTypeNew
  if (s.container && !s.container.matches(':popover-open')) {
    s.previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    s.container.showPopover()
  }
  s.container?.classList.add('active')
  if (paletteMode.actionType === 'command') {
    showCommandItems(s)
  } else if (paletteMode.actionType === 'connector') {
    s.bar.value = ''
    paletteMode.currentSearchTerm = ''
    syncPaletteHelper(s)
    getCommandItems(s, s.connectorActs)
  } else if (paletteMode.actionType === 'quick_open') {
    s.bar.value = ''
    paletteMode.currentSearchTerm = ''
    syncPaletteHelper(s)
    quickOpenStatus(s, '')
    quickOpenProgress(s, false)
    void displayNearbyNotes(s, s.queryGeneration)
  }

  sizePaletteToDefaultItems(s)
  s.bar?.focus()
  s.bar?.setSelectionRange(s.bar.value.length, s.bar.value.length)
}

function eventTargetsCodeEditor(e: KeyboardEvent) {
  const target = e.target
  if (target instanceof Element && target.closest('.cm-editor')) return true
  const activeElement = document.activeElement
  return activeElement instanceof Element && activeElement.closest('.cm-editor') !== null
}

function moveSelection(s: PaletteState, direction: -1 | 1) {
  const items = [...s.output.querySelectorAll<HTMLElement>('.suggestion-item, .search-load-more')]
  if (!items.length) return
  const currentIndex = s.currentHover ? items.indexOf(s.currentHover) : -1
  s.currentHover?.classList.remove('focus')
  const nextIndex =
    currentIndex === -1
      ? direction === 1
        ? 0
        : items.length - 1
      : (currentIndex + direction + items.length) % items.length
  s.currentHover = items[nextIndex]
  s.currentHover.classList.add('focus')
  s.currentHover.focus()
}

async function shortcutHandler(s: PaletteState, e: HTMLElementEventMap['keydown']) {
  if (e.defaultPrevented) return
  if (eventTargetsCodeEditor(e)) return
  const searchOpen = document.querySelector<HTMLDivElement>('search.search-container')
  const noteContainer = document.getElementById('stacked-notes-container')
  if (
    (searchOpen && searchOpen.classList.contains('active')) ||
    (noteContainer && noteContainer.classList.contains('active'))
  )
    return

  const key = e.key.toLowerCase()
  if (
    s.container.classList.contains('active') &&
    (e.ctrlKey || e.metaKey) &&
    !e.altKey &&
    !e.shiftKey &&
    (key === 'n' || key === 'p')
  ) {
    e.preventDefault()
    moveSelection(s, key === 'n' ? 1 : -1)
    return
  }
  if (key === 'o' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault()
    const barOpen = s.container?.classList.contains('active')
    if (barOpen) {
      hidePalette(s)
    } else {
      showPalette(s, 'quick_open')
    }
    return
  } else if (key === 'p' && e.shiftKey && (e.ctrlKey || e.metaKey)) {
    e.preventDefault()
    const barOpen = s.container?.classList.contains('active')
    if (barOpen) {
      hidePalette(s)
    } else {
      showPalette(s, 'command', 'quick_open')
    }
    return
  } else if (key === 'p' && (e.altKey || e.metaKey) && !e.ctrlKey) {
    e.preventDefault()
    const barOpen = s.container?.classList.contains('active')
    if (barOpen) {
      hidePalette(s)
    } else {
      showPalette(s, 'connector')
    }
    return
  } else if (
    e.key.startsWith('Esc') &&
    s.container?.classList.contains('active') &&
    s.bar.matches(':focus')
  ) {
    // Handle Escape key when input is focused
    e.preventDefault()
    hidePalette(s)
    return
  }

  if (!s.container?.classList.contains('active')) return
  if (
    e.target instanceof HTMLButtonElement &&
    e.target.matches('.image-similar-button, .search-load-more') &&
    (e.key === 'Enter' || e.key === ' ')
  )
    return
  if (
    (e.key === 'Tab' || e.key === 'ArrowRight') &&
    s.currentHover?.querySelector('.image-similar-button')
  ) {
    e.preventDefault()
    const button = s.currentHover.querySelector<HTMLButtonElement>('.image-similar-button')
    if (document.activeElement === button) {
      if (e.shiftKey) s.currentHover.focus()
      else moveSelection(s, 1)
    } else button?.focus()
    return
  }
  if (
    e.key === 'ArrowLeft' &&
    e.target instanceof HTMLButtonElement &&
    e.target.classList.contains('image-similar-button')
  ) {
    e.preventDefault()
    s.currentHover?.focus()
    return
  }

  if (e.metaKey && e.altKey && e.key === 'Enter') {
    if (!s.currentHover) return
    const slug = s.currentHover.dataset.slug
    if (!slug || !isFullSlug(slug)) return

    let request: ReturnType<typeof beginSidePanelRequest> | undefined
    try {
      const asidePanel = getOrCreateSidePanel()
      const activeRequest = beginSidePanelRequest(asidePanel)
      request = activeRequest
      const innerDiv = await fetchContent(s.currentSlug, slug)
      if (!activeRequest.mount(slug, ...innerDiv)) return
      window.notifyNav(slug)
      hidePalette(s)
    } catch (error) {
      const wasAborted = request?.signal.aborted ?? false
      request?.cancel()
      if (!wasAborted) console.error('Failed to create side panel:', error)
    }
    return
  } else if (e.target === s.rankButton) {
    if (e.key === 'Tab') {
      e.preventDefault()
      s.bar.focus()
    }
    return
  } else if (e.key === 'Tab' && e.shiftKey && !s.rankButton.hidden) {
    e.preventDefault()
    s.rankButton.focus()
  } else if (e.key === 'Enter') {
    // If result has focus, navigate to that one, otherwise pick first result
    if (s.output?.contains(s.currentHover)) {
      e.preventDefault()
      s.currentHover!.click()
    } else {
      e.preventDefault()
      s.output.querySelector<HTMLDivElement>('.suggestion-item')?.click()
    }
  } else if (
    paletteMode.actionType === 'command' &&
    (e.key === 'ArrowLeft' || e.key === 'ArrowRight')
  ) {
    if (navigateCommandHistory(s, e.key === 'ArrowLeft' ? -1 : 1)) e.preventDefault()
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    moveSelection(s, -1)
  } else if (e.key === 'Tab') {
    e.preventDefault()
    if (paletteMode.actionType === 'quick_open' && parseCodeQuery(s.bar.value)) {
      moveSelection(s, e.shiftKey ? -1 : 1)
      return
    }
    const focusedElement = s.currentHover
      ? s.currentHover
      : s.output.querySelector<HTMLDivElement>('.suggestion-item.focus')
    paletteMode.currentSearchTerm =
      focusedElement?.querySelector<HTMLDivElement>('.suggestion-title')?.textContent ?? ''
    s.bar.value =
      paletteMode.actionType === 'command'
        ? commandInputValue(paletteMode.currentSearchTerm)
        : paletteMode.currentSearchTerm
    s.queryGeneration += 1
    return await querySearch(s, paletteMode.currentSearchTerm, s.queryGeneration)
  } else if (e.key === 'ArrowDown') {
    e.preventDefault()
    moveSelection(s, 1)
  }
}

async function onType(s: PaletteState, e: HTMLElementEventMap['input']) {
  s.resultPages?.dispose()
  s.resultPages = null
  delete s.output.dataset.imageSeed
  s.imageSimilarity.cancel()
  s.imageSemantic?.cancel()
  s.pdfSearch.cancel()
  s.codeSearch.cancelPending()
  s.codePreview?.abort()
  s.quickOpenSemantic?.cancel()
  s.queryGeneration += 1
  const value = (e.target as HTMLInputElement).value
  const commandMode = value.trimStart().startsWith('>')
  let nextActionType: ActionType
  if (commandMode) {
    if (paletteMode.actionType !== 'command')
      paletteMode.fallbackActionType = paletteMode.actionType
    nextActionType = 'command'
  } else {
    nextActionType =
      paletteMode.actionType === 'command' ? paletteMode.fallbackActionType : paletteMode.actionType
  }
  if (paletteMode.actionType !== nextActionType) {
    paletteMode.actionType = nextActionType
    if (paletteMode.actionType === 'command' && s.commandHistoryIndex === -1) {
      recordCommandLocation(s, { type: 'commands' })
    }
    syncPaletteHelper(s)
  }
  paletteMode.currentSearchTerm =
    paletteMode.actionType === 'command' ? commandSearchTerm(value) : value
  window.clearTimeout(s.quickOpenTimer)
  if (s.completions.update()) {
    s.output.replaceChildren()
    s.output.removeAttribute('aria-busy')
    s.currentHover = null
    quickOpenProgress(s, false)
    return
  }
  if (paletteMode.actionType === 'quick_open' && paletteMode.currentSearchTerm.trim()) {
    const { term: query, scope } = parseSearchQuery(paletteMode.currentSearchTerm)
    const rawQuery = paletteMode.currentSearchTerm
    const generation = s.queryGeneration
    const cached = s.quickOpenSemantic?.cached(query, s.quickOpenSemantic.index.ids.length)
    if (!parseCodeQuery(rawQuery) && query && scope === 'notes' && cached !== undefined) {
      displayQuickOpenResults(s, cached, query, scope)
      return
    }
    s.quickOpenTimer = window.setTimeout(() => {
      void querySearch(s, rawQuery, generation)
    }, 60)
  } else {
    if (paletteMode.actionType !== 'quick_open') {
      quickOpenProgress(s, false)
      s.output.removeAttribute('aria-busy')
    }
    await querySearch(s, paletteMode.currentSearchTerm, s.queryGeneration)
  }
}

document.addEventListener('nav', () => {
  window.addCleanup(disposeSidePanel)
  const signal = currentNavSignal()
  if (activePaletteSignal === signal) return
  activePaletteSignal = signal
  signal.addEventListener(
    'abort',
    () => {
      if (activePaletteSignal === signal) activePaletteSignal = undefined
    },
    { once: true },
  )

  const currentSlug = getFullSlug(window)
  const containerElement = document.getElementById('palette-container')
  if (!containerElement) return
  const container = containerElement
  const imageSimilarity = createImageSimilarity()
  const pdfSearch = new PdfSemantic()
  const codeSearch = new CodeSearchClient()
  window.addCleanup(() => codeSearch.dispose())
  window.addCleanup(() => pdfSearch.dispose())
  window.addCleanup(() => imageSimilarity.dispose())
  window.addCleanup(() => {
    s.imageActive = false
    s.imageSemantic?.dispose()
  })

  const bar = container.querySelector('#bar') as HTMLInputElement
  const output = container.getElementsByTagName('output')[0]
  const helper = container.querySelector('ul#helper') as HTMLUListElement
  const rankButton = document.createElement('a')
  rankButton.href = `${pathToRoot(currentSlug)}/pagerank`
  rankButton.className = 'palette-rank-button'
  rankButton.setAttribute('aria-label', 'PageRank')
  rankButton.innerHTML =
    '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3v18h18M6 7l5 4 4 6 6 2"/><circle cx="6" cy="7" r="1"/><circle cx="11" cy="11" r="1"/><circle cx="15" cy="17" r="1"/></svg>'
  bar.parentElement?.append(rankButton)
  rankButton.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    const destination = new URL(rankButton.href)
    hidePalette(s)
    window.spaNavigate(destination)
  })
  window.addCleanup(() => rankButton.remove())

  window.addCleanup(() => {
    s.isActive = false
    s.resultPages?.dispose()
    imageSimilarity.cancel()
    s.imageSemantic?.cancel()
    s.pdfSearch.cancel()
    s.codeSearch.cancel()
    s.codePreview?.abort()
    s.quickOpenSemantic?.cancel()
  })

  const progress = document.createElement('div')
  progress.className = 'semantic-search-progress'
  progress.setAttribute('role', 'progressbar')
  progress.setAttribute('aria-label', 'Loading search')
  bar.parentElement?.append(progress)
  window.addCleanup(() => progress.remove())
  const s: PaletteState = {
    currentSlug,
    container,
    bar,
    output,
    helper,
    rankButton,
    progress,
    imageSimilarity,
    pdfSearch,
    codeSearch,
    completions: createSearchCompletions(
      bar,
      container,
      () => paletteMode.actionType === 'quick_open',
    ),
    codePreview: null,
    actionCache: new WeakMap<Action, HTMLDivElement>(),
    connectorActs: connectorActions(currentSlug),
    commandActs: [],
    imageIds: undefined,
    imageSemantic: null,
    imageReady: null,
    imageActive: true,
    currentHover: null,
    resultPages: null,
    previousFocus: null,
    data: null,
    isActive: true,
    queryGeneration: 0,
    commandHistory: [],
    commandHistoryIndex: -1,
    contentDataReady: null,
    attachmentDataReady: null,
    quickOpenReady: null,
    quickOpenSemantic: null,
    quickOpenTimer: 0,
    measuredPaletteWidth: 0,
  }
  s.commandActs = commandActions(s)
  quickOpenProgress(s, false)

  const refreshPaletteHeight = () => {
    s.measuredPaletteWidth = 0
    sizePaletteToDefaultItems(s)
  }
  window.addEventListener('resize', refreshPaletteHeight)
  document.fonts.addEventListener('loadingdone', refreshPaletteHeight)
  window.addCleanup(() => {
    window.removeEventListener('resize', refreshPaletteHeight)
    document.fonts.removeEventListener('loadingdone', refreshPaletteHeight)
  })

  const onKeydown = (e: HTMLElementEventMap['keydown']) => shortcutHandler(s, e)
  window.addCleanup(() => s.completions.dispose())
  const onInput = (e: HTMLElementEventMap['input']) => onType(s, e)
  document.addEventListener('keydown', onKeydown)
  bar.addEventListener('input', onInput)
  window.addCleanup(() => {
    document.removeEventListener('keydown', onKeydown)
    bar.removeEventListener('input', onInput)
    window.clearTimeout(s.quickOpenTimer)
  })

  registerEscapeHandler(container, () => hidePalette(s))
  if (currentSlug === 'index') {
    let idle = 0
    let timer = 0
    const preload = () => {
      if (!s.isActive) return
      for (const action of [...s.connectorActs, ...s.commandActs]) {
        if (!s.actionCache.has(action)) s.actionCache.set(action, createActComponent(s, action, ''))
      }
    }
    const schedule = () => {
      if (window.requestIdleCallback) idle = window.requestIdleCallback(preload, { timeout: 2000 })
      else timer = window.setTimeout(preload, 1000)
    }
    if (document.readyState === 'complete') schedule()
    else window.addEventListener('load', schedule, { once: true, signal })
    window.addCleanup(() => {
      window.removeEventListener('load', schedule)
      if (idle) window.cancelIdleCallback(idle)
      window.clearTimeout(timer)
    })
  }
})
