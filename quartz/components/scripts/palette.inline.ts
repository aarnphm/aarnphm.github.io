import { cellSourceLegacyPrefix, cellSourceStoragePrefix } from '../../runtime/notebook/persistence'
import {
  commentRoomToggleEvent,
  readCommentRoomEnabled,
  writeCommentRoomEnabled,
} from '../../util/comment-room'
import { fetchCanonical } from '../../util/fetch-canonical'
import { createNoteRanker, type NoteRanker } from '../../util/note-pagerank'
import {
  notebookKernelCommandEvent,
  notebookKernelRequestEvent,
  notebookKernelRunAllEvent,
  type NotebookKernelCommand,
  type NotebookKernelSnapshot,
  type NotebookKernelStatus,
} from '../../util/notebook-kernel-events'
import { notebookLocalSourcesClearedEvent } from '../../util/notebook-source-events'
import { notebookLanguageIconSvg } from '../../util/notebook/render/icons'
import {
  FullSlug,
  getFullSlug,
  isFullSlug,
  normalizeRelativeURLs,
  pathToRoot,
  resolveRelative,
} from '../../util/path'
import { isSearchExcluded } from '../../util/search-exclude'
import { parseSearchQuery, searchScopeIncludes, type SearchScope } from '../../util/search-scope'
import { tokenizeTerm } from '../../util/search-text'
import { fuseSearchRanks, type SemanticDocument } from '../../util/semantic-search'
import {
  setTriActivityLayout,
  TRI_ACTIVITY_LAYOUTS,
  type TriActivityLayout,
} from '../../util/triathlon-activity-layout'
import { openAvatarEditor } from '../multiplayer/avatar-editor'
import { registerEscapeHandler } from './escape-handler'
import {
  imageSimilarityButton,
  imageSimilarityStatus,
  searchableImage,
} from './image-search-controls'
import { createImageSemantic, type ImageSemantic } from './image-semantic'
import { createImageSimilarity } from './image-similarity'
import { getLexicalSearchClient } from './lexical-search'
import { currentNavSignal } from './nav-lifecycle'
import { noteRankDistribution } from './note-rank-distribution'
import { isSemanticCancellation } from './semantic.inline'
import { beginSidePanelRequest, disposeSidePanel, getOrCreateSidePanel } from './side-panel'
import { siteSemantic, type SiteSemantic } from './site-semantic'

interface Item {
  slug: FullSlug
  name: string
  similarity?: number
}

const numSearchResults = 10
const noteRankers = new WeakMap<ContentIndex, NoteRanker>()

function appendHighlightedText(parent: HTMLElement, searchTerm: string, text: string) {
  const terms = tokenizeTerm(searchTerm)
    .map(term => term.toLowerCase())
    .filter(term => term.length > 0)
    .sort((a, b) => b.length - a.length)
  if (terms.length === 0) {
    parent.append(document.createTextNode(text))
    return
  }

  const lowerText = text.toLowerCase()
  let cursor = 0
  while (cursor < text.length) {
    let nextIndex = -1
    let nextTerm = ''
    for (const term of terms) {
      const index = lowerText.indexOf(term, cursor)
      if (index === -1) continue
      if (
        nextIndex === -1 ||
        index < nextIndex ||
        (index === nextIndex && term.length > nextTerm.length)
      ) {
        nextIndex = index
        nextTerm = term
      }
    }

    if (nextIndex === -1) {
      parent.append(document.createTextNode(text.slice(cursor)))
      return
    }
    if (nextIndex > cursor) {
      parent.append(document.createTextNode(text.slice(cursor, nextIndex)))
    }
    const highlight = document.createElement('span')
    highlight.className = 'highlight'
    highlight.textContent = text.slice(nextIndex, nextIndex + nextTerm.length)
    parent.append(highlight)
    cursor = nextIndex + nextTerm.length
  }
}

function appendActionAux(parent: HTMLElement, auxInnerHtml: string) {
  const action = document.createElement('span')
  action.className = 'suggestion-action'
  const kbdMatch = auxInnerHtml.match(/^<kbd>([^<]+)<\/kbd>(.*)$/)
  const svgMatch = auxInnerHtml.match(
    /^<svg width="1em" height="1em"><use href="([^"]+)" \/><\/svg>$/,
  )
  if (kbdMatch) {
    const key = document.createElement('kbd')
    key.textContent = kbdMatch[1]
    action.appendChild(key)
    action.append(document.createTextNode(kbdMatch[2]))
  } else if (svgMatch) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('width', '1em')
    svg.setAttribute('height', '1em')
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use')
    use.setAttribute('href', svgMatch[1])
    svg.appendChild(use)
    action.appendChild(svg)
  } else {
    action.textContent = auxInnerHtml
  }
  parent.appendChild(action)
}

function appendTrustedSvg(parent: HTMLElement, svgMarkup: string) {
  const template = document.createElement('template')
  template.innerHTML = svgMarkup
  const svg = template.content.firstElementChild
  if (svg) parent.append(svg)
}

const commentAuthorKey = 'comment-author'
const commentAuthorSourceKey = 'comment-author-source'
const commentAuthorLastRenameKey = 'comment-author-last-rename'
const commentAuthorGithubLoginKey = 'comment-author-github-login'
const commentAuthorRenameWindowMs = 1000 * 60 * 60 * 24 * 90

function notifyToast(message: string) {
  document.dispatchEvent(new CustomEvent('toast', { detail: { message } }))
}

function notebookKernelSnapshots(): NotebookKernelSnapshot[] {
  const snapshots: NotebookKernelSnapshot[] = []
  const event: CustomEventMap['notebookkernelrequest'] = new CustomEvent(
    notebookKernelRequestEvent,
    { detail: { respond: snapshot => snapshots.push(snapshot) } },
  )
  document.dispatchEvent(event)
  return snapshots
}

function notebookKernelSourceLabel(sourcePath: string): string {
  const name = sourcePath.split('/').filter(Boolean).at(-1)
  return name ?? sourcePath
}

function notebookKernelStatusLabel(snapshot: NotebookKernelSnapshot): string | undefined {
  if (snapshot.status === 'ready') return undefined
  const detail = snapshot.status === 'running' ? snapshot.runningCellId : snapshot.statusDetail
  if (detail) {
    return `${snapshot.status} ${detail}`
  }
  return snapshot.status
}

function notebookKernelLanguageToken(language: string): string {
  const normalized = language.toLowerCase()
  if (normalized === 'python') return 'py'
  if (normalized === 'javascript') return 'js'
  if (normalized === 'typescript') return 'ts'
  if (normalized === 'haskell') return 'hs'
  if (normalized === 'rust') return 'rs'
  if (normalized === 'ocaml') return 'ml'
  if (normalized === 'bash') return 'sh'
  return normalized
}

function notebookKernelLanguageIcon(language: string): string {
  const normalized = language.toLowerCase()
  if (Object.hasOwn(notebookLanguageIconSvg, normalized)) return notebookLanguageIconSvg[normalized]
  return notebookLanguageIconSvg.text
}

function dispatchNotebookKernelCommand(
  snapshot: NotebookKernelSnapshot,
  command: NotebookKernelCommand,
) {
  const event: CustomEventMap['notebookkernelcommand'] = new CustomEvent(
    notebookKernelCommandEvent,
    { detail: { runtimeId: snapshot.runtimeId, language: snapshot.language, command } },
  )
  document.dispatchEvent(event)
}

function dispatchNotebookKernelRunAll() {
  const event: CustomEventMap['notebookkernelrunall'] = new CustomEvent(notebookKernelRunAllEvent, {
    detail: {},
  })
  document.dispatchEvent(event)
}

function clearNotebookLocalSources(storage: Storage = localStorage): number {
  const keys: string[] = []
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i)
    if (
      key &&
      (key.startsWith(cellSourceStoragePrefix) || key.startsWith(cellSourceLegacyPrefix))
    ) {
      keys.push(key)
    }
  }
  for (const key of keys) storage.removeItem(key)
  return keys.length
}

function dispatchCommentAuthorUpdated(oldAuthor: string, newAuthor: string) {
  document.dispatchEvent(
    new CustomEvent('commentauthorupdated', { detail: { oldAuthor, newAuthor } }),
  )
}

function getLastRenameTime(): number | null {
  const raw = localStorage.getItem(commentAuthorLastRenameKey)
  if (!raw) return null
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) ? parsed : null
}

function isRenameWindowActive() {
  const last = getLastRenameTime()
  if (last === null) return false
  return Date.now() - last < commentAuthorRenameWindowMs
}

async function requestCommentAuthorRename(oldAuthor: string, newAuthor: string): Promise<boolean> {
  const githubLogin = localStorage.getItem(commentAuthorGithubLoginKey)
  const payload: { oldAuthor: string; newAuthor: string; githubLogin?: string } = {
    oldAuthor,
    newAuthor,
  }
  if (githubLogin) {
    payload.githubLogin = githubLogin
  }
  try {
    const response = await fetch('/comments/author/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (response.ok) return true
    if (response.status === 429) {
      notifyToast('comment name can change every 3 months')
      return false
    }
    const text = await response.text()
    notifyToast(text || 'failed to update comment author')
  } catch {
    notifyToast('failed to update comment author')
  }
  return false
}

async function updateCommentAuthor(author: string, source: 'manual' | 'github') {
  const next = author.trim()
  if (!next) return
  const existing = localStorage.getItem(commentAuthorKey) ?? ''
  if (!existing) {
    localStorage.setItem(commentAuthorKey, next)
    localStorage.setItem(commentAuthorSourceKey, source)
    notifyToast(`name set to ${next}`)
    return
  }

  if (existing === next) {
    localStorage.setItem(commentAuthorSourceKey, source)
    notifyToast(`name set to ${next}`)
    return
  }

  if (isRenameWindowActive()) {
    notifyToast('name can change every 3 months')
    return
  }

  const renamed = await requestCommentAuthorRename(existing, next)
  if (!renamed) return

  localStorage.setItem(commentAuthorLastRenameKey, `${Date.now()}`)
  localStorage.setItem(commentAuthorKey, next)
  localStorage.setItem(commentAuthorSourceKey, source)
  dispatchCommentAuthorUpdated(existing, next)
  notifyToast(`name set to ${next}`)
}

function promptForCommentAuthor() {
  const existing = localStorage.getItem(commentAuthorKey) ?? ''
  const hint = 'your name picks your blob; change it with commenter avatar'
  const promptText = existing
    ? `current comment name: ${existing}\nset comment name (${hint})`
    : `set comment name (${hint})`
  const raw = window.prompt(promptText, existing)
  if (raw === null) return
  void updateCommentAuthor(raw, 'manual')
}

function startGithubCommentLogin(returnTo: string) {
  const target = new URL('/comments/github/login', window.location.origin)
  target.searchParams.set('returnTo', returnTo)
  const existing = localStorage.getItem(commentAuthorKey)
  if (existing) {
    target.searchParams.set('author', existing)
  }
  window.location.assign(target.toString())
}

const p = new DOMParser()
const fetchContentCache: Map<FullSlug, HTMLElement[]> = new Map()
async function fetchContent(currentSlug: FullSlug, slug: FullSlug): Promise<HTMLElement[]> {
  if (fetchContentCache.has(slug)) {
    return fetchContentCache.get(slug) as HTMLElement[]
  }

  const targetUrl = new URL(resolveRelative(currentSlug, slug), location.toString())
  const contents = await fetchCanonical(targetUrl)
    .then(res => res.text())
    .then(contents => {
      if (contents === undefined) {
        throw new Error(`Could not fetch ${targetUrl}`)
      }
      const html = p.parseFromString(contents ?? '', 'text/html')
      normalizeRelativeURLs(html, targetUrl)
      return [...html.getElementsByClassName('popover-hint')] as HTMLElement[]
    })

  fetchContentCache.set(slug, contents)
  return contents
}

type ActionType = 'quick_open' | 'connector' | 'command'
type FallbackActionType = Exclude<ActionType, 'command'>
type CommandLocation =
  | { readonly type: 'commands' }
  | { readonly type: 'kernels' }
  | { readonly type: 'kernel-actions'; readonly snapshot: NotebookKernelSnapshot }
interface Action {
  name: string
  onClick: (e: MouseEvent) => void
  auxInnerHtml: string
  keepOpen?: boolean
  detail?: string
  titlePrefixInnerHtml?: string
  statusLabel?: string
  statusTone?: NotebookKernelStatus
}

let actionType: ActionType = 'quick_open'
let fallbackActionType: FallbackActionType = 'quick_open'
let currentSearchTerm: string = ''
let activePaletteSignal: AbortSignal | undefined

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
  let imageIds: ReadonlySet<string> | undefined
  window.addCleanup(() => imageSimilarity.dispose())
  let imageSemantic: ImageSemantic | null = null
  let imageReady: Promise<ImageSemantic> | null = null
  let imageActive = true
  function ensureImageSearch() {
    imageReady ??= createImageSemantic()
      .then(service => {
        if (!imageActive) {
          service.dispose()
          throw new Error('image search page closed')
        }
        imageSemantic = service
        return service
      })
      .catch(error => {
        imageReady = null
        throw error
      })
    return imageReady
  }
  window.addCleanup(() => {
    imageActive = false
    imageSemantic?.dispose()
  })

  const bar = container.querySelector('#bar') as HTMLInputElement
  const output = container.getElementsByTagName('output')[0]
  const helper = container.querySelector('ul#helper') as HTMLUListElement
  const rankButton = document.createElement('a')
  rankButton.href = `${pathToRoot(currentSlug)}/pagerank`
  rankButton.className = 'palette-rank-button'
  rankButton.textContent = 'pagerank ↗'
  rankButton.title = 'Global note ranking and link graph'
  bar.parentElement?.append(rankButton)
  rankButton.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    const destination = new URL(rankButton.href)
    hidePalette()
    window.spaNavigate(destination)
  })
  window.addCleanup(() => rankButton.remove())
  let currentHover: HTMLDivElement | null = null
  let previousFocus: HTMLElement | null = null
  let data: ContentIndex | null = null
  let isActive = true
  let queryGeneration = 0
  let commandHistory: CommandLocation[] = []
  let commandHistoryIndex = -1

  window.addCleanup(() => {
    isActive = false
    imageSimilarity.cancel()
    imageSemantic?.cancel()
    quickOpenSemantic?.cancel()
  })

  let contentDataReady: Promise<ContentIndex> | null = null
  let attachmentDataReady: Promise<ContentIndex> | null = null
  let quickOpenReady: ReturnType<typeof siteSemantic> | null = null
  let quickOpenSemantic: SiteSemantic | null = null
  let quickOpenTimer = 0
  const progress = document.createElement('div')
  progress.className = 'semantic-search-progress'
  progress.setAttribute('role', 'progressbar')
  progress.setAttribute('aria-label', 'Loading search')
  bar.parentElement?.append(progress)
  window.addCleanup(() => progress.remove())
  function quickOpenProgress(loading: boolean) {
    progress.setAttribute('aria-hidden', String(!loading))
    progress.style.opacity = loading ? '1' : '0'
    progress.style.width = loading ? '100%' : '0'
    progress.style.animation = loading ? 'semantic-progress-sweep 2.5s linear infinite' : 'none'
  }
  quickOpenProgress(false)

  function ensureContentData() {
    contentDataReady ??= fetchData.then(resolved => {
      if (isActive) data = resolved
      return resolved
    })
    return contentDataReady
  }

  function ensureAttachmentData() {
    attachmentDataReady ??= (typeof fetchSearchData === 'undefined' ? fetchData : fetchSearchData)
      .catch(error => {
        console.warn('[QuickOpen] attachment index unavailable:', error)
        return fetchData
      })
      .then(resolved => {
        if (isActive) data = resolved
        return resolved
      })
    return attachmentDataReady
  }

  function ensureQuickOpen() {
    quickOpenReady ??= Promise.all([ensureContentData(), siteSemantic()]).then(
      ([, service]) => {
        quickOpenSemantic = service
        return service
      },
      error => {
        throw error
      },
    )
    return quickOpenReady
  }

  function quickOpenStatus(message: string) {
    currentHover = null
    if (!message) {
      output.replaceChildren()
      return
    }
    const status = document.createElement('p')
    status.setAttribute('role', 'status')
    status.textContent = message
    output.replaceChildren(status)
  }

  function hidePalette() {
    imageSimilarity.cancel()
    imageSemantic?.cancel()
    quickOpenSemantic?.cancel()
    queryGeneration += 1
    window.clearTimeout(quickOpenTimer)
    output.removeAttribute('aria-busy')
    quickOpenProgress(false)
    container?.classList.remove('active')
    if (container?.matches(':popover-open')) container.hidePopover()
    if (bar) {
      bar.value = '' // clear the input when we dismiss the search
    }
    if (output) {
      output.replaceChildren()
    }

    actionType = 'quick_open' // reset search type after closing
    commandHistory = []
    commandHistoryIndex = -1
    syncPaletteHelper()
    if (previousFocus?.isConnected) previousFocus.focus({ preventScroll: true })
    previousFocus = null
  }

  function commandInputValue(query = '') {
    return query.length === 0 ? '> ' : `> ${query}`
  }

  function commandSearchTerm(value: string) {
    return value.replace(/^\s*>\s?/, '').trimStart()
  }

  function syncPaletteHelper() {
    bar.placeholder =
      actionType === 'quick_open'
        ? 'rechercher des notes… filter: images · pdf · code · py · all'
        : 'sélectionnez une option…'
    bar.setAttribute('aria-label', bar.placeholder)
    helper.querySelectorAll<HTMLLIElement>('li[data-quick-open]').forEach(el => {
      el.style.display = actionType === 'quick_open' ? '' : 'none'
    })
    helper.querySelectorAll<HTMLLIElement>('li[data-command]').forEach(el => {
      el.style.display = actionType === 'command' ? '' : 'none'
    })
  }

  let measuredPaletteWidth = 0
  function sizePaletteToDefaultItems() {
    const panel = output.parentElement
    if (!panel || !container.matches(':popover-open')) return
    const width = panel.clientWidth
    if (width === measuredPaletteWidth) return
    const measurement = document.createElement('output')
    measurement.setAttribute('aria-hidden', 'true')
    measurement.inert = true
    measurement.style.cssText = `position:absolute;visibility:hidden;pointer-events:none;flex:none;height:auto;max-height:none;overflow:visible;box-sizing:border-box;width:${width}px`
    panel.append(measurement)
    let itemsHeight = 0
    for (const actions of [CONNECTOR_ACTS, COMMAND_ACTS]) {
      measurement.replaceChildren(
        ...actions.map(action =>
          (actionCache.get(action) ?? actionComponent(action)).cloneNode(true),
        ),
      )
      itemsHeight = Math.max(itemsHeight, measurement.getBoundingClientRect().height)
    }
    const inputHeight = bar.parentElement?.getBoundingClientRect().height ?? 0
    const height =
      itemsHeight +
      inputHeight +
      helper.getBoundingClientRect().height +
      panel.offsetHeight -
      panel.clientHeight
    measurement.remove()
    panel.style.setProperty('--palette-content-height', `${Math.ceil(height)}px`)
    measuredPaletteWidth = width
  }
  const refreshPaletteHeight = () => {
    measuredPaletteWidth = 0
    sizePaletteToDefaultItems()
  }
  window.addEventListener('resize', refreshPaletteHeight)
  document.fonts.addEventListener('loadingdone', refreshPaletteHeight)
  window.addCleanup(() => {
    window.removeEventListener('resize', refreshPaletteHeight)
    document.fonts.removeEventListener('loadingdone', refreshPaletteHeight)
  })

  function showPalette(actionTypeNew: ActionType, fallbackActionTypeNew?: FallbackActionType) {
    queryGeneration += 1
    actionType = actionTypeNew
    fallbackActionType =
      actionTypeNew === 'command' ? (fallbackActionTypeNew ?? fallbackActionType) : actionTypeNew
    if (container && !container.matches(':popover-open')) {
      previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
      container.showPopover()
    }
    container?.classList.add('active')
    if (actionType === 'command') {
      showCommandItems()
    } else if (actionType === 'connector') {
      bar.value = ''
      currentSearchTerm = ''
      syncPaletteHelper()
      getCommandItems(CONNECTOR_ACTS)
    } else if (actionType === 'quick_open') {
      bar.value = ''
      currentSearchTerm = ''
      syncPaletteHelper()
      quickOpenStatus('')
      quickOpenProgress(false)
      void displayNearbyNotes(queryGeneration)
    }

    sizePaletteToDefaultItems()
    bar?.focus()
    bar?.setSelectionRange(bar.value.length, bar.value.length)
  }

  const CONNECTOR_ACTS: Action[] = [
    {
      name: 'x.com (formerly Twitter)',
      auxInnerHtml: `<svg width="1em" height="1em"><use href="#twitter-icon" /></svg>`,
      onClick: () => {
        window.location.href = 'https://x.com/aarnphm'
      },
    },
    {
      name: 'bsky.app',
      auxInnerHtml: `<svg width="1em" height="1em"><use href="#bsky-icon" /></svg>`,
      onClick: () => {
        window.location.href = 'https://bsky.app/profile/aarnphm.xyz'
      },
    },
    {
      name: 'substack',
      auxInnerHtml: `<svg width="1em" height="1em"><use href="#substack-icon" /></svg>`,
      onClick: () => {
        window.location.href = 'https://livingalonealone.com'
      },
    },
    {
      name: 'github',
      auxInnerHtml: `<svg width="1em" height="1em"><use href="#github-icon" /></svg>`,
      onClick: () => {
        window.location.href = 'https://github.com/aarnphm'
      },
    },
    {
      name: 'comments room',
      auxInnerHtml: '<kbd>↵</kbd> toggle on/off',
      onClick: () => {
        const enabled = !readCommentRoomEnabled()
        writeCommentRoomEnabled(enabled)
        const event: CustomEventMap['commentsroomtoggle'] = new CustomEvent(
          commentRoomToggleEvent,
          { detail: { enabled } },
        )
        document.dispatchEvent(event)
        notifyToast(`comments ${enabled ? 'on' : 'off'}`)
      },
    },
    {
      name: 'commenter name',
      auxInnerHtml: '<kbd>↵</kbd> set comment handle',
      onClick: () => {
        promptForCommentAuthor()
      },
    },
    {
      name: 'commenter avatar',
      auxInnerHtml: '<kbd>↵</kbd> change your blob',
      onClick: () => {
        openAvatarEditor()
      },
    },
    {
      name: 'commenter login with github',
      auxInnerHtml: '<kbd>↵</kbd> verify via github',
      onClick: () => {
        startGithubCommentLogin(window.location.toString())
      },
    },
    {
      name: 'pets',
      auxInnerHtml: '<kbd>↵</kbd> toggle on/off',
      onClick: () => {
        const event: CustomEventMap['petstoggle'] = new CustomEvent('petstoggle', { detail: {} })
        document.dispatchEvent(event)
      },
    },
    {
      name: 'are.na',
      auxInnerHtml: '<kbd>↵</kbd> a rundown version of are.na',
      onClick: () => {
        window.spaNavigate(
          new URL(resolveRelative(currentSlug, '/arena' as FullSlug), window.location.toString()),
        )
      },
    },
    {
      name: 'stream',
      auxInnerHtml: '<kbd>↵</kbd> microblog',
      onClick: () => {
        window.spaNavigate(
          new URL(resolveRelative(currentSlug, '/stream' as FullSlug), window.location.toString()),
        )
      },
    },
    {
      name: 'triathlon',
      auxInnerHtml: '<kbd>↵</kbd> training',
      onClick: () => {
        window.spaNavigate(
          new URL(
            resolveRelative(currentSlug, '/triathlon' as FullSlug),
            window.location.toString(),
          ),
        )
      },
    },
    {
      name: 'friends',
      auxInnerHtml: '<kbd>↵</kbd> as virtuosic',
      onClick: () => {
        window.spaNavigate(
          new URL(resolveRelative(currentSlug, '/friends' as FullSlug), window.location.toString()),
        )
      },
    },
    {
      name: 'coffee chat',
      auxInnerHtml: '<kbd>↵</kbd> on calendly',
      onClick: () => {
        window.location.href = 'https://calendly.com/aarnphm/30min'
      },
    },
    {
      name: 'work',
      auxInnerHtml: '<kbd>↵</kbd> as craft',
      onClick: () => {
        window.spaNavigate(
          new URL(
            resolveRelative(currentSlug, '/thoughts/craft' as FullSlug),
            window.location.toString(),
          ),
        )
      },
    },
    {
      name: 'cool people',
      auxInnerHtml: '<kbd>↵</kbd> as inspiration',
      onClick: () => {
        window.spaNavigate(
          new URL(
            resolveRelative(currentSlug, '/influence' as FullSlug),
            window.location.toString(),
          ),
        )
      },
    },
    {
      name: 'old fashioned resume (maybe not up-to-date)',
      auxInnerHtml: '<kbd>↵</kbd>',
      onClick: () => {
        window.spaNavigate(
          new URL(
            resolveRelative(currentSlug, '/thoughts/pdfs/2025q1-resume.pdf' as FullSlug),
            window.location.toString(),
          ),
        )
      },
    },
  ]

  const TRI_ACTIVITY_LAYOUT_LABELS: Record<TriActivityLayout, string> = {
    one: 'one activity per view',
    two: 'two activities per view',
    all: 'all activities',
  }
  const pressKit = (tab: 'note' | 'quote' | 'links') => () =>
    document.dispatchEvent(new CustomEvent('presskitopen', { detail: { tab } }))
  const COMMAND_ACTS: Action[] = [
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
        showKernelItems()
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
        showCommandItems(true)
      },
    },
  ]

  function createStatusComponent(tone: NotebookKernelStatus, label: string | undefined) {
    const status = document.createElement('span')
    status.className = 'suggestion-status'
    status.dataset.statusTone = tone
    status.setAttribute('aria-label', label ? `${tone}: ${label}` : tone)
    if (label) status.textContent = label
    return status
  }

  const actionCache = new WeakMap<Action, HTMLDivElement>()
  const createActComponent = (
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
    query = currentSearchTerm,
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
      if (keepOpen !== true) hidePalette()
    }
    item.addEventListener('click', mainOnClick)
    window.addCleanup(() => item.removeEventListener('click', mainOnClick))
    return item
  }

  function actionComponent(action: Action) {
    if (currentSearchTerm) return createActComponent(action)
    let item = actionCache.get(action)
    if (!item) {
      item = createActComponent(action, '')
      actionCache.set(action, item)
    }
    item.classList.remove('focus')
    return item
  }

  function transitionCommandItems() {
    output.removeAttribute('data-palette-transition')
    void output.offsetHeight
    output.dataset.paletteTransition = ''
    window.setTimeout(() => {
      output.removeAttribute('data-palette-transition')
    }, 220)
  }

  function getCommandItems(acts: Action[], transition = false) {
    if (output) {
      output.replaceChildren()
    }
    currentHover = null
    if (acts.length === 0) {
      if (bar.matches(':focus') && currentSearchTerm === '') {
        output.append(
          ...(actionType === 'command' ? COMMAND_ACTS : CONNECTOR_ACTS).map(actionComponent),
        )
      } else {
        const fallbackAction = actionType === 'command' ? COMMAND_ACTS[0] : CONNECTOR_ACTS[0]
        if (fallbackAction) output.append(actionComponent(fallbackAction))
      }
    } else {
      output.append(...acts.map(actionComponent))
    }
    setFocusFirstChild()
    if (transition) transitionCommandItems()
  }

  function recordCommandLocation(location: CommandLocation) {
    commandHistory = commandHistory.slice(0, commandHistoryIndex + 1)
    commandHistory.push(location)
    commandHistoryIndex = commandHistory.length - 1
  }

  function renderCommandLocation(location: CommandLocation, transition = false) {
    queryGeneration += 1
    actionType = 'command'
    syncPaletteHelper()
    if (location.type === 'commands') {
      bar.value = commandInputValue()
      currentSearchTerm = ''
      getCommandItems(COMMAND_ACTS, transition)
      return
    }
    if (location.type === 'kernels') {
      renderKernelItems(transition)
      return
    }
    renderKernelActionItems(location.snapshot, transition)
  }

  function showCommandLocation(location: CommandLocation, transition = false, record = true) {
    if (record) recordCommandLocation(location)
    renderCommandLocation(location, transition)
  }

  function showCommandItems(transition = false, record = true) {
    showCommandLocation({ type: 'commands' }, transition, record)
  }

  function showKernelItems(transition = true, record = true) {
    showCommandLocation({ type: 'kernels' }, transition, record)
  }

  function renderKernelItems(transition = true) {
    bar.value = commandInputValue('kernels')
    currentSearchTerm = ''
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
            onClick: () => showKernelActionItems(snapshot),
          }))
    getCommandItems(acts, transition)
  }

  function showKernelActionItems(
    snapshot: NotebookKernelSnapshot,
    transition = true,
    record = true,
  ) {
    showCommandLocation({ type: 'kernel-actions', snapshot }, transition, record)
  }

  function renderKernelActionItems(snapshot: NotebookKernelSnapshot, transition = true) {
    bar.value = commandInputValue(`kernels ${snapshot.language}`)
    currentSearchTerm = ''
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
        window.setTimeout(() => showKernelItems(true, false), 140)
      },
    })
    getCommandItems([action('interrupt'), action('restart'), action('kill')], transition)
  }

  function navigateCommandHistory(delta: -1 | 1): boolean {
    if (actionType !== 'command') return false
    const nextIndex = commandHistoryIndex + delta
    if (nextIndex < 0 || nextIndex >= commandHistory.length) return false
    commandHistoryIndex = nextIndex
    renderCommandLocation(commandHistory[commandHistoryIndex], true)
    return true
  }

  function eventTargetsCodeEditor(e: KeyboardEvent) {
    const target = e.target
    if (target instanceof Element && target.closest('.cm-editor')) return true
    const activeElement = document.activeElement
    return activeElement instanceof Element && activeElement.closest('.cm-editor') !== null
  }

  function moveSelection(direction: -1 | 1) {
    const items = [...output.querySelectorAll<HTMLDivElement>('.suggestion-item')]
    if (!items.length) return
    const currentIndex = currentHover ? items.indexOf(currentHover) : -1
    currentHover?.classList.remove('focus')
    const nextIndex =
      currentIndex === -1
        ? direction === 1
          ? 0
          : items.length - 1
        : (currentIndex + direction + items.length) % items.length
    currentHover = items[nextIndex]
    currentHover.classList.add('focus')
    currentHover.focus()
  }

  async function shortcutHandler(e: HTMLElementEventMap['keydown']) {
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
      container.classList.contains('active') &&
      (e.ctrlKey || e.metaKey) &&
      !e.altKey &&
      !e.shiftKey &&
      (key === 'n' || key === 'p')
    ) {
      e.preventDefault()
      moveSelection(key === 'n' ? 1 : -1)
      return
    }
    if (key === 'o' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const barOpen = container?.classList.contains('active')
      if (barOpen) {
        hidePalette()
      } else {
        showPalette('quick_open')
      }
      return
    } else if (key === 'p' && e.shiftKey && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const barOpen = container?.classList.contains('active')
      if (barOpen) {
        hidePalette()
      } else {
        showPalette('command', 'quick_open')
      }
      return
    } else if (key === 'p' && (e.altKey || e.metaKey) && !e.ctrlKey) {
      e.preventDefault()
      const barOpen = container?.classList.contains('active')
      if (barOpen) {
        hidePalette()
      } else {
        showPalette('connector')
      }
      return
    } else if (
      e.key.startsWith('Esc') &&
      container?.classList.contains('active') &&
      bar.matches(':focus')
    ) {
      // Handle Escape key when input is focused
      e.preventDefault()
      hidePalette()
      return
    }

    if (!container?.classList.contains('active')) return
    if (
      e.target instanceof HTMLButtonElement &&
      e.target.classList.contains('image-similar-button') &&
      (e.key === 'Enter' || e.key === ' ')
    )
      return
    if (
      (e.key === 'Tab' || e.key === 'ArrowRight') &&
      currentHover?.querySelector('.image-similar-button')
    ) {
      e.preventDefault()
      const button = currentHover.querySelector<HTMLButtonElement>('.image-similar-button')
      if (document.activeElement === button) {
        if (e.shiftKey) currentHover.focus()
        else moveSelection(1)
      } else button?.focus()
      return
    }
    if (
      e.key === 'ArrowLeft' &&
      e.target instanceof HTMLButtonElement &&
      e.target.classList.contains('image-similar-button')
    ) {
      e.preventDefault()
      currentHover?.focus()
      return
    }

    if (e.metaKey && e.altKey && e.key === 'Enter') {
      if (!currentHover) return
      const slug = currentHover.dataset.slug
      if (!slug || !isFullSlug(slug)) return

      let request: ReturnType<typeof beginSidePanelRequest> | undefined
      try {
        const asidePanel = getOrCreateSidePanel()
        const activeRequest = beginSidePanelRequest(asidePanel)
        request = activeRequest
        const innerDiv = await fetchContent(currentSlug, slug)
        if (!activeRequest.mount(slug, ...innerDiv)) return
        window.notifyNav(slug)
        hidePalette()
      } catch (error) {
        const wasAborted = request?.signal.aborted ?? false
        request?.cancel()
        if (!wasAborted) console.error('Failed to create side panel:', error)
      }
      return
    } else if (e.target === rankButton) {
      if (e.key === 'Tab') {
        e.preventDefault()
        bar.focus()
      }
      return
    } else if (e.key === 'Tab' && e.shiftKey) {
      e.preventDefault()
      rankButton.focus()
    } else if (e.key === 'Enter') {
      // If result has focus, navigate to that one, otherwise pick first result
      if (output?.contains(currentHover)) {
        e.preventDefault()
        currentHover!.click()
      } else {
        e.preventDefault()
        output.querySelector<HTMLDivElement>('.suggestion-item')?.click()
      }
    } else if (actionType === 'command' && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      if (navigateCommandHistory(e.key === 'ArrowLeft' ? -1 : 1)) e.preventDefault()
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      moveSelection(-1)
    } else if (e.key === 'Tab') {
      e.preventDefault()
      const focusedElement = currentHover
        ? currentHover
        : output.querySelector<HTMLDivElement>('.suggestion-item.focus')
      currentSearchTerm =
        focusedElement?.querySelector<HTMLDivElement>('.suggestion-title')?.textContent ?? ''
      bar.value =
        actionType === 'command' ? commandInputValue(currentSearchTerm) : currentSearchTerm
      queryGeneration += 1
      return await querySearch(currentSearchTerm, queryGeneration)
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      moveSelection(1)
    }
  }

  async function displayNearbyNotes(generation: number) {
    try {
      const metadata = await ensureContentData()
      if (generation !== queryGeneration || actionType !== 'quick_open' || !isActive) return
      let ranker = noteRankers.get(metadata)
      if (!ranker) {
        ranker = createNoteRanker(metadata)
        noteRankers.set(metadata, ranker)
      }
      const items = ranker
        .related(currentSlug, numSearchResults)
        .map(slug => ({ slug, name: metadata[slug].title || metadata[slug].fileName }))
      output.removeAttribute('aria-busy')
      quickOpenProgress(false)
      if (items.length) displayResults(items, '')
      else quickOpenStatus('')
    } catch (error) {
      if (generation !== queryGeneration || actionType !== 'quick_open' || !isActive) return
      console.warn('[QuickOpen] nearby notes unavailable:', error)
      output.removeAttribute('aria-busy')
      quickOpenProgress(false)
      quickOpenStatus('Nearby notes are unavailable.')
    }
  }

  async function querySearch(currentSearchTerm: string, generation = queryGeneration) {
    const queryActionType = actionType
    if (actionType === 'quick_open') {
      const { term: query, scope, explicit } = parseSearchQuery(currentSearchTerm)
      if (!query && (!explicit || scope === 'notes')) {
        await displayNearbyNotes(generation)
        return
      }
      output.setAttribute('aria-busy', 'true')
      quickOpenProgress(true)
      let scopedDocuments: SemanticDocument[] = []
      let settled = false
      try {
        if (scope !== 'notes' || !query) {
          const content = await ensureContentData()
          const loaded = scope === 'notes' ? content : await ensureAttachmentData()
          if (generation !== queryGeneration || queryActionType !== actionType || !isActive) return
          if (scope === 'images') {
            const ids = await imageSimilarity.ids().catch(() => undefined)
            if (generation !== queryGeneration || !isActive) return
            imageIds = ids
          }
          const slugs = Object.entries(loaded)
            .filter(
              ([slug, source]) => !isSearchExcluded(slug) && !isSearchExcluded(source.fileName),
            )
            .map(([slug]) => slug)
          const hits = query
            ? await getLexicalSearchClient().search(loaded, {
                query,
                scope,
                limit: scope === 'images' ? slugs.length : numSearchResults,
                index: ['title', 'fileName', 'aliases'],
              })
            : [{ result: slugs.map((_, id) => id) }]
          if (generation !== queryGeneration || !isActive) return
          scopedDocuments = [...new Set(hits.flatMap(hit => hit.result))].flatMap(id => {
            const slug = slugs[id]
            return slug ? [{ slug, score: Number.NaN }] : []
          })
          displayQuickOpenResults(scopedDocuments, query, scope)
          if (scope === 'images' && query) {
            quickOpenProgress(true)
            output.setAttribute('aria-busy', 'true')
            const service = await ensureImageSearch()
            if (generation !== queryGeneration || !isActive) return
            const documents = await service.search(query, numSearchResults * 8)
            if (generation !== queryGeneration || queryActionType !== actionType || !isActive)
              return
            const scores = new Map(documents.map(document => [document.slug, document.score]))
            const combined = fuseSearchRanks([
              { ids: scopedDocuments.map(document => document.slug), weight: 1 },
              { ids: documents.map(document => document.slug), weight: 1 },
            ]).map(({ id }) => ({ slug: id, score: scores.get(id) ?? Number.NaN }))
            displayQuickOpenResults(combined, query, scope)
            return
          }
          if ((scope !== 'notes' && scope !== 'all') || !query) return
          output.setAttribute('aria-busy', 'true')
          quickOpenProgress(true)
        }
        void ensureContentData().then(
          () => {
            if (settled || generation !== queryGeneration || !isActive) return
            displayPendingTitles(query, undefined, scope)
          },
          () => {},
        )
        const service = await ensureQuickOpen()
        if (generation !== queryGeneration || queryActionType !== actionType || !isActive) return
        const documents = await service.search(query, numSearchResults * 8)
        if (generation !== queryGeneration || queryActionType !== actionType || !isActive) return
        const scores = new Map(documents.map(document => [document.slug, document.score]))
        const combined = fuseSearchRanks([
          { ids: scopedDocuments.map(document => document.slug), weight: 1 },
          { ids: documents.map(document => document.slug), weight: 1 },
        ]).map(({ id }) => ({ slug: id, score: scores.get(id) ?? Number.NaN }))
        displayQuickOpenResults(combined, query, scope)
      } catch (error) {
        if (isSemanticCancellation(error)) return
        if (generation !== queryGeneration || queryActionType !== actionType || !isActive) return
        console.warn('[QuickOpen] semantic search failed:', error)
        output.removeAttribute('aria-busy')
        quickOpenProgress(false)
        if (scope === 'images') {
          imageSemantic?.dispose()
          imageSemantic = null
          imageReady = null
          displayQuickOpenResults(scopedDocuments, query, scope)
          const status = document.createElement('p')
          status.setAttribute('role', 'status')
          status.textContent = 'Visual search unavailable. Showing image metadata matches.'
          output.prepend(status)
        } else if (scopedDocuments.length) displayQuickOpenResults(scopedDocuments, query, scope)
        else
          displayPendingTitles(
            query,
            'semantic unavailable; use ctrl+k for full-text search',
            scope,
          )
      } finally {
        settled = true
      }
    } else {
      const query = currentSearchTerm.toLowerCase().trim()
      const actions = actionType === 'command' ? COMMAND_ACTS : CONNECTOR_ACTS
      const matchedActions = query
        ? actions.filter(
            action =>
              action.name.toLowerCase().includes(query) ||
              action.detail?.toLowerCase().includes(query) ||
              action.auxInnerHtml.toLowerCase().includes(query),
          )
        : actions

      getCommandItems(matchedActions)
    }
  }

  function displayPendingTitles(query: string, message?: string, scope: SearchScope = 'notes') {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
    const items = Object.entries(data ?? {})
      .flatMap(([slug, note]) => {
        if (
          !isFullSlug(slug) ||
          note.protected ||
          isSearchExcluded(slug) ||
          isSearchExcluded(note.fileName) ||
          !searchScopeIncludes(scope, { ...note, slug })
        )
          return []
        const title = note.title || note.fileName
        const text = [title, note.fileName, ...(note.aliases ?? [])].join(' ').toLowerCase()
        return terms.every(term => text.includes(term)) ? [{ slug, name: title }] : []
      })
      .sort((a, b) => {
        const match = (item: Item) => Number(item.name.toLowerCase() === query.toLowerCase())
        return match(b) - match(a)
      })
      .slice(0, numSearchResults)
    if (items.length) displayResults(items, query)
    else output.replaceChildren()
    if (message) {
      const status = document.createElement('p')
      status.setAttribute('role', 'status')
      status.textContent = items.length ? `${message}; title matches shown` : message
      output.prepend(status)
    } else {
      output.setAttribute('aria-busy', 'true')
    }
  }

  async function showSimilarImages(seed: string) {
    const generation = ++queryGeneration
    imageSemantic?.cancel()
    quickOpenSemantic?.cancel()
    window.clearTimeout(quickOpenTimer)
    currentSearchTerm = 'filter: images'
    bar.value = currentSearchTerm
    bar.focus()
    output.replaceChildren(imageSimilarityStatus('finding similar images…'))
    quickOpenProgress(true)
    output.setAttribute('aria-busy', 'true')
    try {
      await ensureAttachmentData()
      if (generation !== queryGeneration || !isActive) return
      const documents = await imageSimilarity.search(seed, numSearchResults * 8)
      if (generation !== queryGeneration || !isActive) return
      displayQuickOpenResults(documents, '', 'images')
      output.dataset.imageSeed = seed
    } catch (error) {
      if (isSemanticCancellation(error) || generation !== queryGeneration || !isActive) return
      output.replaceChildren(
        imageSimilarityStatus('similar images unavailable. Type an image query to continue.'),
      )
    } finally {
      if (generation === queryGeneration) {
        quickOpenProgress(false)
        output.removeAttribute('aria-busy')
      }
    }
  }

  function displayQuickOpenResults(
    documents: SemanticDocument[],
    query: string,
    scope: SearchScope,
  ) {
    const items: Item[] = []
    for (const document of documents) {
      if (!isFullSlug(document.slug) || isSearchExcluded(document.slug)) continue
      if (
        scope === 'images' &&
        (!searchableImage(document.slug) || (imageIds && !imageIds.has(document.slug)))
      )
        continue
      const fileData = data?.[document.slug]
      if (
        !fileData ||
        fileData.protected ||
        isSearchExcluded(fileData.fileName) ||
        !searchScopeIncludes(scope, { ...fileData, slug: document.slug })
      )
        continue
      items.push({
        slug: document.slug,
        name: fileData.title || fileData.fileName,
        similarity: document.score,
      })
      if (items.length === numSearchResults) break
    }
    output.removeAttribute('aria-busy')
    quickOpenProgress(false)
    displayResults(items, query)
  }

  async function onType(e: HTMLElementEventMap['input']) {
    delete output.dataset.imageSeed
    imageSimilarity.cancel()
    imageSemantic?.cancel()
    quickOpenSemantic?.cancel()
    queryGeneration += 1
    const value = (e.target as HTMLInputElement).value
    const commandMode = value.trimStart().startsWith('>')
    let nextActionType: ActionType
    if (commandMode) {
      if (actionType !== 'command') fallbackActionType = actionType
      nextActionType = 'command'
    } else {
      nextActionType = actionType === 'command' ? fallbackActionType : actionType
    }
    if (actionType !== nextActionType) {
      actionType = nextActionType
      if (actionType === 'command' && commandHistoryIndex === -1) {
        recordCommandLocation({ type: 'commands' })
      }
      syncPaletteHelper()
    }
    currentSearchTerm = actionType === 'command' ? commandSearchTerm(value) : value
    window.clearTimeout(quickOpenTimer)
    if (actionType === 'quick_open' && currentSearchTerm.trim()) {
      const { term: query, scope } = parseSearchQuery(currentSearchTerm)
      const rawQuery = currentSearchTerm
      const generation = queryGeneration
      const cached = quickOpenSemantic?.cached(query, numSearchResults * 8)
      if (scope === 'notes' && cached !== undefined) {
        displayQuickOpenResults(cached, query, scope)
        return
      }
      quickOpenTimer = window.setTimeout(() => {
        void querySearch(rawQuery, generation)
      }, 60)
    } else {
      if (actionType !== 'quick_open') {
        quickOpenProgress(false)
        output.removeAttribute('aria-busy')
      }
      await querySearch(currentSearchTerm, queryGeneration)
    }
  }

  function displayResults(finalResults: Item[], currentSearchTerm: string) {
    if (actionType !== 'quick_open') return
    if (!finalResults) return

    output.replaceChildren()
    currentHover = null

    const noMatchEl = document.createElement('div')
    noMatchEl.classList.add('suggestion-item', 'no-match')
    noMatchEl.tabIndex = -1
    const noMatchContent = document.createElement('div')
    noMatchContent.className = 'suggestion-content'
    const noMatchTitle = document.createElement('div')
    noMatchTitle.className = 'suggestion-title'
    noMatchTitle.textContent = currentSearchTerm
    noMatchContent.appendChild(noMatchTitle)
    const noMatchAux = document.createElement('div')
    noMatchAux.className = 'suggestion-aux'
    const noMatchAction = document.createElement('span')
    noMatchAction.className = 'suggestion-action'
    noMatchAction.textContent = 'enter to schedule a chat'
    noMatchAux.appendChild(noMatchAction)
    noMatchEl.append(noMatchContent, noMatchAux)

    const onNoMatchClick = () => {
      window.location.href = `mailto:contact@aarnphm.xyz?subject=Chat about: ${encodeURIComponent(currentSearchTerm)}`
      hidePalette()
    }

    noMatchEl.addEventListener('click', onNoMatchClick)
    window.addCleanup(() => noMatchEl.removeEventListener('click', onNoMatchClick))
    if (finalResults.length === 0) {
      output.appendChild(noMatchEl)
    } else {
      output.append(...finalResults.map(toHtml))
    }
    setFocusFirstChild()
  }

  function setFocusFirstChild() {
    const firstChild = output.firstElementChild
    if (!(firstChild instanceof HTMLDivElement)) {
      currentHover = null
      return
    }
    firstChild.classList.add('focus')
    currentHover = firstChild
  }

  function toHtml({ name, slug, similarity }: Item) {
    const item = document.createElement('div')
    item.classList.add('suggestion-item')
    item.tabIndex = -1
    item.dataset.slug = slug

    const content = document.createElement('a')
    content.href = new URL(resolveRelative(currentSlug, slug), location.toString()).toString()
    content.classList.add('suggestion-content')
    const title = document.createElement('div')
    title.classList.add('suggestion-title')
    appendHighlightedText(title, parseSearchQuery(currentSearchTerm).term, name)
    content.appendChild(title)

    const aux = document.createElement('div')
    aux.classList.add('suggestion-aux')
    if (!searchableImage(slug) && similarity !== undefined && Number.isFinite(similarity)) {
      item.dataset.similarity = String(similarity)
      const score = document.createElement('span')
      score.className = 'result-likelihood'
      score.title = 'cosine similarity'
      score.setAttribute('aria-label', `Cosine similarity: ${similarity.toFixed(3)}`)
      score.textContent = similarity.toFixed(3)
      aux.append(score)
    }
    if (data) {
      let ranker = noteRankers.get(data)
      if (!ranker) {
        ranker = createNoteRanker(data)
        noteRankers.set(data, ranker)
      }
      const rank = ranker.position(slug)
      if (rank) {
        aux.append(noteRankDistribution(ranker.global(), rank))
      }
    }

    if (searchableImage(slug))
      aux.append(imageSimilarityButton(slug, showSimilarImages, similarity))
    item.append(content, aux)

    const onClick = (event: MouseEvent) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
      event.preventDefault()
      window.spaNavigate(new URL(resolveRelative(currentSlug, slug), location.toString()))
      hidePalette()
    }

    const onMouseEnter = () => {
      // Remove focus class from all other items
      output.querySelectorAll<HTMLDivElement>('.suggestion-item.focus').forEach(el => {
        el.classList.remove('focus')
      })
      // Add focus to current item
      item.classList.add('focus')
      currentHover = item
    }

    item.addEventListener('click', onClick)
    item.addEventListener('mouseenter', onMouseEnter)
    window.addCleanup(() => {
      item.removeEventListener('click', onClick)
      item.removeEventListener('mouseenter', onMouseEnter)
    })

    return item
  }

  document.addEventListener('keydown', shortcutHandler)
  bar.addEventListener('input', onType)
  window.addCleanup(() => {
    document.removeEventListener('keydown', shortcutHandler)
    bar.removeEventListener('input', onType)
    window.clearTimeout(quickOpenTimer)
  })

  registerEscapeHandler(container, hidePalette)
  if (currentSlug === 'index') {
    let idle = 0
    let timer = 0
    const preload = () => {
      if (!isActive) return
      for (const action of [...CONNECTOR_ACTS, ...COMMAND_ACTS]) {
        if (!actionCache.has(action)) actionCache.set(action, createActComponent(action, ''))
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
