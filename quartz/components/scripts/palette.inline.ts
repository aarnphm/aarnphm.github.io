import { cellSourceLegacyPrefix, cellSourceStoragePrefix } from '../../runtime/notebook/persistence'
import {
  commentRoomToggleEvent,
  readCommentRoomEnabled,
  writeCommentRoomEnabled,
} from '../../util/comment-room'
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
import { FullSlug, normalizeRelativeURLs, resolveRelative } from '../../util/path'
import { populateSearchIndex, querySearchIndex, SearchItem } from './search-index'
import {
  tokenizeTerm,
  registerEscapeHandler,
  removeAllChildren,
  fetchCanonical,
  createSidePanel,
  getOrCreateSidePanel,
} from './util'

interface Item extends SearchItem {
  target: string
}

const numSearchResults = 10

const localStorageKey = 'recent-notes'

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
  const kbdPrefix = '<kbd>↵</kbd>'
  const svgMatch = auxInnerHtml.match(
    /^<svg width="1em" height="1em"><use href="([^"]+)" \/><\/svg>$/,
  )
  if (auxInnerHtml.startsWith(kbdPrefix)) {
    const key = document.createElement('kbd')
    key.textContent = '↵'
    action.appendChild(key)
    action.append(document.createTextNode(auxInnerHtml.slice(kbdPrefix.length)))
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

function getRecents(): Set<FullSlug> {
  return new Set(JSON.parse(localStorage.getItem(localStorageKey) ?? '[]'))
}

function addToRecents(slug: FullSlug) {
  const visited = getRecents()
  visited.add(slug)
  localStorage.setItem(localStorageKey, JSON.stringify([...visited]))
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

function notebookKernelLanguageLabel(language: string): string {
  const token = notebookKernelLanguageToken(language)
  return token === language ? language : `${token} ${language}`
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
  const hint = 'suggest: use the username that matches your gravatar'
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
document.addEventListener('nav', e => {
  const currentSlug = e.detail.url
  const container = document.getElementById('palette-container')
  if (!container) return

  const bar = container.querySelector('#bar') as HTMLInputElement
  const output = container.getElementsByTagName('output')[0]
  const helper = container.querySelector('ul#helper') as HTMLUListElement
  let currentHover: HTMLDivElement | null = null
  let data: ContentIndex | null = null
  let idDataMap: FullSlug[] = []
  let isActive = true
  let queryGeneration = 0
  let commandHistory: CommandLocation[] = []
  let commandHistoryIndex = -1

  window.addCleanup(() => {
    isActive = false
  })

  let dataReady: Promise<ContentIndex> | null = null

  function ensureData() {
    dataReady ??= fetchData.then(async resolved => {
      if (!isActive) return resolved
      data = resolved
      idDataMap = Object.keys(resolved) as FullSlug[]
      await fillDocument(resolved)
      if (!isActive) return resolved
      if (actionType === 'quick_open' && container?.classList.contains('active')) {
        getRecentItems()
      }
      return resolved
    })
    return dataReady
  }

  function hidePalette() {
    queryGeneration += 1
    container?.classList.remove('active')
    if (bar) {
      bar.value = '' // clear the input when we dismiss the search
    }
    if (output) {
      removeAllChildren(output)
    }

    actionType = 'quick_open' // reset search type after closing
    commandHistory = []
    commandHistoryIndex = -1
    syncPaletteHelper()
    recentItems = []
  }

  function commandInputValue(query = '') {
    return query.length === 0 ? '> ' : `> ${query}`
  }

  function commandSearchTerm(value: string) {
    return value.replace(/^\s*>\s?/, '').trimStart()
  }

  function syncPaletteHelper() {
    helper.querySelectorAll<HTMLLIElement>('li[data-quick-open]').forEach(el => {
      el.style.display = actionType === 'quick_open' ? '' : 'none'
    })
    helper.querySelectorAll<HTMLLIElement>('li[data-command]').forEach(el => {
      el.style.display = actionType === 'command' ? '' : 'none'
    })
  }

  function showPalette(actionTypeNew: ActionType, fallbackActionTypeNew?: FallbackActionType) {
    queryGeneration += 1
    actionType = actionTypeNew
    fallbackActionType =
      actionTypeNew === 'command' ? (fallbackActionTypeNew ?? fallbackActionType) : actionTypeNew
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
      if (data) {
        getRecentItems()
      } else if (output) {
        removeAllChildren(output)
        void ensureData()
      }
    }

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

  const COMMAND_ACTS: Action[] = [
    {
      name: 'kernels: run all',
      auxInnerHtml: '<kbd>↵</kbd> current page',
      keepOpen: true,
      onClick: () => {
        dispatchNotebookKernelRunAll()
        notifyToast('running all notebook kernels')
        showKernelItems(true)
      },
    },
    {
      name: 'show available kernels',
      auxInnerHtml: '<kbd>↵</kbd> notebooks',
      keepOpen: true,
      onClick: () => {
        showKernelItems()
      },
    },
    {
      name: 'show command listing',
      auxInnerHtml: '<kbd>↵</kbd> commands',
      keepOpen: true,
      onClick: () => {
        showCommandItems(true)
      },
    },
    {
      name: 'clean all local code segments',
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

  const createActComponent = ({
    name,
    auxInnerHtml,
    onClick,
    keepOpen,
    detail,
    titlePrefixInnerHtml,
    statusLabel,
    statusTone,
  }: Action) => {
    const item = document.createElement('div')
    item.classList.add('suggestion-item')

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
    appendHighlightedText(title, currentSearchTerm, name)
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
      removeAllChildren(output)
    }
    if (acts.length === 0) {
      if (bar.matches(':focus') && currentSearchTerm === '') {
        output.append(
          ...(actionType === 'command' ? COMMAND_ACTS : CONNECTOR_ACTS).map(createActComponent),
        )
      } else {
        const fallbackAction = actionType === 'command' ? COMMAND_ACTS[0] : CONNECTOR_ACTS[0]
        if (fallbackAction) output.append(createActComponent(fallbackAction))
      }
    } else {
      output.append(...acts.map(createActComponent))
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
            name: notebookKernelLanguageLabel(snapshot.language),
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
      name: `${command} ${snapshot.language} kernel`,
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

  let recentItems: Item[] = []
  function getRecentItems() {
    if (!data) {
      if (output) {
        removeAllChildren(output)
      }
      return
    }
    const loadedData = data
    const dataMap = idDataMap
    const visited = getRecents()

    if (output) {
      removeAllChildren(output)
    }

    const visitedArray = [...visited]
    const els =
      visited.size > numSearchResults
        ? visitedArray.slice(-numSearchResults).reverse()
        : visitedArray.reverse()

    // If visited >= 10, then we get the first recent 10 items
    // Otherwise, we will choose randomly from the set of data
    els.forEach(slug => {
      const id = dataMap.findIndex(s => s === slug)
      if (id !== -1) {
        //@ts-ignore
        recentItems.push({
          id,
          slug,
          name: loadedData[slug].fileName,
          title: loadedData[slug].title ?? '',
          content: loadedData[slug].content ?? '',
          aliases: loadedData[slug].aliases,
          target: '',
        })
      }
    })
    // Fill with random items from data
    const needed = numSearchResults - els.length
    if (needed != 0) {
      const availableSlugs = dataMap.filter(slug => !els.includes(slug))

      // Then add random items
      for (let i = 0; i < needed && availableSlugs.length > 0; i++) {
        const randomIndex = Math.floor(Math.random() * availableSlugs.length)
        const slug = availableSlugs[randomIndex]
        const id = dataMap.findIndex(s => s === slug)

        //@ts-ignore
        recentItems.push({
          id,
          slug: slug as FullSlug,
          name: loadedData[slug].fileName,
          title: loadedData[slug].title ?? '',
          content: loadedData[slug].content ?? '',
          aliases: loadedData[slug].aliases,
          target: '',
        })

        // Remove used slug to avoid duplicates
        availableSlugs.splice(randomIndex, 1)
      }
    }

    output.append(...recentItems.map(toHtml))
    setFocusFirstChild()
  }

  function eventTargetsCodeEditor(e: KeyboardEvent) {
    const target = e.target
    if (target instanceof Element && target.closest('.cm-editor')) return true
    const activeElement = document.activeElement
    return activeElement instanceof Element && activeElement.closest('.cm-editor') !== null
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

    if (e.key === 'o' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const barOpen = container?.classList.contains('active')
      if (barOpen) {
        hidePalette()
      } else {
        showPalette('quick_open')
      }
      return
    } else if (e.key === 'p' && (e.altKey || e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      const barOpen = container?.classList.contains('active')
      if (barOpen) {
        hidePalette()
      } else {
        showPalette('command', 'connector')
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

    if (currentHover) currentHover.classList.remove('focus')
    if (!container?.classList.contains('active')) return

    if (e.metaKey && e.altKey && e.key === 'Enter') {
      if (!currentHover) return
      const slug = currentHover.dataset.slug
      if (!slug) return

      try {
        const asidePanel = getOrCreateSidePanel()
        await fetchContent(currentSlug, slug as FullSlug).then(innerDiv => {
          asidePanel.dataset.slug = slug
          createSidePanel(asidePanel, ...innerDiv)
          window.notifyNav(slug as FullSlug)
          hidePalette()
        })
      } catch (error) {
        console.error('Failed to create side panel:', error)
      }
      return
    } else if (e.key === 'Enter') {
      // If result has focus, navigate to that one, otherwise pick first result
      if (output?.contains(currentHover)) {
        e.preventDefault()
        currentHover!.click()
      } else {
        const anchor = output.getElementsByClassName('suggestion-item')[0] as HTMLDivElement
        e.preventDefault()
        anchor.click()
      }
    } else if (actionType === 'command' && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      if (navigateCommandHistory(e.key === 'ArrowLeft' ? -1 : 1)) e.preventDefault()
    } else if (e.key === 'ArrowUp' || (e.ctrlKey && e.key === 'p')) {
      e.preventDefault()
      const items = output.querySelectorAll<HTMLDivElement>('.suggestion-item')
      if (items.length === 0) return

      const focusedElement = currentHover
        ? currentHover
        : output.querySelector<HTMLDivElement>('.suggestion-item.focus')

      // Remove focus from current element
      if (focusedElement) {
        focusedElement.classList.remove('focus')
        // Get the previous element or cycle to the last
        const currentIndex = Array.from(items).indexOf(focusedElement)
        const prevIndex = currentIndex <= 0 ? items.length - 1 : currentIndex - 1
        currentHover = items[prevIndex]
        items[prevIndex].classList.add('focus')
        items[prevIndex].focus()
      } else {
        // If no element is focused, start from the last one
        const lastIndex = items.length - 1
        items[lastIndex].classList.add('focus')
        items[lastIndex].focus()
      }
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
    } else if (e.key === 'ArrowDown' || (e.ctrlKey && e.key === 'n')) {
      e.preventDefault()
      const items = output.querySelectorAll<HTMLDivElement>('.suggestion-item')
      if (items.length === 0) return

      const focusedElement = currentHover
        ? currentHover
        : output.querySelector<HTMLDivElement>('.suggestion-item.focus')

      // Remove focus from current element
      if (focusedElement) {
        focusedElement.classList.remove('focus')
        // Get the next element or cycle to the first
        const currentIndex = Array.from(items).indexOf(focusedElement)
        const nextIndex = currentIndex >= items.length - 1 ? 0 : currentIndex + 1
        currentHover = items[nextIndex]
        items[nextIndex].classList.add('focus')
        items[nextIndex].focus()
      } else {
        // If no element is focused, start from the first one
        items[0].classList.add('focus')
        items[0].focus()
      }
    }
  }

  async function querySearch(currentSearchTerm: string, generation = queryGeneration) {
    const queryActionType = actionType
    if (actionType === 'quick_open') {
      await ensureData()
      if (generation !== queryGeneration || queryActionType !== actionType) return
      const searchResults = await querySearchIndex(currentSearchTerm, numSearchResults)
      if (generation !== queryGeneration || queryActionType !== actionType) return

      displayResults(
        searchResults
          .map(item => {
            const target =
              item.aliases.find(alias =>
                alias.toLowerCase().includes(currentSearchTerm.toLowerCase()),
              ) || ''
            return { ...item, target }
          })
          .sort((a, b) => {
            if ((!a?.target && !b?.target) || (a?.target && b?.target)) return 0
            if (a?.target && !b?.target) return -1
            if (!a?.target && b?.target) return 1
            return 0
          }),
        currentSearchTerm,
      )
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

  async function onType(e: HTMLElementEventMap['input']) {
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
    await querySearch(currentSearchTerm, queryGeneration)
  }

  function displayResults(finalResults: Item[], currentSearchTerm: string) {
    if (actionType !== 'quick_open') return
    if (!finalResults) return

    removeAllChildren(output)

    const noMatchEl = document.createElement('div')
    noMatchEl.classList.add('suggestion-item', 'no-match')
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
      if (bar.matches(':focus') && currentSearchTerm === '') {
        output.append(...recentItems.map(toHtml))
      } else {
        output.appendChild(noMatchEl)
      }
    } else {
      output.append(...finalResults.map(toHtml))
    }
    setFocusFirstChild()
  }

  function setFocusFirstChild() {
    // focus on first result, then also dispatch preview immediately
    const firstChild = output.firstElementChild as HTMLElement
    firstChild.classList.add('focus')
    currentHover = firstChild as HTMLInputElement
  }

  function toHtml({ name, slug, target }: Item) {
    const item = document.createElement('div')
    item.classList.add('suggestion-item')
    item.dataset.slug = slug

    const content = document.createElement('div')
    content.classList.add('suggestion-content')
    const title = document.createElement('div')
    title.classList.add('suggestion-title')
    appendHighlightedText(title, currentSearchTerm, target || name)
    if (target) {
      title.appendChild(document.createElement('br'))
      const subscript = document.createElement('span')
      subscript.className = 'subscript'
      subscript.textContent = slug
      title.appendChild(subscript)
    }
    content.appendChild(title)

    const aux = document.createElement('div')
    aux.classList.add('suggestion-aux')

    item.append(content, aux)

    const onClick = () => {
      addToRecents(slug)
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
  })

  registerEscapeHandler(container, hidePalette)
})

async function fillDocument(data: ContentIndex) {
  await populateSearchIndex(data)
}
