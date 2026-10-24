import type {
  NotebookKernelSnapshot,
  NotebookKernelStatus,
} from '../../../util/notebook-kernel-events'
import type { ImageSimilarity } from '../image-similarity'
import type { PdfSemantic } from '../pdf-semantic'
import type { ResultPages } from '../search-result-pages'
import { FullSlug } from '../../../util/path'
import { createImageSemantic, type ImageSemantic } from '../image-semantic'
import { siteSemantic, type SiteSemantic } from '../site-semantic'

export type ActionType = 'quick_open' | 'connector' | 'command'
export type FallbackActionType = Exclude<ActionType, 'command'>
export type CommandLocation =
  | { readonly type: 'commands' }
  | { readonly type: 'kernels' }
  | { readonly type: 'kernel-actions'; readonly snapshot: NotebookKernelSnapshot }
export interface Action {
  name: string
  onClick: (e: MouseEvent) => void
  auxInnerHtml: string
  keepOpen?: boolean
  detail?: string
  titlePrefixInnerHtml?: string
  statusLabel?: string
  statusTone?: NotebookKernelStatus
}

// The mode outlives each nav instance, so it stays module state.
export const paletteMode: {
  actionType: ActionType
  fallbackActionType: FallbackActionType
  currentSearchTerm: string
} = { actionType: 'quick_open', fallbackActionType: 'quick_open', currentSearchTerm: '' }

export interface PaletteState {
  readonly currentSlug: FullSlug
  readonly container: HTMLElement
  readonly bar: HTMLInputElement
  readonly output: HTMLOutputElement
  readonly helper: HTMLUListElement
  readonly rankButton: HTMLAnchorElement
  readonly progress: HTMLDivElement
  readonly imageSimilarity: ImageSimilarity
  readonly pdfSearch: PdfSemantic
  readonly actionCache: WeakMap<Action, HTMLDivElement>
  readonly connectorActs: Action[]
  commandActs: Action[]
  imageIds: ReadonlySet<string> | undefined
  imageSemantic: ImageSemantic | null
  imageReady: Promise<ImageSemantic> | null
  imageActive: boolean
  currentHover: HTMLElement | null
  resultPages: ResultPages | null
  previousFocus: HTMLElement | null
  data: ContentIndex | null
  isActive: boolean
  queryGeneration: number
  commandHistory: CommandLocation[]
  commandHistoryIndex: number
  contentDataReady: Promise<ContentIndex> | null
  attachmentDataReady: Promise<ContentIndex> | null
  quickOpenReady: ReturnType<typeof siteSemantic> | null
  quickOpenSemantic: SiteSemantic | null
  quickOpenTimer: number
  measuredPaletteWidth: number
}

export function ensureImageSearch(s: PaletteState) {
  s.imageReady ??= createImageSemantic()
    .then(service => {
      if (!s.imageActive) {
        service.dispose()
        throw new Error('image search page closed')
      }
      s.imageSemantic = service
      return service
    })
    .catch(error => {
      s.imageReady = null
      throw error
    })
  return s.imageReady
}

export function quickOpenProgress(s: PaletteState, loading: boolean) {
  s.progress.setAttribute('aria-hidden', String(!loading))
  s.progress.style.opacity = loading ? '1' : '0'
  s.progress.style.width = loading ? '100%' : '0'
  s.progress.style.animation = loading ? 'semantic-progress-sweep 2.5s linear infinite' : 'none'
}

export function ensureContentData(s: PaletteState) {
  s.contentDataReady ??= fetchData.then(resolved => {
    if (s.isActive) s.data = resolved
    return resolved
  })
  return s.contentDataReady
}

export function ensureAttachmentData(s: PaletteState) {
  s.attachmentDataReady ??= (typeof fetchSearchData === 'undefined' ? fetchData : fetchSearchData)
    .catch(error => {
      console.warn('[QuickOpen] attachment index unavailable:', error)
      return fetchData
    })
    .then(resolved => {
      if (s.isActive) s.data = resolved
      return resolved
    })
  return s.attachmentDataReady
}

export function ensureQuickOpen(s: PaletteState) {
  s.quickOpenReady ??= Promise.all([ensureContentData(s), siteSemantic()]).then(
    ([, service]) => {
      s.quickOpenSemantic = service
      return service
    },
    error => {
      throw error
    },
  )
  return s.quickOpenReady
}

export function quickOpenStatus(s: PaletteState, message: string) {
  s.resultPages?.dispose()
  s.resultPages = null
  s.currentHover = null
  if (!message) {
    s.output.replaceChildren()
    return
  }
  const status = document.createElement('p')
  status.setAttribute('role', 'status')
  status.textContent = message
  s.output.replaceChildren(status)
}

export function hidePalette(s: PaletteState) {
  s.resultPages?.dispose()
  s.resultPages = null
  s.imageSimilarity.cancel()
  s.imageSemantic?.cancel()
  s.pdfSearch.cancel()
  s.quickOpenSemantic?.cancel()
  s.queryGeneration += 1
  window.clearTimeout(s.quickOpenTimer)
  s.output.removeAttribute('aria-busy')
  quickOpenProgress(s, false)
  s.container?.classList.remove('active')
  if (s.container?.matches(':popover-open')) s.container.hidePopover()
  if (s.bar) {
    s.bar.value = '' // clear the input when we dismiss the search
  }
  if (s.output) {
    s.output.replaceChildren()
  }

  paletteMode.actionType = 'quick_open' // reset search type after closing
  s.commandHistory = []
  s.commandHistoryIndex = -1
  syncPaletteHelper(s)
  if (s.previousFocus?.isConnected) s.previousFocus.focus({ preventScroll: true })
  s.previousFocus = null
}

export function commandInputValue(query = '') {
  return query.length === 0 ? '> ' : `> ${query}`
}

export function commandSearchTerm(value: string) {
  return value.replace(/^\s*>\s?/, '').trimStart()
}

export function syncPaletteHelper(s: PaletteState) {
  s.container.dataset.mode = paletteMode.actionType
  s.rankButton.hidden = paletteMode.actionType !== 'quick_open'
  s.bar.placeholder =
    paletteMode.actionType === 'quick_open'
      ? 'filter: images · pdf · code · py · all'
      : 'sélectionnez une option…'
  s.bar.setAttribute('aria-label', s.bar.placeholder)
  s.helper.querySelectorAll<HTMLLIElement>('li[data-quick-open]').forEach(el => {
    el.style.display = paletteMode.actionType === 'quick_open' ? '' : 'none'
  })
  s.helper.querySelectorAll<HTMLLIElement>('li[data-command]').forEach(el => {
    el.style.display = paletteMode.actionType === 'command' ? '' : 'none'
  })
}

export function setFocusFirstChild(s: PaletteState) {
  const firstChild = s.output.firstElementChild
  if (!(firstChild instanceof HTMLDivElement)) {
    s.currentHover = null
    return
  }
  firstChild.classList.add('focus')
  s.currentHover = firstChild
}
