import DOMPurify from 'dompurify'
import type { Link } from '../../types/curius'
import {
  isCuriusPreviewImageUrl,
  parseCuriusPreview,
  type CuriusPreviewResponse,
} from '../../util/curius-preview'
import { parseGithubRepositoryUrl } from '../../util/github-embed'
import { parseWikipediaTarget } from '../../util/wikipedia'
import { buildYouTubeEmbed } from '../../util/youtube'
import { registerEscapeHandler } from './escape-handler'
import { currentNavSignal } from './nav-lifecycle'

interface PreviewRequest {
  link: Link
  trigger: HTMLElement
}

declare global {
  interface DocumentEventMap {
    'curius:preview': CustomEvent<PreviewRequest>
  }
}

export function isCuriusPrimaryClick(event: MouseEvent): boolean {
  return event.button === 0 && !event.altKey && !event.ctrlKey && !event.metaKey
}

export function requestCuriusPreview(link: Link, trigger: HTMLElement): void {
  trigger.dispatchEvent(
    new CustomEvent<PreviewRequest>('curius:preview', { bubbles: true, detail: { link, trigger } }),
  )
}

export function activateCuriusLink(link: Link, trigger: HTMLElement, original: boolean): void {
  if (original) {
    const href = safeSourceUrl(link.link)
    if (href) window.open(href, '_blank', 'noopener,noreferrer')
    return
  }
  requestCuriusPreview(link, trigger)
}

export function bindCuriusPreview(link: Link, anchor: HTMLAnchorElement): void {
  const signal = currentNavSignal()
  anchor.setAttribute('aria-controls', 'curius-preview')
  anchor.setAttribute('aria-haspopup', 'dialog')
  anchor.addEventListener(
    'click',
    event => {
      if (!isCuriusPrimaryClick(event)) return
      event.preventDefault()
      event.stopPropagation()
      activateCuriusLink(link, anchor, event.shiftKey)
    },
    { signal },
  )
  anchor.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Enter' || event.altKey || event.ctrlKey || event.metaKey) return
      event.preventDefault()
      event.stopPropagation()
      activateCuriusLink(link, anchor, event.shiftKey)
    },
    { signal },
  )
}

function plainText(value: string): string {
  return new DOMParser().parseFromString(value, 'text/html').body.textContent?.trim() ?? ''
}

function safeSourceUrl(raw: string): string | null {
  try {
    const url = new URL(raw)
    return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password
      ? url.href
      : null
  } catch {
    return null
  }
}

function articleContent(html: string): HTMLElement {
  const article = document.createElement('article')
  article.className = 'curius-preview-article'
  article.innerHTML = DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true, mathMl: true },
    FORBID_TAGS: [
      'style',
      'form',
      'input',
      'button',
      'textarea',
      'iframe',
      'object',
      'embed',
      'link',
      'meta',
    ],
    FORBID_ATTR: ['style', 'srcset', 'autofocus'],
    ALLOW_DATA_ATTR: false,
    ADD_ATTR: ['data-lang', 'data-latex', 'data-callout'],
  })
  for (const element of article.querySelectorAll('[src], [poster]')) {
    for (const attribute of ['src', 'poster']) {
      const raw = element.getAttribute(attribute)
      if (!raw) continue
      if (
        element.tagName !== 'IMG' ||
        attribute !== 'src' ||
        !isCuriusPreviewImageUrl(raw, location.origin)
      ) {
        element.removeAttribute(attribute)
      }
    }
  }
  for (const anchor of article.querySelectorAll('a')) {
    if (anchor.classList.contains('arena-footnote-backref')) {
      anchor.textContent = (anchor.textContent ?? '').replace(/↩[\uFE0E\uFE0F]?/gu, '↩\uFE0E')
    }
    const raw = anchor.getAttribute('href')
    if (!raw || raw.startsWith('#')) continue
    const href = safeSourceUrl(raw)
    if (!href) anchor.removeAttribute('href')
    else {
      anchor.href = href
      anchor.target = '_blank'
      anchor.rel = 'noopener noreferrer'
      const wikipediaTarget = parseWikipediaTarget(href)
      if (wikipediaTarget) {
        anchor.classList.add('internal')
        anchor.dataset.wikipediaLang = wikipediaTarget.lang
        anchor.dataset.wikipediaTitle = wikipediaTarget.title
      }
    }
  }
  for (const image of article.querySelectorAll('img')) {
    image.loading = 'lazy'
    image.decoding = 'async'
    image.referrerPolicy = 'no-referrer'
  }
  return article
}

function savedText(link: Link): { text: string; kind: 'article' | 'excerpt' } | undefined {
  if (parseGithubRepositoryUrl(link.link)) return undefined
  const article = link.metadata?.full_text?.trim()
  if (article) return { text: article, kind: 'article' }
  const excerpt = plainText(link.snippet || '')
  if (parseWikipediaTarget(link.link) && /^Couldn't find lead section for \S+$/i.test(excerpt))
    return undefined
  return excerpt ? { text: excerpt, kind: 'excerpt' } : undefined
}

function youtubeContent(link: Link): HTMLIFrameElement | undefined {
  const href = safeSourceUrl(link.link)
  const video = href ? buildYouTubeEmbed(href) : undefined
  if (!video) return undefined
  const frame = document.createElement('iframe')
  frame.className = 'curius-preview-video'
  frame.src = video.src
  frame.title = plainText(link.title) || 'YouTube video'
  frame.allow = 'encrypted-media; fullscreen; picture-in-picture'
  frame.allowFullscreen = true
  frame.referrerPolicy = 'strict-origin-when-cross-origin'
  frame.sandbox.add('allow-scripts', 'allow-same-origin', 'allow-presentation')
  return frame
}

function savedContent(link: Link, includeText: boolean): HTMLElement[] {
  const sections: HTMLElement[] = []
  const highlights = link.highlights ?? []
  const saved = includeText ? savedText(link) : undefined
  if (saved) {
    const summary = document.createElement('section')
    summary.className = 'curius-preview-summary'
    summary.title =
      saved.kind === 'article' ? 'Article text saved by Curius' : 'Excerpt saved by Curius'
    summary.setAttribute('aria-label', summary.title)
    for (const paragraph of saved.text.split(/\n\s*\n/)) {
      const p = document.createElement('p')
      p.textContent = paragraph
      summary.append(p)
    }
    sections.push(summary)
  }
  if (highlights.length > 0) {
    const details = document.createElement('details')
    details.className = 'curius-preview-highlights'
    details.open = includeText
    const summary = document.createElement('summary')
    summary.textContent = `Saved highlights (${highlights.length})`
    const list = document.createElement('ul')
    for (const highlight of highlights) {
      const item = document.createElement('li')
      const quote = document.createElement('blockquote')
      quote.textContent = highlight.highlight
      item.append(quote)
      list.append(item)
    }
    details.append(summary, list)
    sections.push(details)
  }
  return sections
}

export function setupCuriusPreview(): void {
  const panel = document.querySelector<HTMLElement>('#curius-preview')
  const closeButton = document.querySelector<HTMLButtonElement>('#curius-preview-close')
  const toggleButton = document.querySelector<HTMLButtonElement>('#curius-preview-toggle')
  const backdrop = document.querySelector<HTMLButtonElement>('#curius-preview-backdrop')
  const source = document.querySelector<HTMLAnchorElement>('#curius-preview-source')
  const original = document.querySelector<HTMLAnchorElement>('#curius-preview-original')
  const content = document.querySelector<HTMLElement>('#curius-preview-content')
  const retryButton = document.querySelector<HTMLButtonElement>('#curius-preview-retry')
  if (
    !panel ||
    !closeButton ||
    !toggleButton ||
    !backdrop ||
    !source ||
    !original ||
    !content ||
    !retryButton
  )
    return

  const signal = currentNavSignal()
  const mobile = window.matchMedia('(max-width: 800px)')
  const cache = new Map<number, CuriusPreviewResponse>()
  let request: AbortController | undefined
  let selected: PreviewRequest | undefined
  let returnFocus: HTMLElement | undefined
  let activeRow: Element | null = null

  const clearSelection = () => {
    activeRow?.classList.remove('preview-active')
    activeRow = null
  }

  const close = (restoreFocus = true) => {
    request?.abort()
    request = undefined
    const trigger = returnFocus
    clearSelection()
    panel.hidden = true
    backdrop.hidden = true
    content.replaceChildren()
    panel.setAttribute('aria-busy', 'false')
    retryButton.disabled = false
    toggleButton.setAttribute('aria-expanded', 'false')
    toggleButton.setAttribute('aria-label', 'Open preview')
    toggleButton.title = 'Open preview'
    for (const anchor of [source, original]) {
      anchor.removeAttribute('href')
      anchor.hidden = true
    }
    document.body.classList.remove('curius-preview-open')
    if (restoreFocus) {
      const target =
        trigger?.isConnected && trigger.getClientRects().length > 0 ? trigger : toggleButton
      target?.focus({ preventScroll: true })
    }
  }

  const renderFallback = (link: Link, message?: string) => {
    const saved = savedContent(link, true)
    if (message && parseWikipediaTarget(link.link)) {
      const status = document.createElement('p')
      status.className = 'curius-preview-status'
      status.setAttribute('role', 'status')
      status.textContent = message
      content.replaceChildren(status, ...saved)
      return
    }
    content.replaceChildren(...saved)
    if (!savedText(link)) {
      document.dispatchEvent(
        new CustomEvent('toast', { detail: { message: 'No preview available for this link.' } }),
      )
    }
  }

  const render = (result: CuriusPreviewResponse, link: Link) => {
    content.replaceChildren()
    if (result.status === 'ready') {
      content.append(articleContent(result.readerHtml), ...savedContent(link, false))
    } else {
      renderFallback(link, result.message)
    }
  }

  const open = async (
    selection: PreviewRequest,
    refresh = false,
    focusTarget = selection.trigger,
  ) => {
    request?.abort()
    request = undefined
    panel.setAttribute('aria-busy', 'false')
    retryButton.disabled = false
    clearSelection()
    selected = selection
    returnFocus = focusTarget
    const { link, trigger } = selection
    activeRow =
      trigger.closest('.curius-item') ?? trigger.closest('.curius-item-title, .curius-search-link')
    activeRow?.classList.add('preview-active')
    toggleButton.setAttribute('aria-expanded', 'true')
    toggleButton.setAttribute('aria-label', 'Close preview')
    toggleButton.title = 'Close preview'
    source.textContent = plainText(link.title) || link.link
    const href = safeSourceUrl(link.link)
    for (const anchor of [source, original]) {
      if (href) anchor.href = href
      else anchor.removeAttribute('href')
      anchor.hidden = !href
    }
    panel.hidden = false
    backdrop.hidden = false
    panel.setAttribute('aria-modal', String(mobile.matches))
    document.body.classList.add('curius-preview-open')
    const video = youtubeContent(link)
    content.replaceChildren(...(video ? [video] : savedContent(link, true)))
    content.scrollTop = 0
    panel.focus({ preventScroll: true })

    if (video) {
      content.append(...savedContent(link, false))
      return
    }

    if (parseWikipediaTarget(link.link)) renderFallback(link, 'Loading Wikipedia article…')

    const id = link.id
    if (!Number.isSafeInteger(id) || id === undefined || id <= 0) {
      retryButton.disabled = true
      renderFallback(link)
      return
    }
    const cached = refresh ? undefined : cache.get(id)
    if (cached) {
      render(cached, link)
      return
    }

    const controller = new AbortController()
    request = controller
    panel.setAttribute('aria-busy', 'true')
    retryButton.disabled = true
    try {
      const response = await fetch(
        `/api/curius?query=preview&id=${id}${refresh ? '&refresh=1' : ''}`,
        {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(35_000)]),
          headers: { Accept: 'application/json' },
        },
      )
      const result = parseCuriusPreview(await response.json())
      if (controller.signal.aborted || signal.aborted) return
      if (!result || (result.status === 'ready' && result.linkId !== id))
        throw new Error('Invalid article preview response')
      if (result.status === 'ready') cache.set(id, result)
      render(result, link)
    } catch (error) {
      if (controller.signal.aborted || signal.aborted) return
      renderFallback(link, 'The preview could not be loaded. Retry or open the original.')
      console.warn('Curius preview could not load', error)
    } finally {
      if (request === controller) {
        request = undefined
        panel.setAttribute('aria-busy', 'false')
        retryButton.disabled = false
      }
    }
  }

  document.addEventListener('curius:preview', event => void open(event.detail), { signal })
  toggleButton.addEventListener(
    'click',
    () => {
      if (!panel.hidden) {
        close(false)
        toggleButton.focus({ preventScroll: true })
      } else if (selected) {
        void open(selected, false, toggleButton)
      } else {
        const first = document.querySelector<HTMLAnchorElement>(
          '#curius-fragments .curius-item-link > a',
        )
        first?.click()
        if (selected) returnFocus = toggleButton
      }
    },
    { signal },
  )
  closeButton.addEventListener('click', () => close(), { signal })
  backdrop.addEventListener('click', () => close(), { signal })
  retryButton.addEventListener(
    'click',
    () => selected && void open(selected, true, returnFocus ?? toggleButton),
    { signal },
  )
  mobile.addEventListener(
    'change',
    () => {
      panel.setAttribute('aria-modal', String(mobile.matches))
      if (mobile.matches && !panel.hidden && !panel.contains(document.activeElement))
        panel.focus({ preventScroll: true })
    },
    { signal },
  )
  panel.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Tab' || !mobile.matches) return
      const focusable = Array.from(
        panel.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), iframe, summary, [tabindex="0"]',
        ),
      ).filter(element => element.getClientRects().length > 0)
      const first = focusable[0]
      const last = focusable.at(-1)
      if (!first || !last) return
      if (
        event.shiftKey &&
        (document.activeElement === first || document.activeElement === panel)
      ) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    },
    { signal },
  )
  document.addEventListener(
    'focusin',
    event => {
      if (
        mobile.matches &&
        !panel.hidden &&
        event.target instanceof Node &&
        !panel.contains(event.target)
      )
        closeButton.focus({ preventScroll: true })
    },
    { signal },
  )
  registerEscapeHandler(
    panel,
    () => close(),
    () => !panel.hidden,
  )
  window.addCleanup(() => {
    close(false)
    selected = undefined
    returnFocus = undefined
    cache.clear()
  })
}
