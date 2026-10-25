import { documentPreviewUrl, readDocumentPreview } from '../../../util/document-preview'
import { fetchCanonical } from '../../../util/fetch-canonical'
import { FullSlug, isFullSlug, normalizeRelativeURLs, resolveRelative } from '../../../util/path'
import { tokenizeTerm } from '../../../util/search-text'
import { codeMatchLabel, codeMatchLocation } from '../code-search-controls'
import { renderCodeMatch } from '../code-search-preview'
import { renderCoursePreview } from '../course-preview-card'
import { renderDocumentPreview } from '../document-preview-card'
import { imageSearchPreview, searchableImage } from '../image-search-controls'
import { requireData } from './engine'
import { renderFilePreview } from './file-preview'
import { completeSearchProgress, startSearchProgress } from './progress'
import { shared, type SearchSession } from './state'

const p = new DOMParser()
const fetchContentCache: Map<FullSlug, Document> = new Map()

function highlightHTML(searchTerm: string, el: HTMLElement) {
  const p = new DOMParser()
  const tokenizedTerms = tokenizeTerm(searchTerm)
  const html = p.parseFromString(el.innerHTML, 'text/html')
  html
    .querySelectorAll('script[data-notebook-runtime-data], [data-notebook-preview]')
    .forEach(node => node.remove())

  const createHighlightSpan = (text: string) => {
    const span = document.createElement('span')
    span.className = 'highlight'
    span.textContent = text
    return span
  }

  const highlightTextNodes = (node: Node, term: string) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const nodeText = node.nodeValue ?? ''
      const regex = new RegExp(term.toLowerCase(), 'gi')
      const matches = nodeText.match(regex)
      if (!matches || matches.length === 0) return
      const spanContainer = document.createElement('span')
      let lastIndex = 0
      for (const match of matches) {
        const matchIndex = nodeText.indexOf(match, lastIndex)
        spanContainer.appendChild(document.createTextNode(nodeText.slice(lastIndex, matchIndex)))
        spanContainer.appendChild(createHighlightSpan(match))
        lastIndex = matchIndex + match.length
      }
      spanContainer.appendChild(document.createTextNode(nodeText.slice(lastIndex)))
      node.parentNode?.replaceChild(spanContainer, node)
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      const element = node as Element
      if (
        element.classList.contains('highlight') ||
        element instanceof SVGElement ||
        element.tagName === 'SCRIPT' ||
        element.tagName === 'STYLE' ||
        element.hasAttribute('hidden')
      )
        return
      Array.from(node.childNodes).forEach(child => highlightTextNodes(child, term))
    }
  }

  for (const term of tokenizedTerms) {
    highlightTextNodes(html.body, term)
  }

  return html.body
}

export function clearPreview(s: SearchSession) {
  s.previewRequest?.abort()
  s.previewRequest = undefined
  if (s.preview) window.quartzPdfEmbeds?.cleanup(s.preview)
  s.preview?.replaceChildren()
  if (s.preview) delete s.preview.dataset.slug
}

export function setImageSearch(s: SearchSession, images: boolean) {
  s.imageSearch = images
  s.searchLayout.dataset.imageSearch = String(images)
  if (s.previewToggle) s.previewToggle.disabled = images
  if (images) setPreviewVisible(s, false)
  else updatePreviewToggle(s)
}

function updatePreviewToggle(s: SearchSession) {
  const label = s.imageSearch
    ? 'preview unavailable for images'
    : s.previewVisible
      ? 'hide preview'
      : 'show preview'
  s.previewToggle?.setAttribute('aria-label', label)
  s.previewToggle?.setAttribute('title', label)
  s.previewToggle?.setAttribute('aria-expanded', String(s.previewVisible))
}

export function setPreviewVisible(s: SearchSession, visible: boolean) {
  const { searchLayout, preview } = s
  s.previewVisible = visible && s.enablePreview && !s.imageSearch
  searchLayout.dataset.preview = String(s.previewVisible)
  updatePreviewToggle(s)
  if (preview) preview.inert = !s.previewVisible
  if (s.previewVisible) void displayPreview(s, s.currentHover)
  else clearPreview(s)
}

export function togglePreview(s: SearchSession) {
  setPreviewVisible(s, !s.previewVisible)
}

export function resolveUrl(s: SearchSession, slug: FullSlug): URL {
  const baseSlug = (s.searchElement.dataset.currentSlug as FullSlug | undefined) ?? s.currentSlug
  return new URL(resolveRelative(baseSlug, slug), location.toString())
}

async function fetchContent(
  s: SearchSession,
  slug: FullSlug,
  signal: AbortSignal,
): Promise<Document> {
  const cached = fetchContentCache.get(slug)
  if (cached) return cached

  const targetUrl = resolveUrl(s, slug)
  const response = await fetchCanonical(targetUrl, { signal })
  if (!response.ok) throw new Error(`Could not fetch ${targetUrl}: ${response.status}`)
  const page = p.parseFromString(await response.text(), 'text/html')
  normalizeRelativeURLs(page, targetUrl)
  fetchContentCache.set(slug, page)
  return page
}

function isPdfHit(s: SearchSession, el: HTMLElement, slug: FullSlug): boolean {
  if (s.codeMatches.has(el) || el.dataset.directFile !== 'true') return false
  if (el.dataset.canvas === 'true' || el.dataset.bases === 'true') return false
  const file = s.data?.[slug]
  if (el.dataset.protected === 'true' || file?.protected === true) return false
  return (file?.fileName || slug).toLowerCase().endsWith('.pdf')
}

export async function displayPreview(s: SearchSession, el: HTMLElement | null) {
  const { preview } = s
  if (!s.previewVisible || !el || !preview) return
  const codeMatch = s.codeMatches.get(el)
  const slug = el.dataset.codeOwner ?? el.id
  if (!isFullSlug(slug)) return
  const previewKey = codeMatch
    ? `${slug}:${codeMatch.unit.id}:${codeMatch.ranges[codeMatch.selectedRange]?.start.offset}`
    : el.dataset.pdfPage
      ? `${slug}#${el.dataset.pdfPage}`
      : slug
  if (s.previewRequest && !s.previewRequest.signal.aborted && preview.dataset.slug === previewKey)
    return
  // Between PDF hits or code matches the mounted chrome stays; the renderer swaps its content.
  if (
    (codeMatch && preview.querySelector(':scope > .code-match-preview')) ||
    (isPdfHit(s, el, slug) && preview.querySelector(':scope > .search-pdf-preview'))
  ) {
    s.previewRequest?.abort()
  } else {
    clearPreview(s)
  }
  const request = new AbortController()
  s.previewRequest = request
  preview.dataset.slug = previewKey
  startSearchProgress(s, 'preview')
  request.signal.addEventListener('abort', () => completeSearchProgress(s, 'preview'), {
    once: true,
  })
  try {
    if (codeMatch) {
      await renderCodeMatch(s.currentSlug, codeMatch, preview, request.signal, false, () => {
        const location = el.querySelector('.code-search-mobile-location')
        if (location) {
          location.textContent = `${codeMatchLocation(codeMatch)} · match ${codeMatch.selectedRange + 1}/${codeMatch.ranges.length}`
        }
        el.setAttribute('aria-label', el.textContent?.trim() ?? '')
        const excerpt = el.closest('tr')?.querySelector('.search-column-match > span')
        if (excerpt) excerpt.textContent = `${codeMatchLabel(codeMatch)} · ${codeMatch.excerpt}`
        preview.dataset.slug = `${slug}:${codeMatch.unit.id}:${codeMatch.ranges[codeMatch.selectedRange]?.start.offset}`
      })
      return
    }
    const isProtected = el.dataset.protected === 'true' || requireData(s)[slug]?.protected === true
    if (isProtected) {
      const previewInner = document.createElement('div')
      previewInner.classList.add('preview-inner', 'preview-redacted')

      const blurLayer = document.createElement('div')
      blurLayer.className = 'preview-redacted-blur'
      const label = document.createElement('span')
      label.className = 'preview-redacted-label'
      label.textContent = 'redacted'

      previewInner.append(blurLayer, label)
      preview.replaceChildren(previewInner)
      return
    }

    const documentType =
      el.dataset.canvas === 'true' ? 'canvas' : el.dataset.bases === 'true' ? 'base' : null
    if (documentType) {
      const loading = document.createElement('p')
      loading.setAttribute('role', 'status')
      loading.textContent = 'Loading preview…'
      preview.appendChild(loading)
      try {
        const response = await fetch(documentPreviewUrl(slug, resolveUrl(s, slug).toString()), {
          signal: request.signal,
        })
        const summary = response.ok
          ? readDocumentPreview(await response.json(), documentType, slug)
          : null
        if (request.signal.aborted) return
        if (summary) {
          const inner = document.createElement('div')
          inner.classList.add('preview-inner', 'document-popover', 'search-document-preview')
          renderDocumentPreview(summary, resolveUrl(s, slug).toString(), inner)
          preview.replaceChildren(inner)
          preview.scrollTop = 0
          return
        }
      } catch (error) {
        if (request.signal.aborted) return
        console.warn('[Search] document preview unavailable:', error)
      }
    }

    if (searchableImage(slug)) {
      preview.replaceChildren(
        imageSearchPreview(slug, resolveUrl(s, slug).toString(), s.showSimilarImages),
      )
      return
    }

    if (documentType || el.dataset.directFile === 'true') {
      const fileData = requireData(s)[slug]
      if (
        await renderFilePreview(
          resolveUrl(s, slug),
          fileData?.fileName || slug,
          preview,
          request.signal,
          Number(el.dataset.pdfPage) || undefined,
        )
      )
        return
      if (request.signal.aborted) return
      const previewInner = document.createElement('div')
      previewInner.classList.add('preview-inner')

      const metaContainer = document.createElement('div')
      metaContainer.style.padding = '1rem'
      metaContainer.style.color = 'var(--gray)'
      metaContainer.style.fontSize = '0.9rem'

      const metadata = (label: string, value: string) => {
        const row = document.createElement('p')
        const heading = document.createElement('strong')
        heading.textContent = `${label}:`
        row.append(heading, ` ${value}`)
        metaContainer.appendChild(row)
      }

      metadata('File', fileData?.fileName || slug)

      if (fileData?.description) {
        metadata('Description', fileData.description)
      }

      if (fileData?.tags && fileData.tags.length > 0) {
        metadata('Tags', fileData.tags.join(', '))
      }

      previewInner.appendChild(metaContainer)
      preview.replaceChildren(previewInner)
      return
    }

    const page = await fetchContent(s, slug, request.signal)
    if (request.signal.aborted) return
    if (page.querySelector('.course-home, [data-course-card]')) {
      const inner = document.createElement('div')
      inner.classList.add('preview-inner', 'course-popover', 'search-course-preview')
      renderCoursePreview(page, resolveUrl(s, slug), inner)
      preview.replaceChildren(inner)
      preview.scrollTop = 0
      return
    }

    const innerDiv = Array.from(page.querySelectorAll<HTMLElement>('.popover-hint')).flatMap(el => {
      const highlighted = highlightHTML(shared.currentSearchTerm, el)
      if (
        !el.classList.contains('notebook-page') &&
        !el.classList.contains('notebook-runtime-page')
      )
        return [...highlighted.children]
      const article = document.createElement('article')
      article.className = el.className
      article.classList.remove('popover-hint')
      article.append(...highlighted.children)
      return [article]
    })
    const previewInner = document.createElement('div')
    previewInner.classList.add('preview-inner')
    previewInner.append(...innerDiv)
    if (previewInner.querySelector('.stream-feed')) previewInner.classList.add('stream-preview')
    preview.replaceChildren(previewInner)

    const highlights = [...preview.getElementsByClassName('highlight')].sort(
      (a, b) => b.innerHTML.length - a.innerHTML.length,
    )
    if (highlights.length > 0) {
      const highlight = highlights[0]
      const containerRect = preview.getBoundingClientRect()
      const highlightRect = highlight.getBoundingClientRect()
      const relativeTop = highlightRect.top - containerRect.top + preview.scrollTop - 20
      preview.scrollTo({ top: relativeTop, behavior: 'smooth' })
    }
  } catch (error) {
    if (request.signal.aborted) return
    console.warn('[Search] preview unavailable:', error)
    window.quartzPdfEmbeds?.cleanup(preview)
    const message = document.createElement('p')
    message.setAttribute('role', 'status')
    message.textContent = 'preview unavailable. '
    const open = document.createElement('a')
    open.href = resolveUrl(s, slug).toString()
    open.textContent = 'open page'
    message.appendChild(open)
    preview.replaceChildren(message)
  } finally {
    if (s.previewRequest === request) completeSearchProgress(s, 'preview')
  }
}
