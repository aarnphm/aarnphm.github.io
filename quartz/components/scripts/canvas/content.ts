import { Marked } from 'marked'
import type { NodeData } from './data'
import { fetchCanonical } from '../../../util/fetch-canonical'
import { normalizeRelativeURLs } from '../../../util/path'
import { loadPageStyles } from '../page-styles'
import { cleanupHydratedRoot } from '../root-lifecycle'

// page chunks share one `marked` module, so canvas options stay off its global instance
const marked = new Marked({ breaks: true, gfm: true })

export function renderTextNodeContent(node: NodeData): string {
  const html = marked.parse(node.resolvedText ?? node.text ?? '', { async: false })

  const doc = htmlParser.parseFromString(html, 'text/html')
  const anchors = Array.from(doc.body.querySelectorAll('a'))
  const remaining = [...(node.wikilinks ?? [])]
  const isLabel = doc.body.children.length === 1 && doc.body.firstElementChild?.tagName === 'P'

  anchors.forEach(anchor => {
    const href = anchor.getAttribute('href') ?? ''
    const idx = remaining.findIndex(link => link.resolvedHref === href)
    const match = idx >= 0 ? remaining.splice(idx, 1)[0] : undefined

    if (!match) {
      return
    }

    if (match.resolvedHref) {
      anchor.classList.add('internal')
      if (match.resolvedSlug) {
        const slug = match.resolvedSlug.startsWith('/')
          ? match.resolvedSlug
          : `/${match.resolvedSlug}`
        anchor.dataset.slug = slug
      }
    }

    if (match.missing) {
      anchor.classList.add('is-missing')
    }
  })

  normalizeRelativeURLs(doc, window.location.href)

  return `<div class="node-text${isLabel ? ' node-text-label' : ''}">${doc.body.innerHTML}</div>`
}

const filePreviewCache = new Map<string, Promise<string | null>>()
const htmlParser = new DOMParser()

export async function hydrateFileNodeContent(
  container: HTMLElement,
  node: NodeData,
  isActive: () => boolean,
  waitForWork: (isActive: () => boolean) => Promise<boolean>,
): Promise<void> {
  container.dataset.loading = 'true'

  try {
    const preview = await getFileNodeMarkup(node)
    if (!preview) {
      if (container.dataset.placeholder === 'true' && isActive()) {
        container.textContent = 'Preview unavailable'
        container.removeAttribute('data-placeholder')
      }
      return
    }

    if (!(await waitForWork(isActive)) || !isActive()) return
    const doc = htmlParser.parseFromString(preview, 'text/html')
    const article = doc.querySelector('article')
    if (!article) return
    const noteUrl = resolveFileNodeUrl(node)
    normalizeRelativeURLs(article, noteUrl ?? window.location.href)
    if (noteUrl) {
      article.querySelectorAll('a[href^="#"]').forEach(anchor => {
        if (!(anchor instanceof HTMLAnchorElement)) return
        const href = anchor.getAttribute('href')
        if (href) anchor.href = new URL(href, noteUrl).toString()
      })
    }
    await loadPageStyles(article)
    if (!(await waitForWork(isActive)) || !isActive()) return
    cleanupHydratedRoot(container)
    container.replaceChildren(...article.childNodes)
    container.removeAttribute('data-placeholder')
    const event: CustomEventMap['contentdecrypted'] = new CustomEvent('contentdecrypted', {
      detail: { article: container, content: container, slug: node.resolvedSlug },
    })
    document.dispatchEvent(event)
  } finally {
    if (isActive()) {
      container.dataset.loading = 'false'
    }
  }
}

async function getFileNodeMarkup(node: NodeData): Promise<string | null> {
  const targetUrl = resolveFileNodeUrl(node)
  if (!targetUrl) return null
  targetUrl.hash = ''

  const cacheKey = targetUrl.toString()

  if (!filePreviewCache.has(cacheKey)) {
    filePreviewCache.set(
      cacheKey,
      (async () => {
        try {
          const response = await fetchCanonical(targetUrl)
          if (!response.ok) {
            if (response.status === 404) return null
            throw new Error(`Failed to load canvas preview (${response.status})`)
          }

          const contentType = response.headers.get('Content-Type') ?? ''
          if (!contentType.startsWith('text/html')) {
            return null
          }

          return await response.text()
        } catch (error) {
          filePreviewCache.delete(cacheKey)
          console.error('Canvas preview failed:', error)
          return null
        }
      })(),
    )
  }

  return filePreviewCache.get(cacheKey)!
}

export function resolveFileNodeUrl(node: NodeData): URL | null {
  const base =
    (typeof node.resolvedHref === 'string' && node.resolvedHref.trim().length > 0
      ? node.resolvedHref
      : null) ??
    (typeof node.resolvedSlug === 'string' && node.resolvedSlug.trim().length > 0
      ? node.resolvedSlug
      : null) ??
    (typeof node.file === 'string' && node.file.trim().length > 0
      ? node.file.replace(/\.md$/, '')
      : null)

  if (!base) return null

  const trimmed = base.trim()
  if (trimmed.length === 0) return null

  const normalizedPath = trimmed.startsWith('/') ? trimmed : `/${trimmed}`
  return new URL(normalizedPath, window.location.origin)
}

export function wrapTextHtml(text: string): string {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}
