import type { Element, Nodes } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import {
  classList,
  findElement,
  findElements,
  firstClassText,
  hasClass,
  hasId,
  metaContent,
  normalizeText,
  textContent,
  attributeValue,
} from './hast-query'
import { lessWrongPostUrl, type LessWrongPreview, type LessWrongTarget } from './lesswrong'
import { normalizeTocDepth, truncateText, type PreviewTocEntry } from './preview'

function parseInteger(value: string | undefined): number | undefined {
  if (!value) return undefined
  const match = value.replace(/−/g, '-').match(/-?\d+/)
  if (!match) return undefined
  const parsed = Number.parseInt(match[0], 10)
  return Number.isFinite(parsed) ? parsed : undefined
}

function parseDate(element: Element | undefined): string | undefined {
  if (!element) return undefined
  const timestamp = attributeValue(element, 'data-js-date')
  if (timestamp) {
    const milliseconds = Number.parseInt(timestamp, 10)
    if (Number.isFinite(milliseconds)) return new Date(milliseconds).toISOString()
  }
  const text = normalizeText(textContent(element))
  return text.length > 0 ? text : undefined
}

function previewExtract(root: Nodes): string | undefined {
  const postBody = findElement(root, item => hasClass(item, 'post-body'))
  if (postBody) {
    const paragraphs = findElements(postBody, item => item.tagName === 'p')
      .map(paragraph => normalizeText(textContent(paragraph)))
      .filter(text => text.length > 0)
      .slice(0, 3)

    const text = truncateText(paragraphs.join(' '), 640)
    if (text.length > 0) return text
  }

  const description = metaContent(root, 'og:description')
  return description ? truncateText(normalizeText(description), 640) : undefined
}

function tocDepth(element: Element): number {
  for (const className of classList(element)) {
    const match = /^toc-item-(\d+)$/.exec(className)
    if (match) return normalizeTocDepth(Number.parseInt(match[1], 10))
  }
  return 1
}

function tocEntriesFromContents(contents: Element): PreviewTocEntry[] {
  return findElements(contents, item => item.tagName === 'li')
    .flatMap(item => {
      const anchor = findElement(item, child => child.tagName === 'a')
      const text = anchor ? normalizeText(textContent(anchor)) : ''
      const href = anchor ? attributeValue(anchor, 'href')?.trim() : undefined
      if (text.length === 0 || !href || !href.startsWith('#')) return []
      return [{ text, href, depth: tocDepth(item) }]
    })
    .slice(0, 24)
}

function tocEntriesFromHeadings(postBody: Element): PreviewTocEntry[] {
  return findElements(postBody, item => /^h[1-6]$/.test(item.tagName))
    .flatMap(item => {
      const id = attributeValue(item, 'id')?.trim()
      const text = normalizeText(textContent(item))
      const depth = Number.parseInt(item.tagName.slice(1), 10)
      if (!id || text.length === 0) return []
      return [{ text, href: `#${id}`, depth: normalizeTocDepth(depth) }]
    })
    .slice(0, 24)
}

function tocEntries(root: Nodes): PreviewTocEntry[] {
  const postBody = findElement(root, item => hasClass(item, 'post-body'))
  if (!postBody) return []

  const contents = findElement(
    postBody,
    item => item.tagName === 'nav' && hasClass(item, 'contents'),
  )
  return contents ? tocEntriesFromContents(contents) : tocEntriesFromHeadings(postBody)
}

function topMeta(root: Nodes): Element | undefined {
  return findElement(root, item => hasClass(item, 'top-post-meta'))
}

function tagsFromMeta(meta: Element | undefined): string[] {
  if (!meta) return []
  const tagsHost = findElement(meta, item => hasId(item, 'tags'))
  if (!tagsHost) return []
  return findElements(tagsHost, item => item.tagName === 'a')
    .map(tag => normalizeText(textContent(tag)))
    .filter(tag => tag.length > 0)
}

export function readGreaterWrongPreviewHtml(
  html: string,
  target: LessWrongTarget,
): LessWrongPreview | undefined {
  const root = fromHtml(html)
  const main = findElement(root, item => hasClass(item, 'post'))
  if (!main) return undefined

  const meta = topMeta(main)
  const title = firstClassText(main, 'post-title') ?? metaContent(root, 'og:title')
  const extract = previewExtract(main) ?? metaContent(root, 'og:description')
  if (!title || !extract) return undefined

  const author = firstClassText(meta ?? main, 'author')
  const dateElement = meta ? findElement(meta, item => hasClass(item, 'date')) : undefined

  return {
    postId: target.postId,
    title,
    pageUrl: lessWrongPostUrl(target),
    extract,
    ...(author ? { author } : {}),
    ...withOptionalString('postedAt', parseDate(dateElement)),
    ...withOptionalNumber('score', parseInteger(firstClassText(meta ?? main, 'karma-value'))),
    ...withOptionalNumber(
      'commentCount',
      parseInteger(firstClassText(meta ?? main, 'comment-count')),
    ),
    ...withOptionalNumber(
      'readTimeMinutes',
      parseInteger(firstClassText(meta ?? main, 'read-time')),
    ),
    ...withOptionalTags(tagsFromMeta(meta)),
    ...withOptionalToc(tocEntries(main)),
  }
}

function withOptionalString(
  key: 'postedAt',
  value: string | undefined,
): Pick<LessWrongPreview, 'postedAt'> {
  return value ? { [key]: value } : {}
}

function withOptionalNumber(
  key: 'score' | 'commentCount' | 'readTimeMinutes',
  value: number | undefined,
): Pick<LessWrongPreview, 'score' | 'commentCount' | 'readTimeMinutes'> {
  return value !== undefined ? { [key]: value } : {}
}

function withOptionalTags(tags: string[]): Pick<LessWrongPreview, 'tags'> {
  return tags.length > 0 ? { tags } : {}
}

function withOptionalToc(toc: PreviewTocEntry[]): Pick<LessWrongPreview, 'toc'> {
  return toc.length > 0 ? { toc } : {}
}
