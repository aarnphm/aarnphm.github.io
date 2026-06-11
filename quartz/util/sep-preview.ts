import type { Element, Nodes } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import {
  attributeValue,
  childNodes,
  findElement,
  findElements,
  hasId,
  isElement,
  metaContent,
  metaContents,
  normalizeText,
  textContent,
} from './hast-query'
import { normalizeTocDepth, truncateText, type PreviewTocEntry } from './preview'
import { sepEntryUrl, type SepPreview, type SepTarget } from './sep'

function formatAuthorName(value: string): string {
  const parts = value
    .split(',')
    .map(part => part.trim())
    .filter(part => part.length > 0)
  return parts.length === 2 ? `${parts[1]} ${parts[0]}` : normalizeText(value)
}

function entryTitle(root: Nodes): string | undefined {
  const title = metaContent(root, 'citation_title')
  if (title) return title
  const heading = findElement(root, item => item.tagName === 'h1')
  const text = heading ? normalizeText(textContent(heading)) : ''
  return text.length > 0 ? text : undefined
}

function pubInfoText(root: Nodes): string | undefined {
  const element = findElement(root, item => hasId(item, 'pubinfo'))
  const text = element ? normalizeText(textContent(element)) : ''
  return text.length > 0 ? text : undefined
}

function previewExtract(root: Nodes): string | undefined {
  const preamble = findElement(root, item => hasId(item, 'preamble'))
  if (!preamble) return undefined

  const paragraphs = findElements(preamble, item => item.tagName === 'p')
    .map(paragraph => normalizeText(textContent(paragraph)))
    .filter(text => text.length > 0)
    .slice(0, 3)

  const text = truncateText(paragraphs.join(' '), 640)
  return text.length > 0 ? text : undefined
}

function directChildAnchor(item: Element): Element | undefined {
  for (const child of childNodes(item)) {
    if (isElement(child) && child.tagName === 'a') return child
  }
  return undefined
}

function collectTocEntries(node: Nodes, depth: number, entries: PreviewTocEntry[]) {
  for (const child of childNodes(node)) {
    if (!isElement(child)) continue
    if (child.tagName === 'li') {
      const anchor = directChildAnchor(child)
      const text = anchor ? normalizeText(textContent(anchor)) : ''
      const href = anchor ? attributeValue(anchor, 'href')?.trim() : undefined
      if (text.length > 0 && href?.startsWith('#')) {
        entries.push({ text, href, depth: normalizeTocDepth(depth) })
      }
      collectTocEntries(child, depth + 1, entries)
    } else {
      collectTocEntries(child, depth, entries)
    }
  }
}

function tocEntries(root: Nodes): PreviewTocEntry[] {
  const toc = findElement(root, item => hasId(item, 'toc'))
  if (!toc) return []
  const entries: PreviewTocEntry[] = []
  collectTocEntries(toc, 1, entries)
  return entries.slice(0, 24)
}

export function readSepPreviewHtml(html: string, target: SepTarget): SepPreview | undefined {
  const root = fromHtml(html)
  const title = entryTitle(root)
  const extract = previewExtract(root) ?? metaContent(root, 'og:description')
  if (!title || !extract) return undefined

  const authors = metaContents(root, 'citation_author').map(formatAuthorName)
  const pubInfo = pubInfoText(root)
  const toc = tocEntries(root)

  return {
    entry: target.entry,
    title,
    pageUrl: sepEntryUrl(target),
    extract,
    ...(authors.length > 0 ? { authors } : {}),
    ...(pubInfo ? { pubInfo } : {}),
    ...(toc.length > 0 ? { toc } : {}),
  }
}
