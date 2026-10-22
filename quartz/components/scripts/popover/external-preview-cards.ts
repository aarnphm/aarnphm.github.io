import type { LessWrongPreview } from '../../../util/lesswrong'
import type { PreviewTocEntry } from '../../../util/preview'
import type { SepPreview } from '../../../util/sep'
import type { WikipediaPreview } from '../../../util/wikipedia'

export function renderWikipediaPreview(preview: WikipediaPreview, popoverInner: HTMLDivElement) {
  popoverInner.dataset.contentType = 'text/x-wikipedia'

  const card = document.createElement('article')
  card.classList.add('wikipedia-popover-card')
  if (preview.thumbnail) card.classList.add('has-thumbnail')

  const header = document.createElement('header')
  header.classList.add('wikipedia-popover-header')

  const heading = document.createElement('ul')
  heading.classList.add('wikipedia-popover-heading')

  const title = document.createElement('li')
  title.classList.add('wikipedia-popover-title')

  const titleLink = document.createElement('a')
  titleLink.href = preview.pageUrl
  titleLink.target = '_blank'
  titleLink.rel = 'noopener noreferrer'
  titleLink.textContent = preview.title
  title.appendChild(titleLink)
  heading.appendChild(title)

  if (preview.description) {
    const description = document.createElement('li')
    description.classList.add('wikipedia-popover-description')
    description.textContent = preview.description
    heading.appendChild(description)
  }

  header.appendChild(heading)
  card.appendChild(header)

  if (preview.thumbnail) {
    const thumbnail = document.createElement('img')
    thumbnail.classList.add('wikipedia-popover-thumbnail')
    thumbnail.src = preview.thumbnail.source
    thumbnail.alt = preview.title
    thumbnail.decoding = 'async'
    if (preview.thumbnail.width) thumbnail.width = preview.thumbnail.width
    if (preview.thumbnail.height) thumbnail.height = preview.thumbnail.height
    card.appendChild(thumbnail)
  }

  const extract = document.createElement('p')
  extract.classList.add('wikipedia-popover-extract')
  extract.textContent = preview.extract
  card.appendChild(extract)

  const source = document.createElement('a')
  source.classList.add('wikipedia-popover-source')
  source.href = preview.pageUrl
  source.target = '_blank'
  source.rel = 'noopener noreferrer'
  source.textContent = 'Wikipedia'
  card.appendChild(source)

  popoverInner.appendChild(card)
}

function appendTextItem(parent: HTMLElement, text: string, className?: string) {
  const item = document.createElement('li')
  if (className) item.classList.add(className)
  item.textContent = text
  parent.appendChild(item)
}

function appendLessWrongTag(parent: HTMLElement, text: string) {
  const item = document.createElement('li')
  const label = document.createElement('span')
  label.classList.add('internal', 'tag-link', 'lesswrong-popover-tag')
  label.textContent = text
  item.appendChild(label)
  parent.appendChild(item)
}

function pluralized(value: number, singular: string, plural: string): string {
  return `${value} ${Math.abs(value) === 1 ? singular : plural}`
}

function formatLessWrongDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  }).format(date)
}

function popoverTocHref(pageUrl: string, href: string): string {
  try {
    return new URL(href, pageUrl).toString()
  } catch {
    return pageUrl
  }
}

function appendPopoverToc(
  parent: HTMLElement,
  pageUrl: string,
  toc: PreviewTocEntry[] | undefined,
  ariaLabel: string,
) {
  if (!toc || toc.length === 0) return

  const section = document.createElement('section')
  section.classList.add('popover-toc')
  section.ariaLabel = ariaLabel

  const label = document.createElement('div')
  label.classList.add('popover-toc-title')
  label.textContent = 'Contents'
  section.appendChild(label)

  const list = document.createElement('ol')
  list.classList.add('popover-toc-list')

  for (const entry of toc) {
    const item = document.createElement('li')
    item.classList.add('popover-toc-item')
    item.style.setProperty('--popover-toc-indent', `${(entry.depth - 1) * 0.72}rem`)

    const link = document.createElement('a')
    link.classList.add('popover-toc-link')
    link.href = popoverTocHref(pageUrl, entry.href)
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    link.textContent = entry.text
    item.appendChild(link)
    list.appendChild(item)
  }

  section.appendChild(list)
  parent.appendChild(section)
}

export function renderLessWrongPreview(preview: LessWrongPreview, popoverInner: HTMLDivElement) {
  popoverInner.dataset.contentType = 'text/x-lesswrong'

  const card = document.createElement('article')
  card.classList.add('lesswrong-popover-card')

  const previewHeader = document.createElement('div')
  previewHeader.classList.add('lesswrong-popover-header')

  const title = document.createElement('h2')
  title.classList.add('lesswrong-popover-title')

  const titleLink = document.createElement('a')
  titleLink.href = preview.pageUrl
  titleLink.target = '_blank'
  titleLink.rel = 'noopener noreferrer'
  titleLink.textContent = preview.title
  title.appendChild(titleLink)
  previewHeader.appendChild(title)

  const meta = document.createElement('ul')
  meta.classList.add('lesswrong-popover-meta')
  meta.ariaLabel = 'LessWrong metadata'
  if (preview.score !== undefined)
    appendTextItem(meta, pluralized(preview.score, 'point', 'points'))
  if (preview.author) appendTextItem(meta, preview.author, 'author')
  if (preview.postedAt) appendTextItem(meta, formatLessWrongDate(preview.postedAt))
  if (preview.commentCount !== undefined) {
    appendTextItem(meta, pluralized(preview.commentCount, 'comment', 'comments'))
  }
  if (preview.readTimeMinutes !== undefined) {
    appendTextItem(meta, pluralized(preview.readTimeMinutes, 'min read', 'min read'))
  }
  if (meta.childElementCount > 0) previewHeader.appendChild(meta)

  if (preview.tags && preview.tags.length > 0) {
    const tags = document.createElement('ul')
    tags.classList.add('lesswrong-popover-tags', 'tags')
    tags.ariaLabel = 'LessWrong tags'
    for (const tag of preview.tags) appendLessWrongTag(tags, tag)
    previewHeader.appendChild(tags)
  }

  card.appendChild(previewHeader)
  appendPopoverToc(card, preview.pageUrl, preview.toc, 'LessWrong contents')

  const extract = document.createElement('p')
  extract.classList.add('lesswrong-popover-extract')
  extract.textContent = preview.extract
  card.appendChild(extract)

  const source = document.createElement('a')
  source.classList.add('lesswrong-popover-source')
  source.href = preview.pageUrl
  source.target = '_blank'
  source.rel = 'noopener noreferrer'
  source.textContent = 'LessWrong'
  card.appendChild(source)

  popoverInner.appendChild(card)
}

export function renderSepPreview(preview: SepPreview, popoverInner: HTMLDivElement) {
  popoverInner.dataset.contentType = 'text/x-sep'

  const card = document.createElement('article')
  card.classList.add('sep-popover-card')

  const previewHeader = document.createElement('div')
  previewHeader.classList.add('sep-popover-header')

  const title = document.createElement('h2')
  title.classList.add('sep-popover-title')

  const titleLink = document.createElement('a')
  titleLink.href = preview.pageUrl
  titleLink.target = '_blank'
  titleLink.rel = 'noopener noreferrer'
  titleLink.textContent = preview.title
  title.appendChild(titleLink)
  previewHeader.appendChild(title)

  const meta = document.createElement('ul')
  meta.classList.add('sep-popover-meta')
  meta.ariaLabel = 'SEP metadata'
  for (const author of preview.authors ?? []) appendTextItem(meta, author, 'author')
  if (preview.pubInfo) appendTextItem(meta, preview.pubInfo)
  if (meta.childElementCount > 0) previewHeader.appendChild(meta)

  card.appendChild(previewHeader)
  appendPopoverToc(card, preview.pageUrl, preview.toc, 'SEP contents')

  const extract = document.createElement('p')
  extract.classList.add('sep-popover-extract')
  extract.textContent = preview.extract
  card.appendChild(extract)

  const source = document.createElement('a')
  source.classList.add('sep-popover-source')
  source.href = preview.pageUrl
  source.target = '_blank'
  source.rel = 'noopener noreferrer'
  source.textContent = 'Stanford Encyclopedia of Philosophy'
  card.appendChild(source)

  popoverInner.appendChild(card)
}
