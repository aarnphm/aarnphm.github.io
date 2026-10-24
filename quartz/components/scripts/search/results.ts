import { highlight } from '../../../util/search-text'
import { codeMatchLocation, codeMatchUrl } from '../code-search-controls'
import { imageSearchRow, searchableImage } from '../image-search-controls'
import { fitSearchPath } from '../search-path'
import { mountResultPages } from '../search-result-pages'
import { clearPreview, displayPreview, resolveUrl } from './preview'
import { shared, type SearchSession, type SimilarityResult } from './state'
import { hideSearch } from './visibility'

function fileExtension(fileName: string): string | null {
  const name = fileName.split('/').at(-1) ?? fileName
  const index = name.lastIndexOf('.')
  if (index <= 0 || index === name.length - 1) {
    return null
  }

  return name.slice(index).toLowerCase()
}

function isDirectContentFile(fileName: string): boolean {
  const ext = fileExtension(fileName)
  return ext !== null && ext !== '.canvas' && ext !== '.base' && ext !== '.bases'
}

export function fitResultPaths(s: SearchSession) {
  for (const label of s.results.querySelectorAll<HTMLElement>('.search-result-path')) {
    fitSearchPath(label, shared.currentSearchTerm)
  }
}

export function commitSearchDisplay(
  s: SearchSession,
  token: number,
  apply: () => void | Promise<void>,
) {
  if (token !== s.searchSeq) return
  if (s.pressedKeys.size > 0 || s.composing) {
    s.pendingDisplay = { token, apply }
    return
  }
  s.pendingDisplay = null
  s.searchLayout.classList.toggle('display-results', shared.rawSearchTerm.trim() !== '')
  s.searchLayout.removeAttribute('aria-busy')
  const applied = apply()
  if (s.pendingOpen === token) {
    s.pendingOpen = null
    openSearchResult(s)
  }
  return applied
}

export function flushSearchDisplay(s: SearchSession) {
  if (s.pressedKeys.size > 0 || s.composing) return
  const pending = s.pendingDisplay
  s.pendingDisplay = null
  if (pending) void commitSearchDisplay(s, pending.token, pending.apply)
}

function navigateSearchResult(s: SearchSession, anchor: HTMLAnchorElement) {
  const href = anchor.getAttribute('href')
  if (!href) return
  window.spaNavigate(new URL(href, window.location.toString()))
  hideSearch(s)
}

export function openSearchResult(s: SearchSession) {
  const { searchLayout, results } = s
  if (searchLayout.getAttribute('aria-busy') === 'true') {
    s.pendingOpen = s.searchSeq
    return
  }
  const focused = document.activeElement
  const anchor =
    focused instanceof HTMLAnchorElement && results.contains(focused)
      ? focused
      : results.querySelector<HTMLAnchorElement>('a.result-card[href]:not(.no-match)')
  if (anchor?.hasAttribute('href')) navigateSearchResult(s, anchor)
}

const resultToHTML = (
  s: SearchSession,
  { item, similarity }: SimilarityResult,
  signal: AbortSignal,
) => {
  const { slug, title, target, fileName } = item
  const isProtected = item.protected === true
  const isImage = searchableImage(slug)
  const itemTile = document.createElement('a')
  const titleContent = isImage
    ? highlight(shared.currentSearchTerm, (fileName || slug).split('/').at(-1) ?? slug)
    : target
      ? highlight(shared.currentSearchTerm, target)
      : title
  itemTile.classList.add('result-card')
  itemTile.id = slug
  itemTile.href = resolveUrl(s, slug).toString()
  if (item.passage) {
    itemTile.hash = `page=${item.passage.page}`
    itemTile.dataset.pdfPage = String(item.passage.page)
  }

  if (item.codeMatch) {
    itemTile.href = codeMatchUrl(s.currentSlug, item.codeMatch).toString()
    itemTile.id = `${slug}--code-${item.id}`
    itemTile.dataset.codeOwner = slug
    itemTile.dataset.codeUnit = item.codeMatch.unit.id
    s.codeMatches.set(itemTile, item.codeMatch)
  }
  const fileData = s.data?.[slug]
  const resolvedFileName = (fileData?.fileName || fileName || slug || '').toLowerCase()
  const extension = fileExtension(resolvedFileName)
  const isCanvas = extension === '.canvas'
  const isBases = extension === '.base' || extension === '.bases'
  const isDirectFile = isDirectContentFile(resolvedFileName)
  if (isCanvas) itemTile.dataset.canvas = 'true'
  if (isBases) itemTile.dataset.bases = 'true'
  if (isDirectFile) itemTile.dataset.directFile = 'true'

  if (isProtected) itemTile.dataset.protected = 'true'
  itemTile.innerHTML = `<hgroup><h3>${titleContent}${isProtected ? ' <span aria-label="protected content">🔒</span>' : ''}</h3></hgroup>`

  if (item.codeMatch) {
    const match = item.codeMatch
    const location = document.createElement('span')
    location.className = 'code-search-mobile-location'
    const ordinal = match.ranges.length
      ? ` · match ${match.selectedRange + 1}/${match.ranges.length}`
      : ''
    location.textContent = `${codeMatchLocation(match)}${ordinal}`
    itemTile.querySelector('hgroup')?.append(location)
    itemTile.setAttribute('aria-label', itemTile.textContent?.trim() ?? '')
  }

  const accessibleName = itemTile.textContent?.trim() ?? ''
  for (const label of itemTile.querySelectorAll<HTMLElement>('h3')) {
    const path = label.textContent ?? ''
    if (!path.includes('/')) continue
    label.classList.add('search-result-path')
    label.title = path
    itemTile.setAttribute('aria-label', accessibleName)
  }

  if (Number.isFinite(similarity)) itemTile.dataset.similarity = String(similarity)

  const handler = (evt: MouseEvent) => {
    if (evt.altKey || evt.ctrlKey || evt.metaKey || evt.shiftKey) return
    const anchor = evt.currentTarget as HTMLAnchorElement | null
    if (!anchor) return
    evt.preventDefault()
    navigateSearchResult(s, anchor)
  }

  async function onMouseEnter(ev: MouseEvent) {
    if (!ev.target) return
    const target = ev.target as HTMLInputElement
    await displayPreview(s, target)
  }

  itemTile.addEventListener('mouseenter', onMouseEnter, { signal })
  itemTile.addEventListener('click', handler, { signal })

  return isImage ? imageSearchRow(itemTile, slug, s.showSimilarImages, similarity) : itemTile
}

export function displayResults(s: SearchSession, finalResults: SimilarityResult[]) {
  const { results, preview } = s
  const imagesOnly =
    finalResults.length > 0 && finalResults.every(({ item }) => searchableImage(item.slug))
  const navigationHint = s.searchSpace.querySelector('#helper li:first-child kbd')
  if (navigationHint) navigationHint.textContent = imagesOnly ? '←↑↓→' : '↑↓'
  s.resultPages?.dispose()
  s.resultPages = null
  results.replaceChildren()
  results.scrollTop = 0
  if (finalResults.length === 0) {
    results.innerHTML = `<a class="result-card no-match">
        <h3>No results.</h3>
        <p>Try another search term?</p>
    </a>`
    s.currentHover = null
  } else if (imagesOnly) {
    const gallery = document.createElement('div')
    gallery.className = 'image-search-grid'
    results.append(gallery)
    s.resultPages = mountResultPages(results, finalResults, (page, signal) => {
      const cards = page.map(result => resultToHTML(s, result, signal))
      gallery.append(...cards)
      return cards[0]?.querySelector<HTMLElement>('a.result-card') ?? null
    })
  } else {
    const scored = finalResults.some(result => Number.isFinite(result.similarity))
    const table = document.createElement('table')
    table.className = 'search-results-table'
    table.setAttribute('aria-label', 'search results')
    const head = table.createTHead().insertRow()
    for (const [label, column] of [
      ['note', 'note'],
      ['match', 'match'],
      ...(scored ? [['score', 'score']] : []),
    ]) {
      const heading = document.createElement('th')
      heading.scope = 'col'
      heading.className = `search-column-${column}`
      heading.textContent = label
      head.append(heading)
    }
    const body = table.createTBody()
    results.append(table)
    s.resultPages = mountResultPages(results, finalResults, (page, signal) => {
      let first: HTMLElement | null = null
      for (const result of page) {
        const row = body.insertRow()
        const note = row.insertCell()
        note.className = 'search-column-note'
        note.append(resultToHTML(s, result, signal))
        const match = row.insertCell()
        match.className = 'search-column-match'
        const excerpt = document.createElement('span')
        excerpt.innerHTML = result.item.protected ? 'protected content' : result.item.content
        match.append(excerpt)
        if (scored) {
          const score = row.insertCell()
          score.className = 'search-column-score'
          score.textContent = Number.isFinite(result.similarity)
            ? result.similarity.toFixed(3)
            : '–'
        }
        const anchor = note.querySelector<HTMLAnchorElement>('a.result-card')
        if (anchor) {
          first ??= anchor
          row.addEventListener('mouseenter', () => void displayPreview(s, anchor), { signal })
          row.addEventListener(
            'click',
            event => {
              if (event.target instanceof Element && event.target.closest('a, button')) return
              if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
              anchor.click()
            },
            { signal },
          )
        }
      }
      fitResultPaths(s)
      return first
    })
  }

  if (finalResults.length === 0 && preview) {
    clearPreview(s)
  } else {
    const firstChild = results.querySelector<HTMLElement>('.result-card')
    if (!firstChild) return
    firstChild.classList.add('focus')
    s.currentHover = firstChild
    void displayPreview(s, firstChild)
  }
}
