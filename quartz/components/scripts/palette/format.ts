import { fetchCanonical } from '../../../util/fetch-canonical'
import { FullSlug, normalizeRelativeURLs, resolveRelative } from '../../../util/path'
import { tokenizeTerm } from '../../../util/search-text'

export function appendHighlightedText(parent: HTMLElement, searchTerm: string, text: string) {
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

export function appendActionAux(parent: HTMLElement, auxInnerHtml: string) {
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

export function appendTrustedSvg(parent: HTMLElement, svgMarkup: string) {
  const template = document.createElement('template')
  template.innerHTML = svgMarkup
  const svg = template.content.firstElementChild
  if (svg) parent.append(svg)
}

const p = new DOMParser()
const fetchContentCache: Map<FullSlug, HTMLElement[]> = new Map()
export async function fetchContent(currentSlug: FullSlug, slug: FullSlug): Promise<HTMLElement[]> {
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
