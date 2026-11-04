import {
  buildPathTree,
  completePath,
  matchPaths,
  type PathEntry,
  type PathQuery,
  type PathTree,
} from '../../util/path-query'
import { isSearchExcluded } from '../../util/search-exclude'
import { appendHighlightedText } from './palette/format'
import { mountResultPages, type ResultPages } from './search-result-pages'
import { searchResultsTable } from './search-results-table'

export type PathRow = { entry: PathEntry; literal?: boolean }

const trees = new WeakMap<ContentIndex, PathTree>()

function pathTree(data: ContentIndex) {
  let tree = trees.get(data)
  if (!tree) {
    tree = buildPathTree(
      Object.entries(data).flatMap(([slug, note]) =>
        note.protected ? [] : [{ slug, title: note.title || note.fileName }],
      ),
    )
    trees.set(data, tree)
  }
  return tree
}

/** `/embeddings` opens that route; `/thoughts/` lists what sits under it. */
export function listPathRows(data: ContentIndex, query: PathQuery): PathRow[] {
  const { exact, items } = matchPaths(pathTree(data), query.path)
  const typed = query.path.replace(/\/+$/, '')
  const browsing = !leafOf(query.path) && !isSearchExcluded(typed)
  const rows: PathRow[] = exact ? [{ entry: exact }] : []
  for (const entry of items) {
    if (!browsing || !isSearchExcluded(entry.path)) rows.push({ entry })
  }
  if (!exact && typed) rows.push({ entry: { path: typed, pages: 0 }, literal: true })
  return rows
}

function leafOf(path: string) {
  const typed = path.replace(/\/+$/, '')
  return path.endsWith('/') ? '' : typed.slice(typed.lastIndexOf('/') + 1)
}

export function mountPathResults(
  container: HTMLElement,
  rows: readonly PathRow[],
  query: PathQuery,
  navigate: (url: URL, literal: boolean) => void,
  onFocus?: (row: HTMLTableRowElement, link: HTMLAnchorElement) => void,
): {
  pages: ResultPages
  firstRow: HTMLTableRowElement | null
  firstLink: HTMLAnchorElement | null
} {
  container.replaceChildren()
  const { table, body } = searchResultsTable(['path', 'pages'], 'paths')
  container.append(table)
  let firstRow: HTMLTableRowElement | null = null
  let firstLink: HTMLAnchorElement | null = null
  const leaf = leafOf(query.path)
  const pages = mountResultPages(container, rows, (page, signal) => {
    const nodes = page.map(row => pathRow(row, query, leaf, navigate, signal, onFocus))
    body.append(...nodes)
    firstRow ??= nodes[0] ?? null
    const link = nodes[0]?.querySelector('a.result-card')
    if (firstLink === null && link instanceof HTMLAnchorElement) firstLink = link
    return nodes[0] ?? null
  })
  container.scrollTop = 0
  return { pages, firstRow, firstLink }
}

function pathRow(
  { entry, literal }: PathRow,
  query: PathQuery,
  leaf: string,
  navigate: (url: URL, literal: boolean) => void,
  signal: AbortSignal,
  onFocus?: (row: HTMLTableRowElement, link: HTMLAnchorElement) => void,
) {
  const item = document.createElement('tr')
  item.classList.add('suggestion-item', 'suggestion-path')
  item.tabIndex = -1
  const path = completePath(entry, query.suffix)
  item.dataset.path = path
  if (literal) item.dataset.pathLiteral = 'true'
  if (entry.slug) item.dataset.slug = entry.slug

  const content = document.createElement('a')
  const folder = entry.path !== '' && entry.slug?.endsWith('/index') === true
  content.href = new URL(`/${entry.path}${folder ? '/' : ''}`, location.origin).toString()
  if (query.suffix) content.href = new URL(query.suffix, content.href).toString()
  content.classList.add('suggestion-content', 'result-card')
  content.dataset.path = path
  if (literal) content.dataset.pathLiteral = 'true'
  if (entry.slug) content.id = entry.slug
  const title = document.createElement('div')
  title.classList.add('suggestion-title')
  const route = document.createElement('span')
  route.className = 'suggestion-route'
  appendHighlightedText(route, leaf, `/${entry.path}`)
  title.append(route)
  const name = literal ? 'open this url' : entry.title
  if (name && name !== entry.path) {
    const label = document.createElement('span')
    label.className = 'suggestion-route-title'
    label.textContent = name
    title.append(label)
  }
  content.append(title)

  const aux = document.createElement('div')
  aux.classList.add('suggestion-aux')
  if (entry.pages > 0) {
    const count = document.createElement('span')
    count.textContent = `${entry.pages} ${entry.pages === 1 ? 'page' : 'pages'}`
    aux.append(count)
  }

  const noteCell = item.insertCell()
  noteCell.className = 'search-column-note'
  noteCell.append(content)
  const countCell = item.insertCell()
  countCell.className = 'search-column-score'
  countCell.append(aux)

  item.addEventListener(
    'click',
    (event: MouseEvent) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
      event.preventDefault()
      navigate(new URL(content.href), literal === true)
    },
    { signal },
  )
  const hover = () => onFocus?.(item, content)
  item.addEventListener('mouseenter', hover, { signal })
  item.addEventListener('focus', hover, { signal })
  content.addEventListener('focus', hover, { signal })
  return item
}
