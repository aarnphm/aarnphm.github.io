import {
  buildPathTree,
  completePath,
  matchPaths,
  type PathEntry,
  type PathQuery,
  type PathTree,
} from '../../../util/path-query'
import { isSearchExcluded } from '../../../util/search-exclude'
import { mountResultPages } from '../search-result-pages'
import { searchResultsTable } from '../search-results-table'
import { appendHighlightedText } from './format'
import {
  ensureContentData,
  hidePalette,
  paletteMode,
  quickOpenProgress,
  quickOpenStatus,
  setFocusFirstChild,
  type PaletteState,
} from './shell'

type PathRow = { entry: PathEntry; literal?: boolean }

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
export async function searchPaths(s: PaletteState, query: PathQuery, generation: number) {
  let data: ContentIndex
  try {
    data = await ensureContentData(s)
  } catch (error) {
    console.warn('[QuickOpen] path index unavailable:', error)
    data = {} as ContentIndex
  }
  if (generation !== s.queryGeneration || paletteMode.actionType !== 'quick_open' || !s.isActive)
    return
  const { exact, items } = matchPaths(pathTree(data), query.path)
  const typed = query.path.replace(/\/+$/, '')
  const leaf = query.path.endsWith('/') ? '' : typed.slice(typed.lastIndexOf('/') + 1)
  // Search-excluded routes (/embeddings, /hinterland) answer typed letters, never a bare listing.
  const browsing = !leaf && !isSearchExcluded(typed)
  const rows: PathRow[] = exact ? [{ entry: exact }] : []
  for (const entry of items) {
    if (!browsing || !isSearchExcluded(entry.path)) rows.push({ entry })
  }
  if (!exact && typed) rows.push({ entry: { path: typed, pages: 0 }, literal: true })

  s.output.removeAttribute('aria-busy')
  quickOpenProgress(s, false)
  if (!rows.length) {
    quickOpenStatus(s, 'no pages here')
    return
  }
  s.resultPages?.dispose()
  s.resultPages = null
  s.output.replaceChildren()
  s.currentHover = null
  const { table, body } = searchResultsTable(['path', 'pages'], 'paths')
  s.output.append(table)
  s.resultPages = mountResultPages(s.output, rows, (page, signal) => {
    const nodes = page.map(row => pathRow(s, row, query, leaf, signal))
    body.append(...nodes)
    return nodes[0] ?? null
  })
  s.output.scrollTop = 0
  setFocusFirstChild(s)
}

function pathRow(
  s: PaletteState,
  { entry, literal }: PathRow,
  query: PathQuery,
  leaf: string,
  signal: AbortSignal,
) {
  const item = document.createElement('tr')
  item.classList.add('suggestion-item', 'suggestion-path')
  item.tabIndex = -1
  item.dataset.path = completePath(entry, query.suffix)
  if (entry.slug) item.dataset.slug = entry.slug

  const content = document.createElement('a')
  // Typed paths are absolute, so resolve from the origin: relative resolution breaks on
  // pages served away from their slug (the 404 page renders at any missing url).
  const folder = entry.path !== '' && entry.slug?.endsWith('/index') === true
  content.href = new URL(`/${entry.path}${folder ? '/' : ''}`, location.origin).toString()
  if (query.suffix) {
    const destination = new URL(query.suffix, content.href)
    content.href = destination.toString()
  }
  content.classList.add('suggestion-content')
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
      const destination = new URL(content.href)
      hidePalette(s)
      // Routes outside the content index (workers, files) skip the SPA fetch.
      if (literal) location.assign(destination)
      else window.spaNavigate(destination)
    },
    { signal },
  )
  const hover = () => {
    s.output.querySelectorAll<HTMLElement>('.suggestion-item.focus').forEach(el => {
      el.classList.remove('focus')
    })
    item.classList.add('focus')
    s.currentHover = item
  }
  item.addEventListener('mouseenter', hover, { signal })
  item.addEventListener('focus', hover, { signal })
  return item
}

/** Tab on a row: write its path into the bar, descending into folders. */
export function completeFocusedPath(s: PaletteState) {
  const focused = s.currentHover ?? s.output.querySelector<HTMLElement>('.suggestion-item.focus')
  const path = focused?.dataset.path
  if (!path) return false
  s.bar.value = path
  s.bar.setSelectionRange(path.length, path.length)
  paletteMode.currentSearchTerm = path
  return true
}
