import type { PathQuery } from '../../../util/path-query'
import { listPathRows, mountPathResults } from '../path-results'
import {
  ensureContentData,
  hidePalette,
  paletteMode,
  quickOpenProgress,
  quickOpenStatus,
  type PaletteState,
} from './shell'

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
  const rows = listPathRows(data, query)

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
  const { pages, firstRow } = mountPathResults(
    s.output,
    rows,
    query,
    (url, literal) => {
      hidePalette(s)
      if (literal) location.assign(url)
      else window.spaNavigate(url)
    },
    row => {
      s.output.querySelectorAll<HTMLElement>('.suggestion-item.focus').forEach(el => {
        el.classList.remove('focus')
      })
      row.classList.add('focus')
      s.currentHover = row
    },
  )
  s.resultPages = pages
  if (firstRow) {
    firstRow.classList.add('focus')
    s.currentHover = firstRow
  }
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
