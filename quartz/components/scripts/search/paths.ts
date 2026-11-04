import type { PathQuery } from '../../../util/path-query'
import type { SearchSession } from './state'
import { listPathRows, mountPathResults } from '../path-results'
import { searchStatusTable } from '../search-results-table'
import { ensureSearchData } from './engine'
import { clearPreview, displayPreview, setImageSearch } from './preview'
import { commitSearchDisplay } from './results'
import { hideSearch } from './visibility'

export async function searchSitePaths(s: SearchSession, query: PathQuery, token: number) {
  setImageSearch(s, false)
  let data: ContentIndex
  try {
    data = await ensureSearchData(s)
  } catch (error) {
    console.warn('[Search] path index unavailable:', error)
    data = {} as ContentIndex
  }
  if (token !== s.searchSeq) return
  const rows = listPathRows(data, query)
  await commitSearchDisplay(s, token, () => {
    clearPreview(s)
    s.resultPages?.dispose()
    s.resultPages = null
    s.currentHover = null
    if (!rows.length) {
      s.results.replaceChildren(searchStatusTable('', 'notes', 'no pages here'))
      return
    }
    const { pages, firstLink } = mountPathResults(
      s.results,
      rows,
      query,
      (url, literal) => {
        hideSearch(s)
        if (literal) location.assign(url)
        else window.spaNavigate(url)
      },
      (_row, link) => {
        s.currentHover?.classList.remove('focus')
        link.classList.add('focus')
        s.currentHover = link
        if (link.id) void displayPreview(s, link)
        else clearPreview(s)
      },
    )
    s.resultPages = pages
    if (firstLink) {
      firstLink.classList.add('focus')
      s.currentHover = firstLink
    }
  })
}
