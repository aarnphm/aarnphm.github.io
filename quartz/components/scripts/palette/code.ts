import type { CodeQuery } from '../../../util/code-search-types'
import {
  codeMatchLabel,
  codeMatchUrl,
  codeSearchRows,
  codeSearchStatus,
  codeSearchSummary,
  codeSearchError,
} from '../code-search-controls'
import { renderCodeMatch } from '../code-search-preview'
import { updateSearchProgress } from '../search-progress'
import { showSearchProgress } from '../search-progress'
import { mountResultPages } from '../search-result-pages'
import { searchStatusTable } from '../search-results-table'
import { hidePalette, paletteMode, quickOpenProgress, type PaletteState } from './shell'

export async function searchPaletteCode(s: PaletteState, query: CodeQuery, generation: number) {
  const current = () =>
    generation === s.queryGeneration && s.isActive && paletteMode.actionType === 'quick_open'
  quickOpenProgress(s, true)
  s.output.setAttribute('aria-busy', 'true')
  try {
    const result = await s.codeSearch.search(query, progress => {
      if (current())
        updateSearchProgress(
          s.progress,
          codeSearchStatus(progress),
          progress.scannedUnits,
          progress.totalUnits,
        )
    })
    if (!current()) return
    s.resultPages?.dispose()
    const rows = codeSearchRows(result.matches)
    s.output.replaceChildren(codeSearchSummary(result, rows.length, s.container))
    s.currentHover = null
    if (!rows.length) {
      s.output.append(searchStatusTable(query.pattern, 'code'))
    }
    s.resultPages = mountResultPages(s.output, rows, (page, signal) => {
      const items = page.map(match => {
        const item = document.createElement('div')
        item.className = 'suggestion-item code-search-result'
        item.tabIndex = -1
        item.dataset.slug = match.unit.ownerSlug
        item.dataset.codeUnit = match.unit.id
        const content = document.createElement('a')
        content.className = 'suggestion-content'
        content.href = codeMatchUrl(s.currentSlug, match).toString()
        const title = document.createElement('div')
        title.className = 'suggestion-title'
        title.textContent = match.unit.title
        const detail = document.createElement('div')
        detail.className = 'code-search-detail'
        detail.textContent = codeMatchLabel(match)
        content.append(title, detail)
        item.append(content)
        const select = () => {
          s.currentHover?.classList.remove('focus')
          s.currentHover = item
          item.classList.add('focus')
          s.codePreview?.abort()
          s.output.querySelector('.code-search-inline-preview')?.remove()
          const request = new AbortController()
          s.codePreview = request
          showSearchProgress(s.progress, true, 'preview')
          const completePreview = () => {
            if (s.codePreview === request && current()) quickOpenProgress(s, false)
          }
          request.signal.addEventListener('abort', completePreview, { once: true })
          signal.addEventListener('abort', () => request.abort(), { once: true })
          const preview = document.createElement('div')
          preview.className = 'code-search-inline-preview'
          // Focus stays on the result; Ctrl+K owns the interactive full preview.
          preview.inert = true
          item.append(preview)
          void renderCodeMatch(s.currentSlug, match, preview, request.signal, true)
            .catch(error => {
              if (!request.signal.aborted) preview.replaceChildren(codeSearchError(error))
            })
            .finally(completePreview)
        }
        item.addEventListener('focus', select, { signal })
        item.addEventListener('mouseenter', select, { signal })
        item.addEventListener(
          'click',
          event => {
            if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
            event.preventDefault()
            const url = new URL(content.href)
            hidePalette(s)
            window.spaNavigate(url)
          },
          { signal },
        )
        return item
      })
      s.output.append(...items)
      return items[0] ?? null
    })
    const first = s.output.querySelector<HTMLElement>('.suggestion-item')
    if (first) {
      first.classList.add('focus')
      s.currentHover = first
    }
  } catch (error) {
    if (current()) s.output.replaceChildren(codeSearchError(error))
  } finally {
    if (current()) {
      quickOpenProgress(s, false)
      s.output.removeAttribute('aria-busy')
    }
  }
}
