import type { CodeQuery } from '../../../util/code-search-types'
import { escapeHTML } from '../../../util/escape'
import { isFullSlug } from '../../../util/path'
import {
  codeMatchLabel,
  codeSearchRows,
  codeSearchStatus,
  codeSearchSummary,
  codeSearchError,
} from '../code-search-controls'
import { clearPreview, setImageSearch } from './preview'
import { completeSearchProgress, startSearchProgress } from './progress'
import { commitSearchDisplay, displayResults } from './results'
import { shared, type SearchSession, type SimilarityResult } from './state'

export async function searchCode(s: SearchSession, query: CodeQuery, token: number) {
  setImageSearch(s, false)
  clearPreview(s)
  startSearchProgress(s)
  shared.currentSearchTerm = query.pattern
  const current = () => token === s.searchSeq && s.container.classList.contains('active')
  try {
    const result = await s.code.search(query, progress => {
      if (current()) s.results.replaceChildren(codeSearchStatus(progress))
    })
    if (!current()) return
    const entries: SimilarityResult[] = codeSearchRows(result.matches).flatMap((match, id) => {
      if (!isFullSlug(match.unit.ownerSlug)) return []
      return [
        {
          item: {
            id,
            slug: match.unit.ownerSlug,
            title: escapeHTML(match.unit.title),
            content: escapeHTML(`${codeMatchLabel(match)} · ${match.excerpt}`),
            tags: [],
            aliases: [],
            target: '',
            fileName: match.unit.path,
            codeMatch: match,
          },
          similarity: Number.NaN,
        },
      ]
    })
    await commitSearchDisplay(s, token, () => {
      displayResults(s, entries)
      s.results.prepend(codeSearchSummary(result, entries.length, s.container))
    })
  } catch (error) {
    if (current())
      await commitSearchDisplay(s, token, () => {
        s.results.replaceChildren(codeSearchError(error))
      })
  } finally {
    if (current()) completeSearchProgress(s)
  }
}
