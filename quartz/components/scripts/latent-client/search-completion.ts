import type { LatentKind } from '../../../util/latent/types'
import type { SearchCompletion } from '../../../util/search-completion'
import type { Runtime } from './runtime'
import { filterCompletions, parseFilter } from '../../../util/latent/filter'
import { parseGoto } from '../../../util/latent/goto'
import { createSearchCompletions } from '../search-completions'
import { benchFeatures, type FeatureDict } from './features'

/** The shared completion UI owns selection; this adapter supplies latent paths and features. */
export function bindSearchCompletion(rt: Runtime, input: HTMLInputElement, form: HTMLElement) {
  const counts: Partial<Record<LatentKind, number>> = {}
  for (const doc of rt.docs) counts[doc.s] = (counts[doc.s] ?? 0) + 1
  let dictionaries: FeatureDict[] | null = null
  let haystack:
    | { dict: FeatureDict; row: FeatureDict['table'][number]; id: string; text: string }[]
    | null = null
  let ids: Set<string> | null = null
  let loading = false
  const source = (value: string): SearchCompletion[] => {
    const choice = (operator: string, label: string, insert: string, description = '') => ({
      operator,
      value: label,
      description,
      insert,
      from: 0,
      to: value.length,
    })
    const kinds = filterCompletions(value, counts)
    if (kinds.length)
      return kinds.map(c => ({
        operator: 'filter:',
        value: c.kind,
        description: `${c.count.toLocaleString('en-US')} documents`,
        insert: c.insert,
        from: c.from,
        to: c.to,
      }))
    // `goto:` lists its documents in the links pane, which owns ↑↓, Enter and Tab (search.ts
    // gotoMode); a second list here would take those keys from it
    if (parseGoto(parseFilter(value).rest) !== null) return []
    if (!value.startsWith('#')) return []
    const features = benchFeatures(rt)
    if (!features) return []
    if (!dictionaries && !loading) {
      loading = true
      void Promise.all(features.names.map(features.dict))
        .then(result => {
          dictionaries = result
          if (!rt.signal.aborted && document.activeElement === input) completion.update()
        })
        .catch(error => {
          if (!rt.signal.aborted) console.warn('[latent] feature completions', error)
        })
        .finally(() => {
          loading = false
        })
    }
    if (!dictionaries) return []
    // every row's id and searchable text, lowercased once rather than on every key
    haystack ??= dictionaries.flatMap(dict =>
      dict.table.map(row => ({
        dict,
        row,
        id: `#${dict.meta.name}:${row.f}`,
        text: `${dict.meta.name}:${row.f} ${features.name(row)} ${row.about ?? ''} ${row.terms.join(' ')}`.toLowerCase(),
      })),
    )
    ids ??= new Set(haystack.map(entry => entry.id))
    if (ids.has(value)) return []
    const words = value.slice(1).trim().toLowerCase().split(/\s+/).filter(Boolean)
    return haystack
      .filter(({ text }) => words.every(word => text.includes(word)))
      .sort((a, b) => b.row.n - a.row.n)
      .slice(0, 12)
      .filter(({ dict, row }) => value !== `#${dict.meta.name}:${row.f}`)
      .map(({ dict, row }) =>
        choice(
          '#',
          features.name(row),
          `#${dict.meta.name}:${row.f}`,
          `${dict.meta.name}:${row.f}`,
        ),
      )
  }
  const completion = createSearchCompletions(input, form, () => true, { source })
  rt.signal.addEventListener('abort', completion.dispose, { once: true })
  input.addEventListener('input', completion.update, { signal: rt.signal })
  return completion
}
