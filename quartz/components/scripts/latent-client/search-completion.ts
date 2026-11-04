import type { SearchCompletion } from '../../../util/search-completion'
import type { Runtime } from './runtime'
import { completeGoto, gotoIndex, matchDocs, parseGoto } from '../../../util/latent/goto'
import { createSearchCompletions } from '../search-completions'
import { benchFeatures, type FeatureDict } from './features'

/** The shared completion UI owns selection; this adapter supplies latent paths and features. */
export function bindSearchCompletion(rt: Runtime, input: HTMLInputElement, form: HTMLElement) {
  const index = gotoIndex(rt.docs)
  let dictionaries: FeatureDict[] | null = null
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
    const needle = parseGoto(value)
    if (needle !== null) {
      const docs = needle ? matchDocs(index, needle, 12) : rt.docs.slice(0, 12)
      if (docs.some(doc => completeGoto(doc) === value)) return []
      return docs.map(doc => choice('goto:', doc.id, completeGoto(doc), doc.t))
    }
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
    if (dictionaries?.some(dict => dict.table.some(row => value === `#${dict.meta.name}:${row.f}`)))
      return []
    const words = value.slice(1).trim().toLowerCase().split(/\s+/).filter(Boolean)
    return (dictionaries ?? [])
      .flatMap(dict => dict.table.map(row => ({ dict, row })))
      .filter(({ dict, row }) => {
        const text =
          `${dict.meta.name}:${row.f} ${features.name(row)} ${row.about ?? ''} ${row.terms.join(' ')}`.toLowerCase()
        return words.every(word => text.includes(word))
      })
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
