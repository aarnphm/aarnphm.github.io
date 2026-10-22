import type { Completion, CompletionContext } from '@codemirror/autocomplete'
import type { LeanLspCompletionItem } from '../../lean/kernel'
import { leanLspConfigFacet } from './editor-context'
import {
  EMPTY_INDEXED_ENTRIES,
  getDocumentSymbolsForSource,
  getDocumentSymbolsForText,
  STATIC_BY_NAMESPACE,
  STATIC_COMMANDS,
  STATIC_NON_COMMANDS,
  type IndexedCatalogEntry,
} from './symbol-index'

function lspKindToCompletionType(kind: number | undefined): string {
  switch (kind) {
    case 2:
    case 3:
      return 'function'
    case 6:
      return 'variable'
    case 7:
    case 8:
    case 22:
      return 'type'
    case 14:
      return 'keyword'
    case 21:
      return 'constant'
    default:
      return 'function'
  }
}

const staticCompletionCache = new Map<string, readonly Completion[]>()
const MAX_STATIC_COMPLETION_CACHE = 128

export function matchIndexedCatalogCandidates(
  prefix: string,
  localIndexed: readonly IndexedCatalogEntry[],
  cacheMap?: Map<string, readonly Completion[]>,
): readonly Completion[] {
  const query = prefix.trim()
  const targetCache = cacheMap ?? (localIndexed.length === 0 ? staticCompletionCache : undefined)
  if (targetCache) {
    const cached = targetCache.get(query)
    if (cached !== undefined) return cached
  }

  const lower = query.toLowerCase()
  const dotIndex = query.lastIndexOf('.')
  const hasDot = dotIndex !== -1

  const results: Completion[] = []

  if (hasDot) {
    const namespacePrefix = query.slice(0, dotIndex + 1)
    const memberQuery = lower.slice(dotIndex + 1)

    for (let i = 0; i < localIndexed.length; i += 1) {
      const item = localIndexed[i]
      if (!item.label.startsWith(namespacePrefix)) continue
      const memberLower =
        item.namespacePrefix === namespacePrefix
          ? item.memberLower
          : item.labelLower.slice(namespacePrefix.length)
      if (memberQuery.length === 0 || memberLower.startsWith(memberQuery)) {
        results.push(item.boost25)
      } else if (memberLower.includes(memberQuery)) {
        results.push(item.boost10)
      }
    }

    const staticBucket = STATIC_BY_NAMESPACE.get(namespacePrefix)
    if (staticBucket) {
      for (let i = 0; i < staticBucket.length; i += 1) {
        const item = staticBucket[i]
        const memberLower = item.memberLower
        if (memberQuery.length === 0 || memberLower.startsWith(memberQuery)) {
          results.push(item.boost25)
        } else if (memberLower.includes(memberQuery)) {
          results.push(item.boost10)
        }
      }
    }
  } else if (query.charCodeAt(0) === 35) {
    for (let i = 0; i < STATIC_COMMANDS.length; i += 1) {
      const item = STATIC_COMMANDS[i]
      if (item.labelLower.startsWith(lower)) {
        results.push(item.label === query ? item.boost40 : item.boost25)
      }
    }
  } else {
    const allowSubstring = lower.length >= 2
    for (let i = 0; i < localIndexed.length; i += 1) {
      const item = localIndexed[i]
      const labelLower = item.labelLower
      const startsWith = labelLower.startsWith(lower)
      const includes = allowSubstring && !startsWith && labelLower.includes(lower)
      if (query.length > 0 && !startsWith && !includes) continue

      if (item.label === query) results.push(item.boost45)
      else if (labelLower === lower) results.push(item.boost40)
      else if (startsWith) results.push(item.boost30)
      else results.push(item.boost5)
    }

    for (let i = 0; i < STATIC_NON_COMMANDS.length; i += 1) {
      const item = STATIC_NON_COMMANDS[i]
      const labelLower = item.labelLower
      const startsWith = labelLower.startsWith(lower)
      const includes = allowSubstring && !startsWith && labelLower.includes(lower)
      if (query.length > 0 && !startsWith && !includes) continue

      if (item.label === query) results.push(item.boost45)
      else if (labelLower === lower) results.push(item.boost40)
      else if (startsWith && !item.hasDot) results.push(item.boost25)
      else if (startsWith) results.push(item.boost18)
      else results.push(item.boost5)
    }
  }

  if (targetCache) {
    if (targetCache.size >= MAX_STATIC_COMPLETION_CACHE) {
      const oldestKey = targetCache.keys().next().value
      if (oldestKey !== undefined) {
        targetCache.delete(oldestKey)
      }
    }
    targetCache.set(query, results)
  }

  return results
}

export function mergeRemoteCompletions(
  prefix: string,
  localOptions: readonly Completion[],
  remoteItems: readonly LeanLspCompletionItem[],
): readonly Completion[] {
  if (remoteItems.length === 0) return localOptions
  const dotIndex = prefix.lastIndexOf('.')
  const namespacePrefix = dotIndex !== -1 ? prefix.slice(0, dotIndex + 1) : ''
  const memberQuery =
    dotIndex !== -1 ? prefix.slice(dotIndex + 1).toLowerCase() : prefix.toLowerCase()

  const byLabel = new Map<string, Completion>()
  for (const option of localOptions) {
    byLabel.set(option.label, option)
  }

  for (const item of remoteItems) {
    const rawLabel = item.label.trim()
    if (!rawLabel) continue
    const normalizedLabel =
      namespacePrefix && !rawLabel.includes('.') && !rawLabel.startsWith(namespacePrefix)
        ? `${namespacePrefix}${rawLabel}`
        : rawLabel
    const compareTarget = namespacePrefix
      ? normalizedLabel.slice(namespacePrefix.length).toLowerCase()
      : normalizedLabel.toLowerCase()
    if (
      memberQuery &&
      !compareTarget.startsWith(memberQuery) &&
      !compareTarget.includes(memberQuery)
    ) {
      continue
    }

    const existing = byLabel.get(normalizedLabel)
    if (existing) {
      if (!existing.detail && item.detail) {
        byLabel.set(normalizedLabel, {
          ...existing,
          detail: item.detail,
          ...(item.documentation && !existing.info ? { info: item.documentation } : {}),
        })
      }
      continue
    }

    byLabel.set(normalizedLabel, {
      label: normalizedLabel,
      type: lspKindToCompletionType(item.kind),
      ...(item.detail ? { detail: item.detail } : {}),
      ...(item.documentation ? { info: item.documentation } : {}),
      boost: compareTarget.startsWith(memberQuery) ? 22 : 12,
    })
  }

  return Array.from(byLabel.values())
}

export function collectLocalIndexedForCompletion(
  context: CompletionContext,
  prefix: string,
): {
  readonly indexed: readonly IndexedCatalogEntry[]
  readonly cacheMap?: Map<string, readonly Completion[]>
} {
  if (prefix.charCodeAt(0) === 35) {
    return { indexed: EMPTY_INDEXED_ENTRIES, cacheMap: staticCompletionCache }
  }
  const currentCache = getDocumentSymbolsForText(context.state.doc)
  const config = context.state.facet(leanLspConfigFacet)
  const siblingCells = config?.cells?.()
  if (!siblingCells || siblingCells.length === 0) {
    return { indexed: currentCache.indexed, cacheMap: currentCache.completionCache }
  }

  const combined: IndexedCatalogEntry[] = [...currentCache.indexed]
  const seen = new Set<string>(currentCache.byLabel.keys())
  for (const cell of siblingCells) {
    if (cell.id === config?.cellId) continue
    const lang = cell.language.trim().toLowerCase()
    if (lang === 'lean' || lang === 'lean4') {
      const siblingCache = getDocumentSymbolsForSource(cell.source)
      for (const item of siblingCache.indexed) {
        if (!seen.has(item.label)) {
          seen.add(item.label)
          combined.push(item)
        }
      }
    }
  }
  return { indexed: combined }
}
