import FlexSearch, { type Document as FlexSearchDocument } from 'flexsearch'
import type { StreamManifestEntry, StreamManifestGroup } from '../../util/stream-manifest'
import { encode } from '../../util/search-text'

export interface IndexedStreamEntry {
  id: number
  entry: StreamManifestEntry
  group: StreamManifestGroup
}

export interface StreamSearchData {
  index: FlexSearchDocument
  entries: IndexedStreamEntry[]
}

const tagsForEntry = (entry: StreamManifestEntry): string[] => {
  if (!entry.metadata || typeof entry.metadata !== 'object' || Array.isArray(entry.metadata))
    return []
  const tags = Reflect.get(entry.metadata, 'tags')
  if (!Array.isArray(tags)) return []
  return tags.map(tag => String(tag).trim()).filter(tag => tag.length > 0)
}

export const tagTokens = (query: string): string[] =>
  Array.from(
    new Set(
      query
        .trim()
        .split(/\s+/)
        .map(token => (token.startsWith('#') ? token.slice(1) : ''))
        .filter(token => token.length > 0),
    ),
  )

export const buildStreamSearchData = async (
  groups: StreamManifestGroup[],
): Promise<StreamSearchData> => {
  const entries = groups.flatMap(group => group.entries.map(entry => ({ id: 0, entry, group })))
  entries.forEach((entry, index) => {
    entry.id = index
  })

  const index = new FlexSearch.Document({
    tokenize: 'forward',
    encode,
    document: { id: 'id', index: ['content', 'metadata', 'isoDate', 'displayDate', 'tags'] },
  })

  for (const indexed of entries) {
    const tags = tagsForEntry(indexed.entry)
    await index.addAsync({
      id: indexed.id,
      content: [indexed.entry.title, indexed.entry.description, indexed.entry.content]
        .filter(value => value !== null && value.length > 0)
        .join(' '),
      metadata: JSON.stringify(indexed.entry.metadata ?? {}),
      isoDate: indexed.entry.isoDate ?? indexed.group.isoDate ?? '',
      displayDate: indexed.entry.displayDate ?? indexed.group.isoDate ?? '',
      tags: tags.flatMap(tag => [tag, `#${tag}`]).join(' '),
    })
  }

  return { index, entries }
}

export const matchStreamEntries = async (
  data: StreamSearchData,
  query: string,
): Promise<IndexedStreamEntry[]> => {
  const normalizedQuery = query.trim().toLowerCase()
  if (normalizedQuery.startsWith('#')) {
    const queryTags = tagTokens(normalizedQuery)
    if (queryTags.length === 0) return []
    return data.entries.filter(({ entry }) => {
      const tags = tagsForEntry(entry).map(tag => tag.toLowerCase())
      return queryTags.every(queryTag => tags.some(tag => tag.startsWith(queryTag)))
    })
  }

  const results = await data.index.searchAsync({
    query,
    limit: 500,
    index: ['content', 'metadata', 'isoDate', 'displayDate', 'tags'],
  })
  const ids = new Set<number>()
  for (const result of results) {
    for (const id of result.result) {
      const numericId = Number(id)
      if (Number.isInteger(numericId)) ids.add(numericId)
    }
  }
  return data.entries.filter(entry => ids.has(entry.id))
}
