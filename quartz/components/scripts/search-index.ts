import FlexSearch, { DocumentData } from 'flexsearch'
import { FullSlug, FilePath } from '../../util/path'
import { isSearchExcluded } from '../../util/search-exclude'
import { encode, pathSearchText } from '../../util/search-text'

export interface SearchItem extends DocumentData {
  id: number
  slug: FullSlug
  name: FilePath
  path: string
  title: string
  content: string
  aliases: string[]
  [key: string]: any
}

const searchIndex = new FlexSearch.Document<SearchItem>({
  encode,
  document: {
    id: 'id',
    tag: 'slug',
    index: [
      { field: 'title', tokenize: 'forward' },
      { field: 'path', tokenize: 'forward' },
      { field: 'aliases', tokenize: 'forward' },
      { field: 'content', tokenize: 'forward' },
    ],
  },
})

const itemsById = new Map<number, SearchItem>()
let populated: Promise<void> | null = null

async function queryFields(
  query: string,
  fields: Array<'title' | 'path' | 'aliases' | 'content'>,
  limit: number,
): Promise<SearchItem[]> {
  if (!query || query.trim() === '') {
    return [...itemsById.values()].slice(0, limit)
  }

  const results = await searchIndex.searchAsync({ query, limit, index: fields })
  const allIds = new Set<number>()
  for (const fieldResult of results) {
    for (const id of fieldResult.result as number[]) {
      allIds.add(id)
    }
  }

  return [...allIds]
    .map(id => itemsById.get(id))
    .filter((item): item is SearchItem => item !== undefined)
    .slice(0, limit)
}

// Callers share one fill so a query never runs against a partly built index.
export function populateSearchIndex(data: ContentIndex): Promise<void> {
  populated ??= addDocuments(data).catch(err => {
    populated = null
    throw err
  })
  return populated
}

async function addDocuments(data: ContentIndex): Promise<void> {
  let id = 0
  const promises = []

  for (const [slug, fileData] of Object.entries(data)) {
    if (isSearchExcluded(slug) || isSearchExcluded(fileData.fileName)) continue
    const item: SearchItem = {
      id,
      slug: slug as FullSlug,
      name: fileData.fileName,
      path: pathSearchText(fileData.fileName),
      title: fileData.title ?? '',
      content: fileData.content ?? '',
      aliases: fileData.aliases,
    }

    itemsById.set(id, item)
    promises.push(searchIndex.addAsync(id, item))
    id++
  }

  await Promise.all(promises)
}

export async function querySearchIndex(query: string, limit: number = 10): Promise<SearchItem[]> {
  return queryFields(query, ['title', 'path', 'aliases'], limit)
}

export async function querySiteSearchIndex(
  query: string,
  limit: number = 10,
): Promise<SearchItem[]> {
  return queryFields(query, ['title', 'path', 'aliases', 'content'], limit)
}
