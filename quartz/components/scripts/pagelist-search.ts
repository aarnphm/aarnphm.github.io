export interface PageListSearchItem {
  title: string
  tags: string[]
}

export function extractTagsFromQuery(query: string): string[] {
  const matches = query.match(/#[\w/-]+/g) || []
  return matches.map(t => t.slice(1).toLowerCase())
}

export function extractTextQuery(query: string): string {
  return query
    .replace(/#[\w/-]+/g, '')
    .trim()
    .toLowerCase()
}

export function tagMatches(queryTag: string, itemTags: string[]): boolean {
  return itemTags.some(t => t.includes(queryTag))
}

export function pageListItemMatchesQuery(item: PageListSearchItem, query: string): boolean {
  const queryTags = extractTagsFromQuery(query)
  const textQuery = extractTextQuery(query)
  const title = item.title.toLowerCase()
  const itemTags = item.tags.map(t => t.toLowerCase())
  const textMatch = !textQuery || title.includes(textQuery)
  const tagMatch = queryTags.length === 0 || queryTags.some(qt => tagMatches(qt, itemTags))

  return textMatch && tagMatch
}
