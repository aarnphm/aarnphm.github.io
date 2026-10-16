import FlexSearch, { type DefaultDocumentSearchResults, type Id } from 'flexsearch'
import type { LexicalDocument, LexicalFieldResult, LexicalSearchSpec } from './lexical-search-types'
import { encode } from '../../util/search-text'

export function createLexicalDocumentIndex() {
  return new FlexSearch.Document<LexicalDocument>({
    tokenize: 'forward',
    encode,
    document: {
      id: 'id',
      index: [
        { field: 'title', tokenize: 'forward' },
        { field: 'fileName', tokenize: 'forward' },
        { field: 'content', tokenize: 'forward' },
        { field: 'tags', tokenize: 'forward' },
        { field: 'aliases', tokenize: 'forward' },
      ],
    },
  })
}

export type LexicalDocumentIndex = ReturnType<typeof createLexicalDocumentIndex>

function lexicalField(field: string | undefined): LexicalFieldResult['field'] {
  switch (field) {
    case 'title':
    case 'fileName':
    case 'content':
    case 'tags':
    case 'aliases':
      return field
    default:
      return undefined
  }
}

function coerceLexicalIds(
  hit?: DefaultDocumentSearchResults<LexicalDocument>[number] | LexicalFieldResult,
): number[] {
  if (!hit) return []
  return hit.result
    .map((value: Id) => {
      if (typeof value === 'number') {
        return value
      }
      const parsed = Number.parseInt(String(value), 10)
      return Number.isNaN(parsed) ? null : parsed
    })
    .filter((value): value is number => value !== null)
}

function normalizeLexicalSearchResults(
  rawResults: DefaultDocumentSearchResults<LexicalDocument>,
): LexicalFieldResult[] {
  return Object.values(rawResults).map(entry => ({
    field: lexicalField(entry.field),
    tag: lexicalField(entry.tag),
    result: coerceLexicalIds(entry),
  }))
}

export async function searchLexicalDocumentIndex(
  index: LexicalDocumentIndex,
  spec: LexicalSearchSpec,
): Promise<LexicalFieldResult[]> {
  const results = spec.tag
    ? await index.searchAsync({
        query: spec.query,
        limit: spec.limit,
        index: spec.index,
        tag: spec.tag,
      })
    : await index.searchAsync({ query: spec.query, limit: spec.limit, index: spec.index })
  return normalizeLexicalSearchResults(results)
}
