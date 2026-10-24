import type { PdfPassage } from '../../util/pdf-search'
import { createLexicalDocumentIndex, searchLexicalDocumentIndex } from './lexical-search-index'

const normalized = (text: string) => text.replace(/[^\p{L}\p{N}]+/gu, ' ')

export async function createPdfLexical(passages: readonly PdfPassage[]) {
  const index = createLexicalDocumentIndex()
  for (const [id, passage] of passages.entries()) {
    index.add({
      id,
      slug: passage.slug,
      title: normalized(passage.title),
      content: normalized(passage.text),
      fileName: passage.slug,
      aliases: [],
      tags: [],
      target: '',
      protected: false,
    })
    if (id > 0 && id % 32 === 0) await new Promise(resolve => setTimeout(resolve, 0))
  }
  return {
    async search(query: string): Promise<PdfPassage[]> {
      const results = await searchLexicalDocumentIndex(index, {
        query: normalized(query),
        limit: passages.length,
        index: ['content'],
      })
      const seen = new Set<string>()
      return results
        .flatMap(result => result.result)
        .flatMap(id => {
          const passage = passages[id]
          if (!passage || seen.has(passage.slug)) return []
          seen.add(passage.slug)
          return [passage]
        })
    },
    dispose() {
      index.clear()
    },
  }
}
