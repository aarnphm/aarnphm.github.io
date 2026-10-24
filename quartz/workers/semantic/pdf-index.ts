import {
  createLexicalDocumentIndex,
  searchLexicalDocumentIndex,
  type LexicalDocumentIndex,
} from '../../components/scripts/lexical-search-index'
import {
  readPdfPassageShard,
  type PdfIndex,
  type PdfMetadata,
  type PdfPassage,
} from '../../util/pdf-search'

/** Passage text, document grouping and the lexical body index of one PDF generation. */
export type PdfSearchable = {
  generation: string
  coverage: PdfIndex['coverage']
  documents: Record<string, PdfMetadata>
  passages: PdfPassage[]
  /** Row to document index, for the per-document reduction of a scan. */
  groups: Int32Array
  groupCount: number
  lexical: LexicalDocumentIndex
}

const normalized = (text: string) => text.replace(/[^\p{L}\p{N}]+/gu, ' ')

export async function loadPdfPassages(
  index: PdfIndex,
  fetchBinary: (path: string) => Promise<ArrayBuffer>,
): Promise<PdfPassage[]> {
  if (index.passages) return index.passages
  const shards = await Promise.all(
    index.passageShards.map(async shard =>
      readPdfPassageShard(await fetchBinary(shard.path), shard, index.documents),
    ),
  )
  return shards.flat()
}

export function buildPdfSearchable(index: PdfIndex, passages: PdfPassage[]): PdfSearchable {
  if (passages.length !== index.rows) throw new Error('PDF passages do not match the vector rows')
  const groupOf = new Map<string, number>()
  const groups = new Int32Array(passages.length)
  const lexical = createLexicalDocumentIndex()
  for (const [row, passage] of passages.entries()) {
    let group = groupOf.get(passage.slug)
    if (group === undefined) {
      group = groupOf.size
      groupOf.set(passage.slug, group)
    }
    groups[row] = group
    lexical.add({
      id: row,
      slug: passage.slug,
      title: normalized(passage.title),
      content: normalized(passage.text),
      fileName: passage.slug,
      aliases: [],
      tags: [],
      target: '',
      protected: false,
    })
  }
  return {
    generation: index.generation,
    coverage: index.coverage,
    documents: index.documents,
    passages,
    groups,
    groupCount: groupOf.size,
    lexical,
  }
}

/** Body matches, one passage per PDF, in lexical rank order. */
export async function pdfLexicalMatches(pdf: PdfSearchable, query: string): Promise<PdfPassage[]> {
  const results = await searchLexicalDocumentIndex(pdf.lexical, {
    query: normalized(query),
    limit: pdf.passages.length,
    index: ['content'],
  })
  const seen = new Set<string>()
  return results
    .flatMap(result => result.result)
    .flatMap(id => {
      const passage = pdf.passages[id]
      if (!passage || seen.has(passage.slug)) return []
      seen.add(passage.slug)
      return [passage]
    })
}
