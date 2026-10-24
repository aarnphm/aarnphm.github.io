import type { SemanticHit } from '../../util/semantic-search'
import {
  PDF_MANIFEST_URL,
  pdfSearchResults,
  rankPdfResults,
  readPdfIndex,
  type PdfIndex,
  type PdfMetadata,
  type PdfSearchDocument,
} from '../../util/pdf-search'
import { createPdfLexical } from './pdf-lexical'
import { SemanticClient } from './semantic.inline'

export class PdfSemantic {
  private client: SemanticClient | null = null
  private index: PdfIndex | null = null
  private ready: Promise<PdfIndex> | null = null
  private lexical: Awaited<ReturnType<typeof createPdfLexical>> | null = null
  private disposed = false
  private generation = 0

  cancel() {
    this.generation++
    this.client?.cancel()
  }

  dispose() {
    this.disposed = true
    this.cancel()
    this.client?.dispose()
    this.client = null
    this.lexical?.dispose()
    this.lexical = null
  }

  get coverage() {
    return this.index?.coverage
  }

  private load() {
    this.ready ??= fetch(PDF_MANIFEST_URL, { cache: 'no-cache' })
      .then(async response => {
        if (!response.ok) throw new Error(`PDF index unavailable: ${response.status}`)
        const index = readPdfIndex(await response.json())
        if (this.disposed) throw new DOMException('PDF search closed', 'AbortError')
        this.index = index
        const lexical = await createPdfLexical(index.passages)
        if (this.disposed) {
          lexical.dispose()
          throw new DOMException('PDF search closed', 'AbortError')
        }
        this.lexical = lexical
        return index
      })
      .catch(error => {
        this.ready = null
        throw error
      })
    return this.ready
  }

  async search(
    query: string,
    lexical: readonly string[],
    metadata: Readonly<Record<string, PdfMetadata>>,
    allowed: ReadonlySet<string>,
    semantic = true,
  ): Promise<PdfSearchDocument[]> {
    if (this.disposed) throw new DOMException('PDF search closed', 'AbortError')
    const generation = this.generation
    const index = await this.load()
    if (generation !== this.generation || this.disposed)
      throw new DOMException('PDF query superseded', 'AbortError')
    const bodyMatches = (await this.lexical?.search(query)) ?? []
    let hits: SemanticHit[] = []
    if (semantic) {
      this.client ??= new SemanticClient({
        ...semanticCfg,
        manifestUrl: `/embeddings/pdf/${index.generation}/manifest.json`,
      })
      hits = (await this.client.search(query, index.passages.length)).semantic
    }
    if (generation !== this.generation) throw new DOMException('PDF query superseded', 'AbortError')
    const documents = pdfSearchResults(hits, index.passages)
    const scored = new Set(documents.map(document => document.slug))
    const bodyOnly = bodyMatches
      .filter(passage => !scored.has(passage.slug))
      .map(passage => ({ slug: passage.slug, score: Number.NaN, passage }))
    return rankPdfResults(
      query,
      [...bodyMatches.map(passage => passage.slug), ...lexical],
      [...documents, ...bodyOnly],
      { ...metadata, ...index.documents },
      allowed,
    )
  }

  fallback(
    query: string,
    lexical: readonly string[],
    metadata: Readonly<Record<string, PdfMetadata>>,
  ) {
    return rankPdfResults(
      query,
      lexical,
      [],
      { ...metadata, ...this.index?.documents },
      new Set(Object.keys(metadata)),
    )
  }
}
