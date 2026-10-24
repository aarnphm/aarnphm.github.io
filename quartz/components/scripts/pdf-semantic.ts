import {
  PDF_MANIFEST_URL,
  rankPdfResults,
  type PdfIndex,
  type PdfMetadata,
  type PdfSearchDocument,
} from '../../util/pdf-search'
import { isSemanticCancellation, SemanticClient } from './semantic.inline'

/**
 * Thin client over the semantic worker, which owns the PDF index: manifest, passage text,
 * vectors and the lexical passage index. Rank fusion with site metadata stays here because
 * only the page has the attachment inventory.
 */
export class PdfSemantic {
  private client: SemanticClient | null = null
  private documents: Record<string, PdfMetadata> = {}
  private disposed = false
  private generation = 0
  coverage: PdfIndex['coverage'] | undefined

  cancel() {
    this.generation++
    this.client?.cancel()
  }

  dispose() {
    this.disposed = true
    this.cancel()
    this.client?.dispose()
    this.client = null
  }

  private acquire() {
    if (this.disposed) throw new DOMException('PDF search closed', 'AbortError')
    return (this.client ??= new SemanticClient({ ...semanticCfg, manifestUrl: PDF_MANIFEST_URL }))
  }

  async search(
    query: string,
    lexical: readonly string[],
    metadata: Readonly<Record<string, PdfMetadata>>,
    allowed: ReadonlySet<string>,
    semantic = true,
  ): Promise<PdfSearchDocument[]> {
    const generation = this.generation
    const superseded = () => generation !== this.generation || this.disposed
    let result
    try {
      result = await this.acquire().pdfSearch(query, semantic)
    } catch (error) {
      if (superseded() || isSemanticCancellation(error)) throw error
      // A generation may have been replaced under the worker; the retry resolves the new one.
      this.client?.dispose()
      this.client = null
      result = await this.acquire().pdfSearch(query, semantic)
    }
    if (superseded()) throw new DOMException('PDF query superseded', 'AbortError')
    this.coverage = result.coverage
    Object.assign(this.documents, result.documents)
    const scored = new Set(result.semantic.map(document => document.slug))
    const bodyOnly = result.lexical
      .filter(passage => !scored.has(passage.slug))
      .map(passage => ({ slug: passage.slug, score: Number.NaN, passage }))
    return rankPdfResults(
      query,
      [...result.lexical.map(passage => passage.slug), ...lexical],
      [...result.semantic, ...bodyOnly],
      { ...metadata, ...result.documents },
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
      { ...metadata, ...this.documents },
      new Set(Object.keys(metadata)),
    )
  }
}
