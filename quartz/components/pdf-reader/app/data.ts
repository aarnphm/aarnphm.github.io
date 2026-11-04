import {
  parsePdfDocumentRecord,
  pdfReaderPath,
  type PdfDocumentRecord,
} from '../../../util/pdf-marks'

export interface DocumentData {
  slug: string
  readPath: string
  document: PdfDocumentRecord
  /** `course`: the page's strip owns the title; the bar drops its `reader /` block. */
  chrome?: 'course'
}

export interface IndexDocument {
  slug: string
  readPath: string
  title: string
  citations: number
}

export type ReaderData =
  | ({ mode: 'document' } & DocumentData)
  | { mode: 'index'; documents: IndexDocument[] }

export function readData(): ReaderData | null {
  try {
    const value: unknown = JSON.parse(document.getElementById('pdf-reader-data')?.textContent ?? '')
    if (typeof value !== 'object' || value === null) return null
    const record = value as Record<string, unknown>
    if (record.mode === 'index' && Array.isArray(record.documents)) {
      return {
        mode: 'index',
        documents: record.documents.filter(
          (doc): doc is IndexDocument =>
            typeof doc === 'object' &&
            doc !== null &&
            typeof doc.slug === 'string' &&
            typeof doc.readPath === 'string' &&
            typeof doc.title === 'string' &&
            typeof doc.citations === 'number',
        ),
      }
    }
    const parsed = parsePdfDocumentRecord(record.document)
    if (record.mode === 'document' && typeof record.slug === 'string' && parsed) {
      const readPath =
        typeof record.readPath === 'string' && /^\/[^/]/.test(record.readPath)
          ? record.readPath
          : pdfReaderPath(record.slug)
      return {
        mode: 'document',
        slug: record.slug,
        readPath,
        document: parsed,
        ...(record.chrome === 'course' ? { chrome: 'course' as const } : {}),
      }
    }
  } catch {
    /* fall through to the missing-data message */
  }
  return null
}
