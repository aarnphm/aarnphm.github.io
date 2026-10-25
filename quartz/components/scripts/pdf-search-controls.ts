import type { PdfPassage } from '../../util/pdf-search'

export function pdfPageLabel(passage: PdfPassage): string {
  const label =
    passage.pageLabel && passage.pageLabel !== String(passage.page) ? ` (${passage.pageLabel})` : ''
  return `pdf p. ${passage.page}${label}`
}
