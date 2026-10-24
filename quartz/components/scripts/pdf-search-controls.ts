import type { PdfIndex, PdfPassage } from '../../util/pdf-search'

export function pdfCoverageStatus(coverage: PdfIndex['coverage'] | undefined): HTMLElement {
  const status = document.createElement('p')
  status.setAttribute('role', 'status')
  status.dataset.pdfCoverage = 'true'
  if (coverage)
    status.textContent = `text coverage: ${coverage.indexedPages.toLocaleString()} pages across ${coverage.indexedDocuments.toLocaleString()} of ${coverage.eligibleDocuments.toLocaleString()} pdfs`
  return status
}

export function pdfPageLabel(passage: PdfPassage): string {
  const label =
    passage.pageLabel && passage.pageLabel !== String(passage.page) ? ` (${passage.pageLabel})` : ''
  return `pdf p. ${passage.page}${label}`
}
