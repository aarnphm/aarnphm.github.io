import { readNotebookPreview } from '../../util/notebook-preview'
import { previewText } from './page-preview-card'

export function notebookPreviewDetails(page: Document): { meta: string[]; description: string } {
  const article = page.querySelector('article.notebook-page')
  const data = readNotebookPreview(
    article?.querySelector('[data-notebook-preview]')?.getAttribute('data-notebook-preview'),
  )
  const meta: string[] = []
  if (data) {
    const { code, markdown, raw } = data.blocks
    const total = code + markdown + raw
    const breakdown = [`${code} code`, `${markdown} text`]
    if (raw) breakdown.push(`${raw} raw`)
    meta.push(
      `runtime: ${data.runtimes.join(', ') || 'static'}`,
      `${total} ${total === 1 ? 'block' : 'blocks'}`,
      breakdown.join(' · '),
    )
  }
  const description =
    data?.description || page.querySelector<HTMLMetaElement>('meta[name="description"]')?.content
  if (description) return { meta, description: previewText(description, 240) }
  const paragraphs = article?.querySelectorAll('p') ?? []
  for (const paragraph of paragraphs) {
    if (
      paragraph.closest(
        '.notebook-code-cell, .notebook-output, .notebook-runtime, [hidden], [data-skip-preview]',
      )
    )
      continue
    const prose = previewText(paragraph.textContent ?? '', 240)
    if (prose) return { meta, description: prose }
  }
  return { meta, description: '' }
}
