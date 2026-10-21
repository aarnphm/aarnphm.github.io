import { z } from 'zod'
import type { Link } from '../../types/curius'
import { curiusDateKey } from '../../util/curius-date'
import { previewHref, previewText, renderPagePreview, type PagePreview } from './page-preview-card'

const summarySchema = z.object({
  title: z.string().max(200),
  description: z.string().max(650),
  meta: z.array(z.string().max(200)).max(8),
})

type SavedSummary = Pick<Link, 'link' | 'title' | 'snippet' | 'createdDate'> &
  Partial<Pick<Link, 'topics' | 'highlights'>>

/** The feed and popover can live in separate bundles, so the bounded summary travels with its link. */
export function bindCuriusPopover(saved: SavedSummary, anchor: HTMLAnchorElement): void {
  const href = previewHref(saved.link)
  if (!href) return
  const excerpt = new DOMParser().parseFromString(saved.snippet || '', 'text/html')
  excerpt.querySelectorAll('script, style').forEach(node => node.remove())
  const date = saved.createdDate ? new Date(saved.createdDate) : null
  const meta = [new URL(href).hostname.replace(/^www\./, '')]
  if (date && Number.isFinite(date.getTime())) meta.push(`saved ${curiusDateKey(date)}`)
  const highlights = saved.highlights?.length ?? 0
  if (highlights) {
    meta.push(`${highlights} ${highlights === 1 ? 'highlight' : 'highlights'}`)
  }
  meta.push(
    ...(saved.topics ?? [])
      .filter(topic => topic.public)
      .slice(0, 3)
      .map(topic => previewText(topic.topic, 60)),
  )
  anchor.dataset.curiusPopover = JSON.stringify({
    title: previewText(saved.title || href, 100),
    description: previewText(excerpt.body.textContent ?? '', 320),
    meta,
  })
}

export function curiusPopoverSource(anchor: HTMLAnchorElement) {
  const raw = anchor.dataset.curiusPopover
  const href = previewHref(anchor.href)
  if (!raw || !href) return null
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    return null
  }
  const parsed = summarySchema.safeParse(value)
  if (!parsed.success) return null
  const preview: PagePreview = { kind: 'curius-link', href, ...parsed.data }
  return {
    popoverId: `popover-curius-link-${encodeURIComponent(href)}`,
    className: 'page-preview-popover',
    async load() {
      return (inner: HTMLDivElement) => renderPagePreview(preview, inner)
    },
  }
}
