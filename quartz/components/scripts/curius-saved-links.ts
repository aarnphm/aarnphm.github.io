import { z } from 'zod'
import { curiusDateKey, type CuriusDateFilter } from '../../util/curius-date'
import { previewHref, previewText, type PagePreviewRow } from './page-preview-card'

const savedLinksSchema = z.object({
  links: z.array(
    z.object({
      id: z.number().optional(),
      link: z.string(),
      title: z.string().nullish(),
      snippet: z.string().nullish(),
      createdDate: z.string().nullish(),
    }),
  ),
})
export type CuriusSavedLink = z.infer<typeof savedLinksSchema>['links'][number]

export async function fetchCuriusSavedLinks(
  signal: AbortSignal,
  archive = true,
): Promise<CuriusSavedLink[]> {
  const response = await fetch(`/api/curius?query=${archive ? 'searchLinks' : 'links'}`, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(20_000)]),
  })
  if (!response.ok) throw Error('Could not load saved links')
  const data = savedLinksSchema.parse(await response.json())
  const seen = new Set<string>()
  const links: CuriusSavedLink[] = []
  for (const link of data.links) {
    const href = previewHref(link.link)
    const key = link.id === undefined ? href : String(link.id)
    if (!href || !key || seen.has(key)) continue
    seen.add(key)
    links.push({ ...link, link: href })
  }
  return links
}

export function filterCuriusSavedLinks(
  links: CuriusSavedLink[],
  filter: CuriusDateFilter,
): CuriusSavedLink[] {
  if (filter.kind === 'invalid') return []
  if (filter.kind === 'all') return links
  return links.filter(link => {
    if (!link.createdDate) return false
    const date = new Date(link.createdDate)
    return Number.isFinite(date.getTime()) && curiusDateKey(date) === filter.key
  })
}

export function curiusSavedRows(links: CuriusSavedLink[]): PagePreviewRow[] {
  return links.map(link => ({
    title: previewText(link.title || link.link, 160),
    href: previewHref(link.link),
    detail: new URL(link.link).hostname,
  }))
}
