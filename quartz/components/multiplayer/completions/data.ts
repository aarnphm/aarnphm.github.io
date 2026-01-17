import type { ContentDetails } from "../../../plugins/emitters/contentIndex"

export type ContentIndex = Record<string, ContentDetails>

declare const fetchData: Promise<ContentIndex>

let cached: ContentIndex | null = null

export async function getContentIndex(): Promise<ContentIndex> {
  if (cached) return cached
  cached = await fetchData
  return cached
}
