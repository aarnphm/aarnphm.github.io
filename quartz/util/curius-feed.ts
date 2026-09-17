import { z } from 'zod'
import {
  arenaFeedArticleId,
  arenaFeedContentKind,
  arenaFeedSavedDate,
  finalizeArenaFeedManifest,
  normalizeArenaFeedUrl,
  type ArenaFeedEntry,
  type ArenaFeedManifest,
} from './arena-feed'

export const CURIUS_FEED_USER_ID = 3584
export const CURIUS_FEED_URL = `https://curius.app/api/users/${CURIUS_FEED_USER_ID}/searchLinks`

const savedLink = z.object({
  id: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
  link: z.string(),
  title: z.string().nullable(),
  toRead: z.boolean().nullish(),
  createdDate: z.string().nullish(),
})
const response = z.object({ links: z.array(savedLink) })
export type CuriusSavedLink = z.infer<typeof savedLink>

export function parseCuriusFeedLinks(value: unknown): CuriusSavedLink[] | null {
  const parsed = response.safeParse(value)
  return parsed.success ? parsed.data.links : null
}

export async function mergeCuriusFeed(
  arena: ArenaFeedManifest,
  links: readonly CuriusSavedLink[],
  baseUrl: string,
): Promise<ArenaFeedManifest> {
  if (links.length === 0) return arena
  const entries = new Map<string, ArenaFeedEntry>(
    arena.entries.map(entry => [entry.sourceUrl, { ...entry }]),
  )
  // Provider order changes independently of the saved catalogue's contents.
  for (const link of [...links].sort((a, b) => a.id - b.id)) {
    const sourceUrl = normalizeArenaFeedUrl(link.link)
    if (!sourceUrl) continue
    const title = link.title?.trim() || sourceUrl
    const savedAt = arenaFeedSavedDate(link.createdDate)
    const source = { userId: CURIUS_FEED_USER_ID, linkId: link.id }
    const entry = entries.get(sourceUrl)
    if (entry) {
      const curius = entry.curius ?? []
      if (!curius.some(item => item.userId === source.userId && item.linkId === source.linkId))
        entry.curius = [...curius, source]
      entry.later ||= link.toRead === true
      if (entry.title === sourceUrl && title !== sourceUrl) entry.title = title
      if (savedAt && (!entry.savedAt || savedAt < entry.savedAt)) entry.savedAt = savedAt
    } else {
      entries.set(sourceUrl, {
        articleId: await arenaFeedArticleId(sourceUrl),
        sourceUrl,
        title,
        kind: arenaFeedContentKind(sourceUrl, baseUrl),
        later: link.toRead === true,
        tags: [],
        savedAt,
        occurrences: [],
        curius: [source],
      })
    }
  }
  return finalizeArenaFeedManifest([...entries.values()])
}
