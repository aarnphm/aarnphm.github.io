import type { ArenaFeedManifest } from '../quartz/util/arena-feed'
import {
  mergeCuriusFeed,
  parseCuriusFeedLinks,
  CURIUS_FEED_URL,
  CURIUS_FEED_USER_ID,
  type CuriusSavedLink,
} from '../quartz/util/curius-feed'
import { isRecord } from './type-guards'

export const CURIUS_FEED_CACHE_KEY = `arena-reader/catalogue/v1/curius-${CURIUS_FEED_USER_ID}.json`
export const CURIUS_FEED_CACHE_MS = 15 * 60_000
const RETRY_MS = 60_000

interface CachedCuriusFeed {
  schemaVersion: 1
  fetchedAt: number
  retryAt: number
  links: CuriusSavedLink[]
}

export class CuriusFeedUnavailableError extends Error {}

function parseCachedCuriusFeed(value: unknown): CachedCuriusFeed | null {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    typeof value.fetchedAt !== 'number' ||
    !Number.isFinite(value.fetchedAt) ||
    value.fetchedAt < 0 ||
    typeof value.retryAt !== 'number' ||
    !Number.isFinite(value.retryAt) ||
    value.retryAt < 0
  )
    return null
  const links = parseCuriusFeedLinks(value)
  return links
    ? { schemaVersion: 1, fetchedAt: value.fetchedAt, retryAt: value.retryAt, links }
    : null
}

async function fetchCuriusFeedLinks(fetcher: typeof fetch = fetch): Promise<CuriusSavedLink[]> {
  const response = await fetcher(CURIUS_FEED_URL, {
    headers: { Accept: 'application/json' },
    redirect: 'manual',
    signal: AbortSignal.timeout(10_000),
  })
  if (!response.ok) throw new Error(`Curius returned HTTP ${response.status}`)
  const links = parseCuriusFeedLinks(await response.json())
  if (!links) throw new Error('Curius returned an invalid saved-link index')
  return links
}

export async function loadCuriusFeedLinks(
  bucket: Pick<R2Bucket, 'get' | 'put'>,
  refresh: boolean,
  fetcher: typeof fetch = fetch,
): Promise<CuriusSavedLink[]> {
  const stored = await bucket.get(CURIUS_FEED_CACHE_KEY)
  const cached = stored ? parseCachedCuriusFeed(await stored.json()) : null
  const now = Date.now()
  if (cached && (!refresh || now < cached.fetchedAt + CURIUS_FEED_CACHE_MS || now < cached.retryAt))
    return cached.links

  let links: CuriusSavedLink[]
  try {
    links = await fetchCuriusFeedLinks(fetcher)
  } catch (error) {
    console.warn('Curius catalogue refresh failed', error instanceof Error ? error.message : error)
    if (!cached)
      throw new CuriusFeedUnavailableError(
        'Your Curius links could not be loaded. Try again shortly.',
        { cause: error },
      )
    await bucket.put(CURIUS_FEED_CACHE_KEY, JSON.stringify({ ...cached, retryAt: now + RETRY_MS }))
    return cached.links
  }
  const updated: CachedCuriusFeed = { schemaVersion: 1, fetchedAt: now, retryAt: 0, links }
  await bucket.put(CURIUS_FEED_CACHE_KEY, JSON.stringify(updated), {
    httpMetadata: { contentType: 'application/json' },
  })
  return links
}

export async function includeCuriusFeed(
  arena: ArenaFeedManifest,
  bucket: Pick<R2Bucket, 'get' | 'put'>,
  refresh: boolean,
): Promise<ArenaFeedManifest> {
  const links = await loadCuriusFeedLinks(bucket, refresh)
  return mergeCuriusFeed(arena, links, 'https://aarnphm.xyz')
}
