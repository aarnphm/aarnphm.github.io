import { z } from 'zod'
import type { ArenaReaderArtifact } from '../quartz/util/arena-reader'

const legacyReaderProfile = 'anonymous-readability-1-purify-1'
export const ARENA_READER_PROFILE = 'anonymous-defuddle-0.19.3-purify-2'
export const ARENA_TWITTER_PROFILE = 'twitter-defuddle-0.19.3-purify-1'
export const ARENA_GITHUB_PROFILE = 'github-source-1'
export const ARENA_READER_LEASE_MS = 90_000
export const ARENA_READER_COOLDOWN_MS = 10 * 60_000
export const ARENA_READER_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024

const articleId = z.string().regex(/^article-v1-[a-f0-9]{64}$/)
const snapshotId = z.string().uuid()
const resource = z.object({
  id: z.string().regex(/^resource-[a-f0-9]{32}$/),
  url: z.string().url(),
  kind: z.enum(['image', 'pdf']),
  contentType: z.string().optional(),
})
const artifactBase = z.object({
  schemaVersion: z.literal(1),
  articleId,
  snapshotId,
  title: z.string().max(4096),
  sourceUrl: z.string().url(),
  finalUrl: z.string().url(),
  capturedAt: z.number().int().nonnegative(),
  profileVersion: z.enum([
    legacyReaderProfile,
    'anonymous-defuddle-0.19.3-purify-1',
    ARENA_READER_PROFILE,
    'twitter-oembed-1',
    ARENA_TWITTER_PROFILE,
    ARENA_GITHUB_PROFILE,
  ]),
  fingerprint: z.string().regex(/^[a-f0-9]{64}$/),
  resources: z.array(resource).max(300),
})
const artifactSchema = z.discriminatedUnion('kind', [
  artifactBase.extend({
    kind: z.literal('html'),
    readerHtml: z.string().nullable(),
    quality: z.enum(['complete', 'partial']),
    diagnostics: z.array(z.string().max(1024)).max(30),
  }),
  artifactBase.extend({ kind: z.literal('pdf'), resourceId: resource.shape.id }),
  artifactBase.extend({
    kind: z.literal('code'),
    code: z.string(),
    fileName: z.string().max(4096),
  }),
  artifactBase.extend({
    kind: z.literal('video'),
    embedUrl: z.string().url().nullable(),
    description: z.string().nullable(),
  }),
  artifactBase.extend({ kind: z.literal('internal'), internalUrl: z.string() }),
  artifactBase.extend({ kind: z.literal('external'), reason: z.string(), message: z.string() }),
])
const leaseSchema = z.object({
  owner: z.string().uuid(),
  generation: z.number().int().positive(),
  expiresAt: z.number().int().nonnegative(),
})
const failureSchema = z.object({
  reason: z.string().max(100),
  message: z.string().max(2048),
  retryAt: z.number().int().nonnegative(),
})
const stateSchema = z.object({
  schemaVersion: z.literal(1),
  generation: z.number().int().nonnegative(),
  snapshotId: snapshotId.nullable(),
  lease: leaseSchema.nullable(),
  failure: failureSchema.nullable(),
})

export type ArenaRenderState = z.infer<typeof stateSchema>
export type ArenaRenderFailure = z.infer<typeof failureSchema>
export interface ArenaRenderCacheState {
  state: ArenaRenderState
  etag: string | null
}
export interface ArenaRenderLease {
  state: ArenaRenderState
  etag: string
  owner: string
  generation: number
  expiresAt: number
}

export function arenaReaderStateKey(id: string): string {
  // Keep the persisted namespace so an extractor change does not invalidate visited links.
  return `arena-reader/v1/${id}/${legacyReaderProfile}/state.json`
}

export function arenaReaderSnapshotKey(id: string, snapshot: string): string {
  return `arena-reader/v1/${id}/snapshots/${snapshot}.json`
}

export function parseArenaReaderArtifact(value: unknown): ArenaReaderArtifact | null {
  const parsed = artifactSchema.safeParse(value)
  if (!parsed.success) return null
  const ids = new Set(parsed.data.resources.map(item => item.id))
  if (ids.size !== parsed.data.resources.length) return null
  if (parsed.data.kind === 'pdf') {
    const resourceId = parsed.data.resourceId
    if (!parsed.data.resources.some(item => item.id === resourceId && item.kind === 'pdf'))
      return null
  }
  return parsed.data
}

export function emptyArenaRenderState(): ArenaRenderState {
  return { schemaVersion: 1, generation: 0, snapshotId: null, lease: null, failure: null }
}

export async function readArenaRenderCache(
  bucket: R2Bucket,
  id: string,
): Promise<ArenaRenderCacheState> {
  const object = await bucket.get(arenaReaderStateKey(id))
  if (!object) return { state: emptyArenaRenderState(), etag: null }
  if (object.size > 8192) throw new Error('Arena cache state exceeds its size limit')
  const value: unknown = await object.json()
  return { state: stateSchema.parse(value), etag: object.etag }
}

export async function loadArenaReaderSnapshot(
  bucket: R2Bucket,
  id: string,
  snapshot: string,
): Promise<ArenaReaderArtifact | null> {
  if (!articleId.safeParse(id).success || !snapshotId.safeParse(snapshot).success) return null
  const object = await bucket.get(arenaReaderSnapshotKey(id, snapshot))
  if (!object || object.size > ARENA_READER_MAX_ARTIFACT_BYTES) return null
  let value: unknown
  try {
    value = await object.json()
  } catch {
    return null
  }
  const artifact = parseArenaReaderArtifact(value)
  return artifact?.articleId === id && artifact.snapshotId === snapshot ? artifact : null
}

export function arenaRenderCacheDecision(
  state: ArenaRenderState,
  now: number,
  refresh: boolean,
): 'ready' | 'pending' | 'cooldown' | 'render' {
  if (state.snapshotId && !refresh) return 'ready'
  if (state.lease && state.lease.expiresAt > now) return 'pending'
  if (state.failure && state.failure.retryAt > now) return 'cooldown'
  return 'render'
}

export async function claimArenaRenderLease(
  bucket: R2Bucket,
  id: string,
  previous: ArenaRenderCacheState,
  now = Date.now(),
): Promise<ArenaRenderLease | null> {
  if (previous.state.lease && previous.state.lease.expiresAt > now) return null
  const lease = {
    owner: crypto.randomUUID(),
    generation: previous.state.generation + 1,
    expiresAt: now + ARENA_READER_LEASE_MS,
  }
  const state: ArenaRenderState = { ...previous.state, generation: lease.generation, lease }
  const object = await bucket.put(arenaReaderStateKey(id), JSON.stringify(state), {
    onlyIf: previous.etag ? { etagMatches: previous.etag } : { etagDoesNotMatch: '*' },
    httpMetadata: { contentType: 'application/json' },
  })
  return object ? { state, etag: object.etag, ...lease } : null
}

export function arenaRenderPublication(
  lease: ArenaRenderLease,
  snapshot: string | null,
  failure: ArenaRenderFailure | null,
  now = Date.now(),
): ArenaRenderState | null {
  if (now >= lease.expiresAt) return null
  if (lease.state.lease?.owner !== lease.owner || lease.state.lease.generation !== lease.generation)
    return null
  return { ...lease.state, snapshotId: snapshot, lease: null, failure }
}

export async function publishArenaRenderState(
  bucket: R2Bucket,
  id: string,
  lease: ArenaRenderLease,
  snapshot: string | null,
  failure: ArenaRenderFailure | null = null,
  now = Date.now(),
): Promise<boolean> {
  const state = arenaRenderPublication(lease, snapshot, failure, now)
  if (!state) return false
  const result = await bucket.put(arenaReaderStateKey(id), JSON.stringify(state), {
    onlyIf: { etagMatches: lease.etag },
    httpMetadata: { contentType: 'application/json' },
  })
  return result !== null
}

export async function saveArenaReaderSnapshot(
  bucket: R2Bucket,
  artifact: ArenaReaderArtifact,
): Promise<boolean> {
  const parsed = parseArenaReaderArtifact(artifact)
  if (!parsed) throw new Error('Invalid Arena reader artifact')
  const serialized = JSON.stringify(parsed)
  if (new TextEncoder().encode(serialized).byteLength > ARENA_READER_MAX_ARTIFACT_BYTES)
    throw new Error('Arena reader artifact exceeds its size limit')
  const key = arenaReaderSnapshotKey(artifact.articleId, artifact.snapshotId)
  const result = await bucket.put(key, serialized, {
    onlyIf: { etagDoesNotMatch: '*' },
    httpMetadata: { contentType: 'application/json' },
  })
  if (result) return true
  const existing = await loadArenaReaderSnapshot(bucket, artifact.articleId, artifact.snapshotId)
  return existing?.fingerprint === artifact.fingerprint
}

export function keepCompleteArenaSnapshot(
  previous: ArenaReaderArtifact | null,
  next: ArenaReaderArtifact,
): boolean {
  return (
    previous?.kind === 'html' &&
    previous.quality === 'complete' &&
    (next.kind === 'external' || (next.kind === 'html' && next.quality === 'partial'))
  )
}

export async function arenaReaderHash(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}
