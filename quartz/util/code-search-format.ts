import { z } from 'zod'
import type { CodeSearchManifest, CodeShardReference, CodeSourceShard } from './code-search-types'

const hash = z.string().regex(/^[a-f0-9]{64}$/)
const positive = z.number().int().positive()
const unit = z
  .object({
    id: z.string().min(1),
    ownerSlug: z.string().min(1),
    title: z.string(),
    path: z.string(),
    kind: z.enum(['attachment', 'block', 'cell']),
    language: z.string().min(1),
    originalLanguage: z.string(),
    sourceId: hash,
    anchor: z.string().nullable(),
    startLine: positive,
    endLine: positive,
  })
  .refine(value => value.endLine >= value.startLine)
const shard = z.object({
  language: z.string().min(1),
  path: z.string().min(1),
  sha256: hash,
  byteLength: z.number().int().nonnegative(),
  sourceIds: z.array(hash),
})
const manifest = z.object({
  kind: z.literal('quartz-code-search-v1'),
  version: z.literal(1),
  generation: z.string().min(1),
  units: z.array(unit),
  shards: z.array(shard),
  coverage: z.object({
    owners: z.number().int().nonnegative(),
    units: z.number().int().nonnegative(),
    sources: z.number().int().nonnegative(),
  }),
})
const sourceShard = z.object({
  kind: z.literal('quartz-code-sources-v1'),
  generation: z.string(),
  language: z.string(),
  sources: z.array(z.object({ id: hash, text: z.string() })),
})

export function readCodeManifest(value: unknown): CodeSearchManifest {
  const result = manifest.safeParse(value)
  if (!result.success) throw new Error(`Invalid code search manifest: ${result.error.message}`)
  const data = result.data
  if (new Set(data.units.map(unit => unit.id)).size !== data.units.length)
    throw new Error('Invalid duplicate code units')
  const sources = new Map<string, Set<string>>()
  for (const shard of data.shards) {
    const ids = sources.get(shard.language) ?? new Set<string>()
    for (const id of shard.sourceIds) ids.add(id)
    sources.set(shard.language, ids)
  }
  for (const unit of data.units) {
    if (!sources.get(unit.language)?.has(unit.sourceId))
      throw new Error('Invalid code unit source reference')
  }
  return data
}

export async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('')
}

export async function readCodeShard(
  bytes: Uint8Array<ArrayBuffer>,
  reference: CodeShardReference,
  generation: string,
): Promise<CodeSourceShard> {
  if (bytes.byteLength !== reference.byteLength) throw new Error('Code shard length mismatch')
  if ((await sha256(bytes)) !== reference.sha256) throw new Error('Code shard integrity mismatch')
  const parsed: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))
  const result = sourceShard.safeParse(parsed)
  if (!result.success) throw new Error('Invalid code source shard')
  const data = result.data
  const expected = new Set(reference.sourceIds)
  if (
    data.generation !== generation ||
    data.language !== reference.language ||
    data.sources.length !== expected.size ||
    new Set(data.sources.map(source => source.id)).size !== expected.size ||
    data.sources.some(source => !expected.has(source.id))
  )
    throw new Error('Invalid code shard generation or sources')
  for (const source of data.sources) {
    if ((await sha256(new TextEncoder().encode(source.text))) !== source.id)
      throw new Error('Code source integrity mismatch')
  }
  return data
}
