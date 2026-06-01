import type { Dirent } from 'node:fs'
import { randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import type { BuildCtx } from './ctx'
import type { FilePath } from './path'
import { copyFile, resetCopyFileCache } from './copy-file'
import { isRecord } from './type-guards'

export type OutputAssetOwner = 'content-asset' | 'quartz-static'

export type OutputAssetClaim = {
  owner: OutputAssetOwner
  source: FilePath | string
  output: FilePath | string
}

export type OutputAssetEntry = OutputAssetClaim & { size: number; mtimeMs: number }

export type OutputAssetManifest = Map<string, OutputAssetEntry>

const manifestPath = path.join('quartz', '.quartz-cache', 'output-assets.json')

type SerializedOutputAssetManifest = { version: 1; assets: OutputAssetEntry[] }

function normalizeFilePath(fp: string): FilePath {
  return path.resolve(fp) as FilePath
}

function normalizeClaim(claim: OutputAssetClaim): OutputAssetClaim {
  return {
    owner: claim.owner,
    source: normalizeFilePath(claim.source),
    output: normalizeFilePath(claim.output),
  }
}

function manifestKey(output: string): string {
  return normalizeFilePath(output)
}

function isOutputAssetEntry(value: unknown): value is OutputAssetEntry {
  if (!isRecord(value)) return false
  return (
    (value.owner === 'content-asset' || value.owner === 'quartz-static') &&
    typeof value.source === 'string' &&
    typeof value.output === 'string' &&
    typeof value.size === 'number' &&
    typeof value.mtimeMs === 'number'
  )
}

function outputAssetEntries(value: unknown): OutputAssetEntry[] {
  if (!isRecord(value)) return []
  if (value.version !== 1 || !Array.isArray(value.assets)) return []
  return value.assets.filter(isOutputAssetEntry)
}

async function statFingerprint(
  source: string,
): Promise<Pick<OutputAssetEntry, 'size' | 'mtimeMs'>> {
  const stat = await fs.stat(source)
  return { size: stat.size, mtimeMs: stat.mtimeMs }
}

async function outputMatches(output: string, size: number): Promise<boolean> {
  try {
    const stat = await fs.stat(output)
    return stat.isFile() && stat.size === size
  } catch {
    return false
  }
}

function entryMatchesClaim(
  entry: OutputAssetEntry,
  claim: OutputAssetClaim,
  fingerprint: Pick<OutputAssetEntry, 'size' | 'mtimeMs'>,
): boolean {
  const normalized = normalizeClaim(claim)
  return (
    entry.owner === normalized.owner &&
    normalizeFilePath(entry.source) === normalized.source &&
    normalizeFilePath(entry.output) === normalized.output &&
    entry.size === fingerprint.size &&
    entry.mtimeMs === fingerprint.mtimeMs
  )
}

function claimMap(claims: readonly OutputAssetClaim[]): Map<string, OutputAssetClaim> {
  const result = new Map<string, OutputAssetClaim>()
  for (const claim of claims) {
    const normalized = normalizeClaim(claim)
    const key = manifestKey(normalized.output)
    const existing = result.get(key)
    if (existing) {
      const previous = normalizeClaim(existing)
      if (previous.owner !== normalized.owner || previous.source !== normalized.source) {
        throw new Error(
          `output asset collision for ${normalized.output}: ${previous.source} and ${normalized.source}`,
        )
      }
    }
    result.set(key, normalized)
  }
  return result
}

export async function readOutputAssetManifest(
  file: FilePath | string = manifestPath,
): Promise<OutputAssetManifest> {
  try {
    const parsed: unknown = JSON.parse(await fs.readFile(file, 'utf8'))
    return new Map(
      outputAssetEntries(parsed).map(entry => {
        const normalized = normalizeClaim(entry)
        return [
          manifestKey(normalized.output),
          { ...normalized, size: entry.size, mtimeMs: entry.mtimeMs },
        ]
      }),
    )
  } catch {
    return new Map()
  }
}

export async function writeOutputAssetManifest(
  manifest: OutputAssetManifest,
  file: FilePath | string = manifestPath,
): Promise<void> {
  const assets = [...manifest.values()].sort((left, right) =>
    left.output.localeCompare(right.output),
  )
  const payload: SerializedOutputAssetManifest = { version: 1, assets }
  const dir = path.dirname(file)
  const temp = path.join(dir, `output-assets.${process.pid}.${randomUUID()}.tmp`)
  await fs.mkdir(dir, { recursive: true })
  await fs.writeFile(temp, JSON.stringify(payload))
  await fs.rename(temp, file)
}

export function ensureOutputAssetManifest(ctx: BuildCtx): OutputAssetManifest {
  ctx.outputAssetManifest ??= new Map()
  return ctx.outputAssetManifest
}

export async function preservedOutputAssets(
  manifest: OutputAssetManifest,
  claims: readonly OutputAssetClaim[],
): Promise<Set<FilePath>> {
  const currentClaims = claimMap(claims)
  const preserved = new Set<FilePath>()
  for (const [output, entry] of manifest) {
    const claim = currentClaims.get(output)
    if (!claim) continue
    try {
      const fingerprint = await statFingerprint(claim.source)
      if (
        entryMatchesClaim(entry, claim, fingerprint) &&
        (await outputMatches(claim.output, fingerprint.size))
      ) {
        preserved.add(normalizeFilePath(claim.output))
      }
    } catch {}
  }
  return preserved
}

export async function cleanOutputExcept(
  output: string,
  preservedOutputs: ReadonlySet<FilePath | string>,
): Promise<void> {
  const preserved = new Set([...preservedOutputs].map(fp => normalizeFilePath(fp)))
  await cleanDirectory(path.resolve(output), preserved)
  resetCopyFileCache()
}

async function cleanDirectory(dir: string, preserved: ReadonlySet<FilePath>): Promise<void> {
  let entries: Dirent[]
  try {
    entries = await fs.readdir(dir, { withFileTypes: true })
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return
    throw error
  }

  for (const entry of entries) {
    const current = path.join(dir, entry.name)
    const normalized = normalizeFilePath(current)
    if (preserved.has(normalized)) continue

    if (entry.isDirectory() && !entry.isSymbolicLink()) {
      await cleanDirectory(current, preserved)
      try {
        await fs.rmdir(current)
      } catch (error) {
        if (
          error instanceof Error &&
          'code' in error &&
          (error.code === 'ENOTEMPTY' || error.code === 'ENOENT')
        ) {
          continue
        }
        throw error
      }
      continue
    }

    await fs.rm(current, { recursive: true, force: true })
  }
}

export async function emitOutputAsset(ctx: BuildCtx, claim: OutputAssetClaim): Promise<FilePath> {
  const manifest = ensureOutputAssetManifest(ctx)
  const normalized = normalizeClaim(claim)
  if (ctx.outputAssetPreserved?.has(manifestKey(normalized.output) as FilePath)) {
    return normalized.output as FilePath
  }
  const fingerprint = await statFingerprint(normalized.source)
  const existing = manifest.get(manifestKey(normalized.output))
  if (
    existing &&
    entryMatchesClaim(existing, normalized, fingerprint) &&
    (await outputMatches(normalized.output, fingerprint.size))
  ) {
    return normalized.output as FilePath
  }

  await copyFile(normalized.source as FilePath, normalized.output as FilePath)
  manifest.set(manifestKey(normalized.output), { ...normalized, ...fingerprint })
  return normalized.output as FilePath
}

export async function removeOutputAsset(ctx: BuildCtx, output: FilePath | string): Promise<void> {
  const normalized = normalizeFilePath(output)
  ensureOutputAssetManifest(ctx).delete(manifestKey(normalized))
  await fs.rm(normalized, { force: true })
}

export function pruneOutputAssetManifest(
  manifest: OutputAssetManifest,
  claims: readonly OutputAssetClaim[],
): OutputAssetManifest {
  const currentClaims = claimMap(claims)
  for (const output of manifest.keys()) {
    if (!currentClaims.has(output)) {
      manifest.delete(output)
    }
  }
  return manifest
}

export async function writeCurrentOutputAssetManifest(ctx: BuildCtx): Promise<void> {
  if (!ctx.outputAssetManifest) return
  pruneOutputAssetManifest(ctx.outputAssetManifest, ctx.outputAssetClaims ?? [])
  await writeOutputAssetManifest(ctx.outputAssetManifest)
}
