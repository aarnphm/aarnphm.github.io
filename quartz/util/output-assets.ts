import type { Dirent } from 'node:fs'
import { randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import type { BuildCtx } from './ctx'
import type { FilePath } from './path'
import { copyFile, resetCopyFileCache } from './copy-file'
import { isRecord } from './type-guards'

export type OutputAssetOwner =
  | 'content-asset'
  | 'quartz-static'
  | 'notebook-runtime'
  | 'notebook-asset'

export type OutputAssetClaim = {
  owner: OutputAssetOwner
  source: FilePath | string
  output: FilePath | string
}

export type OutputAssetEntry = OutputAssetClaim & { size: number; mtimeMs: number }

export type OutputAssetManifest = Map<string, OutputAssetEntry>

type OutputAssetClaimStore = Record<OutputAssetOwner, Set<FilePath>>

declare global {
  var __quartzOutputAssetClaims: OutputAssetClaimStore | undefined
}

const manifestPath = path.join('quartz', '.quartz-cache', 'output-assets.json')

type SerializedOutputAssetManifest = { version: 1; assets: OutputAssetEntry[] }

function normalizeFilePath(fp: string): FilePath {
  return path.resolve(fp) as FilePath
}

function isOutputAssetOwner(value: unknown): value is OutputAssetOwner {
  return (
    value === 'content-asset' ||
    value === 'quartz-static' ||
    value === 'notebook-runtime' ||
    value === 'notebook-asset'
  )
}

function outputAssetClaimStore(): OutputAssetClaimStore {
  const existing = globalThis.__quartzOutputAssetClaims
  if (existing) {
    existing['content-asset'] ??= new Set()
    existing['quartz-static'] ??= new Set()
    existing['notebook-runtime'] ??= new Set()
    existing['notebook-asset'] ??= new Set()
    return existing
  }
  const created: OutputAssetClaimStore = {
    'content-asset': new Set(),
    'quartz-static': new Set(),
    'notebook-runtime': new Set(),
    'notebook-asset': new Set(),
  }
  globalThis.__quartzOutputAssetClaims = created
  return created
}

function normalizeAssetList(files: FilePath | string | Iterable<FilePath | string>): FilePath[] {
  if (typeof files === 'string') {
    return [normalizeFilePath(files)]
  }
  const result: FilePath[] = []
  for (const file of files) {
    result.push(normalizeFilePath(file))
  }
  return result
}

export function recordOutputAsset(
  owner: OutputAssetOwner,
  files: FilePath | string | Iterable<FilePath | string>,
): void {
  const store = outputAssetClaimStore()[owner]
  for (const file of normalizeAssetList(files)) {
    store.add(file)
  }
}

export function replaceOutputAssets(
  owner: OutputAssetOwner,
  files: Iterable<FilePath | string>,
): void {
  const store = outputAssetClaimStore()[owner]
  store.clear()
  for (const file of files) {
    store.add(normalizeFilePath(file))
  }
}

export function getPreservedOutputFiles(): Set<FilePath> {
  const store = outputAssetClaimStore()
  const preserved = new Set<FilePath>()
  for (const owner of [
    'content-asset',
    'quartz-static',
    'notebook-runtime',
    'notebook-asset',
  ] as const) {
    for (const file of store[owner]) {
      preserved.add(file)
    }
  }
  return preserved
}

export function resetOutputAssetClaims(owner?: OutputAssetOwner): void {
  const store = outputAssetClaimStore()
  if (owner) {
    store[owner].clear()
    return
  }
  store['content-asset'].clear()
  store['quartz-static'].clear()
  store['notebook-runtime'].clear()
  store['notebook-asset'].clear()
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
    isOutputAssetOwner(value.owner) &&
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
    if (entry.owner === 'notebook-runtime' || entry.owner === 'notebook-asset') {
      if (await outputMatches(entry.output, entry.size)) {
        preserved.add(normalizeFilePath(entry.output))
      }
      continue
    }
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
  const store = outputAssetClaimStore()
  for (const owner of ['notebook-runtime', 'notebook-asset'] as const) {
    for (const file of store[owner]) {
      try {
        const stat = await fs.stat(file)
        if (stat.isFile()) {
          preserved.add(normalizeFilePath(file))
        }
      } catch {}
    }
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
  recordOutputAsset(normalized.owner, normalized.output)
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

  await copyFile(
    normalized.source as FilePath,
    normalized.output as FilePath,
    ctx.incremental ? ctx.argv.output : undefined,
  )
  manifest.set(manifestKey(normalized.output), { ...normalized, ...fingerprint })
  return normalized.output as FilePath
}

export async function removeOutputAsset(
  ctxOrOwner: BuildCtx | OutputAssetOwner | FilePath | string,
  output?: FilePath | string | Iterable<FilePath | string>,
): Promise<void> {
  const store = outputAssetClaimStore()
  if (isOutputAssetOwner(ctxOrOwner)) {
    if (output !== undefined) {
      for (const file of normalizeAssetList(output)) {
        store[ctxOrOwner].delete(file)
      }
    }
    return
  }
  if (typeof ctxOrOwner === 'string') {
    const normalized = normalizeFilePath(ctxOrOwner)
    for (const owner of [
      'content-asset',
      'quartz-static',
      'notebook-runtime',
      'notebook-asset',
    ] as const) {
      store[owner].delete(normalized)
    }
    return
  }
  if (typeof output !== 'string') return
  const normalized = normalizeFilePath(output)
  for (const owner of [
    'content-asset',
    'quartz-static',
    'notebook-runtime',
    'notebook-asset',
  ] as const) {
    store[owner].delete(normalized)
  }
  ensureOutputAssetManifest(ctxOrOwner).delete(manifestKey(normalized))
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
  const store = outputAssetClaimStore()
  const dynamicClaims: OutputAssetClaim[] = []
  for (const owner of ['notebook-runtime', 'notebook-asset'] as const) {
    for (const output of store[owner]) {
      try {
        const fingerprint = await statFingerprint(output)
        const claim: OutputAssetClaim = { owner, source: output, output }
        dynamicClaims.push(claim)
        ctx.outputAssetManifest.set(manifestKey(output), { ...claim, ...fingerprint })
      } catch {}
    }
  }
  pruneOutputAssetManifest(ctx.outputAssetManifest, [
    ...(ctx.outputAssetClaims ?? []),
    ...dynamicClaims,
  ])
  await writeOutputAssetManifest(ctx.outputAssetManifest)
}

/** Move output identities with a staged build, retaining original source identities. */
export function relocateOutputAssets(ctx: BuildCtx, from: string, to: string): void {
  const root = path.resolve(from)
  const relocate = (file: string): FilePath => {
    const resolved = path.resolve(file)
    return (
      resolved.startsWith(`${root}${path.sep}`)
        ? path.join(to, path.relative(root, resolved))
        : resolved
    ) as FilePath
  }
  const claim = (entry: OutputAssetClaim): OutputAssetClaim => ({
    ...entry,
    source: relocate(entry.source),
    output: relocate(entry.output),
  })
  if (ctx.outputAssetManifest) {
    ctx.outputAssetManifest = new Map(
      [...ctx.outputAssetManifest.values()].map(entry => {
        const moved = { ...entry, ...claim(entry) }
        return [manifestKey(moved.output), moved]
      }),
    )
  }
  ctx.outputAssetClaims = ctx.outputAssetClaims?.map(claim)
  if (ctx.outputAssetPreserved)
    ctx.outputAssetPreserved = new Set([...ctx.outputAssetPreserved].map(relocate))
  for (const files of Object.values(outputAssetClaimStore())) {
    const moved = [...files].map(relocate)
    files.clear()
    for (const file of moved) files.add(file)
  }
}
