import fs from 'node:fs/promises'
import path from 'node:path'
import type { BuildCtx } from '../ctx'
import { write } from '../../plugins/emitters/helpers'
import { hashAssetContent } from '../asset-manifest'
import { defaultIoConcurrency, mapConcurrent } from '../async-pool'
import { copyFile } from '../copy-file'
import { recordOutputAsset } from '../output-assets'
import { QUARTZ, joinSegments, type FilePath, type FullSlug } from '../path'
import { logBuildSpan, PerfTimer } from '../perf'

const DATA_URI_PATTERN = /data:(image\/(?:png|jpe?g|gif|svg\+xml));base64,([A-Za-z0-9+/=\s]+)/g

const MIN_EXTRACTION_BYTES = 2048

const NOTEBOOK_ASSETS_DIR = 'static/notebook-assets'
const NOTEBOOK_ASSET_CACHE_DIR = path.join(QUARTZ, '.quartz-cache', 'notebook-assets')

const MIME_EXTENSION: Record<string, `.${string}`> = {
  'image/png': '.png',
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'image/gif': '.gif',
  'image/svg+xml': '.svg',
}

export type ExtractedAsset = {
  readonly hash: string
  readonly url: string
  readonly mime: string
  readonly byteLength: number
}

export type AssetExtractionResult = {
  readonly chunks: string[]
  readonly extracted: ExtractedAsset[]
}

type NotebookAssetWriteState = { buildId: string; keys: Set<string> }
type NotebookAssetWriteCache = WeakMap<BuildCtx, NotebookAssetWriteState>

declare global {
  var __quartzNotebookAssetWrites: NotebookAssetWriteCache | undefined
}

function decodeBase64(input: string): Uint8Array {
  const stripped = input.replace(/\s+/g, '')
  if (typeof globalThis.atob === 'function') {
    const binary = globalThis.atob(stripped)
    const out = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i)
    return out
  }
  const buffer = Buffer.from(stripped, 'base64')
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
}

function mimeExtension(mime: string): `.${string}` | undefined {
  return MIME_EXTENSION[mime.toLowerCase()]
}

const writtenAssets = (globalThis.__quartzNotebookAssetWrites ??= new WeakMap<
  BuildCtx,
  NotebookAssetWriteState
>())

function rememberWritten(ctx: BuildCtx, key: string): boolean {
  let state = writtenAssets.get(ctx)
  if (!state || state.buildId !== ctx.buildId) {
    state = { buildId: ctx.buildId, keys: new Set() }
    writtenAssets.set(ctx, state)
  }
  if (state.keys.has(key)) return false
  state.keys.add(key)
  return true
}

function errorCode(error: unknown): string | undefined {
  if (!(error instanceof Error) || !('code' in error)) return undefined
  const code = error.code
  return typeof code === 'string' ? code : undefined
}

function isExistingFileError(error: unknown): boolean {
  return errorCode(error) === 'EEXIST'
}

async function outputAssetExists(dest: FilePath, expectedBytes: number): Promise<boolean> {
  try {
    const stat = await fs.stat(dest)
    return stat.isFile() && stat.size === expectedBytes
  } catch {
    return false
  }
}

async function ensureCachedNotebookAsset(cachePath: FilePath, content: Buffer): Promise<void> {
  await fs.mkdir(path.dirname(cachePath), { recursive: true })
  try {
    await fs.writeFile(cachePath, content, { flag: 'wx' })
  } catch (error) {
    if (!isExistingFileError(error)) throw error
  }
}

async function tryCopyCachedNotebookAsset(cachePath: FilePath, dest: FilePath): Promise<boolean> {
  try {
    await copyFile(cachePath, dest)
    return true
  } catch (error) {
    if (errorCode(error) !== 'ENOENT') throw error
    return false
  }
}

async function writeExtractedNotebookAsset(
  ctx: BuildCtx,
  slug: FullSlug,
  ext: `.${string}`,
  hash: string,
  content: Buffer,
): Promise<void> {
  const dest = joinSegments(ctx.argv.output, `${slug}${ext}`) as FilePath
  recordOutputAsset('notebook-asset', [dest])
  const cachePath = path.join(NOTEBOOK_ASSET_CACHE_DIR, `${hash}${ext}`) as FilePath
  if (await outputAssetExists(dest, content.byteLength)) {
    if (ctx.argv.watch && !(await outputAssetExists(cachePath, content.byteLength))) {
      await ensureCachedNotebookAsset(cachePath, content).catch(() => undefined)
    }
    return
  }

  if (!ctx.argv.watch) {
    await write({ ctx, slug, ext, content })
    await ensureCachedNotebookAsset(cachePath, content).catch(() => undefined)
    return
  }

  const perf = new PerfTimer()
  if (!(await tryCopyCachedNotebookAsset(cachePath, dest))) {
    await ensureCachedNotebookAsset(cachePath, content)
    await copyFile(cachePath, dest)
  }
  logBuildSpan(ctx.argv, 'write:notebook-asset', dest, perf.elapsedMs())
}

export async function ensureExtractedNotebookAssets(
  ctx: BuildCtx,
  extracted: readonly ExtractedAsset[],
): Promise<boolean> {
  for (const asset of extracted) {
    const ext = mimeExtension(asset.mime) ?? (path.extname(asset.url) as `.${string}`)
    if (!ext) return false
    const slug = `${NOTEBOOK_ASSETS_DIR}/${asset.hash}` as FullSlug
    const dest = joinSegments(ctx.argv.output, `${slug}${ext}`) as FilePath
    recordOutputAsset('notebook-asset', [dest])
    if (!rememberWritten(ctx, `${slug}${ext}`)) continue
    if (await outputAssetExists(dest, asset.byteLength)) continue
    const cachePath = path.join(NOTEBOOK_ASSET_CACHE_DIR, `${asset.hash}${ext}`) as FilePath
    if (!(await tryCopyCachedNotebookAsset(cachePath, dest))) {
      return false
    }
  }
  return true
}

export async function extractInlineNotebookAssets(
  chunks: readonly string[],
  ctx: BuildCtx,
): Promise<AssetExtractionResult> {
  const extracted: ExtractedAsset[] = []
  const writes: Array<() => Promise<void>> = []
  const updated = chunks.map(chunk =>
    chunk.replace(DATA_URI_PATTERN, (match, mime: string, payload: string) => {
      const ext = mimeExtension(mime)
      // SVG stays inline: themeNotebookSvgs turns it into themed markup.
      if (!ext || mime === 'image/svg+xml') return match
      let bytes: Uint8Array
      try {
        bytes = decodeBase64(payload)
      } catch {
        return match
      }
      if (bytes.byteLength < MIN_EXTRACTION_BYTES) return match
      const hash = hashAssetContent(bytes)
      const slug = `${NOTEBOOK_ASSETS_DIR}/${hash}` as FullSlug
      const url = `/${slug}${ext}`
      const dest = joinSegments(ctx.argv.output, `${slug}${ext}`) as FilePath
      recordOutputAsset('notebook-asset', [dest])
      if (rememberWritten(ctx, `${slug}${ext}`)) {
        const content = Buffer.from(bytes)
        writes.push(() => writeExtractedNotebookAsset(ctx, slug, ext, hash, content))
      }
      extracted.push({ hash, url, mime, byteLength: bytes.byteLength })
      return url
    }),
  )
  await mapConcurrent(writes, defaultIoConcurrency, write => write())
  return { chunks: updated, extracted }
}

export const notebookAssetDirectory = NOTEBOOK_ASSETS_DIR
export const notebookAssetMinBytes = MIN_EXTRACTION_BYTES
