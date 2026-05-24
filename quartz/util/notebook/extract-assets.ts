import type { BuildCtx } from '../ctx'
import type { FullSlug } from '../path'
import { write } from '../../plugins/emitters/helpers'
import { hashAssetContent } from '../asset-manifest'

const DATA_URI_PATTERN = /data:(image\/(?:png|jpe?g|gif|svg\+xml));base64,([A-Za-z0-9+/=\s]+)/g

const MIN_EXTRACTION_BYTES = 2048

const NOTEBOOK_ASSETS_DIR = 'static/notebook-assets'

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

const writtenAssets = new WeakMap<BuildCtx, Set<string>>()

function rememberWritten(ctx: BuildCtx, key: string): boolean {
  let set = writtenAssets.get(ctx)
  if (!set) {
    set = new Set()
    writtenAssets.set(ctx, set)
  }
  if (set.has(key)) return false
  set.add(key)
  return true
}

export async function extractInlineNotebookAssets(
  chunks: readonly string[],
  ctx: BuildCtx,
): Promise<AssetExtractionResult> {
  const extracted: ExtractedAsset[] = []
  const writes: Promise<void>[] = []
  const updated = chunks.map(chunk =>
    chunk.replace(DATA_URI_PATTERN, (match, mime: string, payload: string) => {
      const ext = mimeExtension(mime)
      if (!ext) return match
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
      if (rememberWritten(ctx, `${slug}${ext}`)) {
        writes.push(write({ ctx, slug, ext, content: Buffer.from(bytes) }).then(() => undefined))
      }
      extracted.push({ hash, url, mime, byteLength: bytes.byteLength })
      return url
    }),
  )
  await Promise.all(writes)
  return { chunks: updated, extracted }
}

export const notebookAssetDirectory = NOTEBOOK_ASSETS_DIR
export const notebookAssetMinBytes = MIN_EXTRACTION_BYTES
