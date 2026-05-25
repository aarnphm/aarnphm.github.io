import type { BuildCtx } from '../../../util/ctx'
import type { FilePath } from '../../../util/path'
import { relativeAssetReference, resolveAssetPath, writeRawAsset } from './asset-writer'

export type ChunkedFileAssetDescriptor = {
  baseDir: string
  manifestName: string
  chunkDir: string
  maxBytes: number
}

export function chunkedFileAssetManifestPath(descriptor: ChunkedFileAssetDescriptor): string {
  return `${descriptor.baseDir}/${descriptor.manifestName}`
}

export function chunkedFileAssetChunkPath(
  descriptor: ChunkedFileAssetDescriptor,
  index: number,
): string {
  return `${descriptor.baseDir}/${descriptor.chunkDir}/${index}.json`
}

export function chunkedFileAssetChunkReference(
  descriptor: ChunkedFileAssetDescriptor,
  chunkPath: string,
): string {
  return relativeAssetReference(chunkedFileAssetManifestPath(descriptor), chunkPath)
}

export async function writeChunkedFileAsset(
  ctx: BuildCtx,
  descriptor: ChunkedFileAssetDescriptor,
  chunks: readonly string[],
): Promise<FilePath[]> {
  const chunkPaths = chunks.map((_chunk, index) => chunkedFileAssetChunkPath(descriptor, index))
  const chunkFiles = await Promise.all(
    chunks.map((chunk, index) => writeRawAsset(ctx, chunkPaths[index], chunk)),
  )
  const manifest = {
    chunks: chunkPaths.map(chunkPath =>
      chunkedFileAssetChunkReference(descriptor, resolveAssetPath(ctx, chunkPath)),
    ),
  }
  const manifestFile = await writeRawAsset(
    ctx,
    chunkedFileAssetManifestPath(descriptor),
    JSON.stringify(manifest),
  )
  return [manifestFile, ...chunkFiles]
}
