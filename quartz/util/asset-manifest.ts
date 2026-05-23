import { createHash, type BinaryLike } from 'node:crypto'
import type { BuildCtx } from './ctx'
import type { FullSlug } from './path'

export type AssetManifest = Map<string, string>

export function hashAssetContent(content: BinaryLike): string {
  return createHash('sha256').update(content).digest('hex').slice(0, 8)
}

export function shouldHashAssets(ctx: BuildCtx): boolean {
  return !ctx.argv.serve && !ctx.argv.watch && !ctx.incremental
}

export function ensureAssetManifest(ctx: BuildCtx): AssetManifest {
  ctx.assetManifest ??= new Map()
  return ctx.assetManifest
}

export function registerAsset(ctx: BuildCtx, logicalPath: string, emittedPath: string): string {
  ensureAssetManifest(ctx).set(logicalPath, emittedPath)
  return emittedPath
}

export function resolveAsset(ctx: BuildCtx, logicalPath: string): string {
  return ctx.assetManifest?.get(logicalPath) ?? logicalPath
}

export function assetPath(slug: string, ext: `.${string}`): string {
  return `${slug}${ext}`
}

export function assetSlugForContent(
  ctx: BuildCtx,
  slug: string,
  ext: `.${string}`,
  content: BinaryLike,
): FullSlug {
  const emittedSlug = shouldHashAssets(ctx) ? `${slug}-${hashAssetContent(content)}` : slug
  registerAsset(ctx, assetPath(slug, ext), assetPath(emittedSlug, ext))
  return emittedSlug as FullSlug
}

export function assetManifestRecord(ctx: BuildCtx): Record<string, string> {
  return Object.fromEntries(
    [...ensureAssetManifest(ctx).entries()].sort(([left], [right]) => left.localeCompare(right)),
  )
}
