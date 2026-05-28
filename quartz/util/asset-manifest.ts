import { createHash, type BinaryLike } from 'node:crypto'
import type { BuildCtx } from './ctx'
import type { FullSlug } from './path'

export type AssetManifest = Map<string, string>
export type ExtractedStaticResources = Map<string, string>
export type AssetContentReference = { slug: FullSlug; path: string }

export function hashAssetContent(content: BinaryLike): string {
  return createHash('sha256').update(content).digest('hex').slice(0, 8)
}

export function contentHashSlug(slug: string, content: BinaryLike): FullSlug {
  return `${slug}-${hashAssetContent(content)}` as FullSlug
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

export function ensureExtractedStaticResources(ctx: BuildCtx): ExtractedStaticResources {
  ctx.extractedStaticResources ??= new Map()
  return ctx.extractedStaticResources
}

export function registerExtractedStaticResource(
  ctx: BuildCtx,
  key: string,
  emittedPath: string,
): string {
  ensureExtractedStaticResources(ctx).set(key, emittedPath)
  return emittedPath
}

export function resolveExtractedStaticResource(ctx: BuildCtx, key: string): string {
  const emittedPath = ctx.extractedStaticResources?.get(key)
  if (!emittedPath) throw new Error(`missing extracted static resource ${key}`)
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

export function assetReferenceForContent(
  ctx: BuildCtx,
  slug: string,
  ext: `.${string}`,
  content: BinaryLike,
): AssetContentReference {
  const emittedSlug = assetSlugForContent(ctx, slug, ext, content)
  const path = assetPath(emittedSlug, ext)
  return {
    slug: emittedSlug,
    path: shouldHashAssets(ctx) ? path : `${path}?v=${hashAssetContent(content)}`,
  }
}

export function assetManifestRecord(ctx: BuildCtx): Record<string, string> {
  return Object.fromEntries(
    [...ensureAssetManifest(ctx).entries()].sort(([left], [right]) => left.localeCompare(right)),
  )
}
