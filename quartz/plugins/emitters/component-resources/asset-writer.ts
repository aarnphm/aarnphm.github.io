import path from 'path'
import type { BuildCtx } from '../../../util/ctx'
import type { FilePath, FullSlug } from '../../../util/path'
import {
  assetManifestRecord,
  assetPath,
  assetSlugForContent,
  resolveAsset,
} from '../../../util/asset-manifest'
import { write } from '../helpers'
import { staticScriptsDir } from './asset-paths'

export type AssetBundleOutput = { path: string; text: string }

export async function writeAssetBundleOutput(
  ctx: BuildCtx,
  outputFile: AssetBundleOutput,
): Promise<FilePath> {
  const rel = path.relative(ctx.argv.output, outputFile.path).split(path.sep).join('/')
  const ext = path.extname(rel) as `.${string}`
  const logicalSlug = rel.slice(0, -ext.length)
  const slug = rel.includes('/chunks/')
    ? (logicalSlug as FullSlug)
    : assetSlugForContent(ctx, logicalSlug, ext, outputFile.text)
  return write({ ctx, slug, ext, content: outputFile.text })
}

export async function writeRawAsset(
  ctx: BuildCtx,
  logicalPath: string,
  content: string | Buffer,
): Promise<FilePath> {
  const ext = path.extname(logicalPath) as `.${string}`
  const logicalSlug = logicalPath.slice(0, -ext.length)
  const slug = assetSlugForContent(ctx, logicalSlug, ext, content)
  return write({ ctx, slug, ext, content })
}

export function resolveAssetPath(ctx: BuildCtx, logicalPath: string): string {
  return resolveAsset(ctx, logicalPath)
}

export function staticScriptAssetReference(ctx: BuildCtx, logicalPath: string): string {
  return path.relative(staticScriptsDir, resolveAsset(ctx, logicalPath)).split(path.sep).join('/')
}

export function relativeAssetReference(fromLogicalPath: string, toPath: string): string {
  return path.relative(path.dirname(fromLogicalPath), toPath).split(path.sep).join('/')
}

export function relativeBundleAssetReference(fromFile: string, toFile: string): string {
  return path.relative(path.dirname(fromFile), toFile).split(path.sep).join('/')
}

export async function writeAssetManifest(ctx: BuildCtx): Promise<FilePath> {
  return write({
    ctx,
    slug: 'static/scripts/asset-manifest' as FullSlug,
    ext: '.json',
    content: JSON.stringify(assetManifestRecord(ctx)),
  })
}

export { assetPath, assetSlugForContent }
