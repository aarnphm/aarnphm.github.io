import { Features, transform } from 'lightningcss'
import path from 'path'
import { compileAsync } from 'sass-embedded'
import type { BuildCtx } from '../../../util/ctx'
import type { FilePath, FullSlug } from '../../../util/path'
import type { StaticResources } from '../../../util/resources'
import type { ComponentResourceSet } from './resource-set'
import collapseHeaderStyle from '../../../components/styles/collapseHeader.inline.scss'
import {
  assetReferenceForContent,
  assetPath,
  contentHashSlug,
  ensureExtractedStaticResources,
  registerExtractedStaticResource,
  shouldHashAssets,
} from '../../../util/asset-manifest'
import { joinSegments } from '../../../util/path'
import {
  componentCssBundleKey,
  componentCssResourceKeyPrefix,
  splitCssBundles,
  staticCssBundleKey,
  staticCssBundleSlug,
} from '../../../util/resource-bundles'
import { googleFontHref, joinStyles, processGoogleFonts } from '../../../util/theme'
import { write } from '../helpers'
import { quartzBaseStylesheetEntry, quartzCustomStylesheetEntry } from './asset-paths'
import { assetSlugForContent } from './asset-writer'

export type FontAssetResult = { googleFontsStyleSheet: string; files: FilePath[] }

const componentCssBundleSlug = 'static/component'

function extractedCssSlug(ctx: BuildCtx, index: number, content: string): FullSlug {
  if (shouldHashAssets(ctx)) return contentHashSlug(staticCssBundleSlug, content)
  return (index === 0 ? staticCssBundleSlug : `${staticCssBundleSlug}-${index}`) as FullSlug
}

export function minifyStylesheet(filename: string, stylesheet: string): string {
  return transform({
    filename,
    code: Buffer.from(stylesheet),
    minify: true,
    targets: {
      safari: (15 << 16) | (6 << 8),
      ios_saf: (15 << 16) | (6 << 8),
      edge: 115 << 16,
      firefox: 102 << 16,
      chrome: 109 << 16,
    },
    include: Features.MediaQueries,
  }).code.toString()
}

async function compileStylesheet(sourcePath: string): Promise<string> {
  const result = await compileAsync(path.resolve(sourcePath), { charset: false, style: 'expanded' })
  return result.css
}

export async function writeFontAssets(ctx: BuildCtx): Promise<FontAssetResult> {
  const cfg = ctx.cfg.configuration
  if (cfg.theme.fontOrigin === 'local') {
    return { googleFontsStyleSheet: '', files: [] }
  }
  if (cfg.theme.fontOrigin !== 'googleFonts' || cfg.theme.cdnCaching) {
    return { googleFontsStyleSheet: '', files: [] }
  }

  const response = await fetch(googleFontHref(cfg.theme))
  let googleFontsStyleSheet = await response.text()

  if (!cfg.baseUrl) {
    throw new Error('baseUrl must be defined when using Google Fonts without cfg.theme.cdnCaching')
  }

  const { processedStylesheet, fontFiles } = await processGoogleFonts(
    googleFontsStyleSheet,
    cfg.baseUrl,
  )
  googleFontsStyleSheet = processedStylesheet

  const files: FilePath[] = []
  for (const fontFile of fontFiles) {
    const res = await fetch(fontFile.url)
    if (!res.ok) {
      throw new Error(`failed to fetch font ${fontFile.filename}`)
    }

    const buf = await res.arrayBuffer()
    files.push(
      await write({
        ctx,
        slug: joinSegments('static', 'fonts', fontFile.filename) as FullSlug,
        ext: `.${fontFile.extension}`,
        content: Buffer.from(buf),
      }),
    )
  }

  return { googleFontsStyleSheet, files }
}

export async function* writeStaticCssResourceBundles(
  ctx: BuildCtx,
  resources: StaticResources,
): AsyncGenerator<FilePath> {
  let index = 0
  for (const part of splitCssBundles(resources.css, [collapseHeaderStyle])) {
    if (part.type !== 'bundle') continue
    const content = minifyStylesheet('resource-style.css', part.content)
    const slug = extractedCssSlug(ctx, index, content)
    index += 1
    registerExtractedStaticResource(ctx, staticCssBundleKey(part.content), assetPath(slug, '.css'))
    yield write({ ctx, slug, ext: '.css', content })
  }
}

export async function* writeComponentStyles(
  ctx: BuildCtx,
  resources: ComponentResourceSet,
): AsyncGenerator<FilePath> {
  const extractedResources = ensureExtractedStaticResources(ctx)
  for (const key of extractedResources.keys()) {
    if (key.startsWith(componentCssResourceKeyPrefix)) {
      extractedResources.delete(key)
    }
  }

  if (resources.componentCss.length === 0) {
    return
  }

  const stylesheet = resources.componentCss.join('\n')
  const content = minifyStylesheet('component.css', `@layer quartz-base {\n${stylesheet}\n}`)
  const asset = assetReferenceForContent(ctx, componentCssBundleSlug, '.css', content)
  registerExtractedStaticResource(ctx, componentCssBundleKey, asset.path)
  yield write({ ctx, slug: asset.slug, ext: '.css', content })
}

async function indexStylesheetContent(
  ctx: BuildCtx,
  componentResources: ComponentResourceSet,
  googleFontsStyleSheet: string,
): Promise<string> {
  const [baseStyles, customStyles] = await Promise.all([
    compileStylesheet(quartzBaseStylesheetEntry),
    compileStylesheet(quartzCustomStylesheetEntry),
  ])
  const componentCss = new Set(componentResources.componentCss)
  const quartzBase = joinStyles(
    ctx.cfg.configuration.theme,
    googleFontsStyleSheet,
    ...componentResources.css.filter(css => !componentCss.has(css)),
    baseStyles,
  )
  return minifyStylesheet('index.css', `@layer quartz-base {\n${quartzBase}\n}\n${customStyles}`)
}

export async function writeIndexStylesheet(
  ctx: BuildCtx,
  componentResources: ComponentResourceSet,
  googleFontsStyleSheet = '',
): Promise<FilePath> {
  const content = await indexStylesheetContent(ctx, componentResources, googleFontsStyleSheet)
  return write({
    ctx,
    slug: assetSlugForContent(ctx, 'index', '.css', content),
    ext: '.css',
    content,
  })
}
