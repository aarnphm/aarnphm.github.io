import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import satori, { SatoriOptions } from 'satori'
import sharp from 'sharp'
import { i18n } from '../../i18n'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { copyFile } from '../../util/copy-file'
import { BuildCtx } from '../../util/ctx'
import { descriptionToPlainText } from '../../util/description'
import { getIconCode } from '../../util/emoji'
import { loadEmoji } from '../../util/emoji-node'
import { unescapeHTML } from '../../util/escape'
import { ImageOptions, SocialImageOptions, defaultImage, getSatoriFonts } from '../../util/og'
import { FilePath, FullSlug, QUARTZ, getFileExtension, joinSegments } from '../../util/path'
import { QuartzPluginData } from '../vfile'

const defaultOptions: SocialImageOptions = {
  colorScheme: 'lightMode',
  width: 1200,
  height: 630,
  imageStructure: defaultImage,
  excludeRoot: false,
}

const cacheDir = path.join(QUARTZ, '.quartz-cache', 'og-images')

function isExistingFileError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EEXIST'
}

async function pathExists(fp: FilePath): Promise<boolean> {
  try {
    await fs.access(fp)
    return true
  } catch {
    return false
  }
}

async function ensureCachedOgImage(cachePath: FilePath, content: Buffer): Promise<void> {
  await fs.mkdir(path.dirname(cachePath), { recursive: true })
  try {
    await fs.writeFile(cachePath, content, { flag: 'wx' })
  } catch (error) {
    if (!isExistingFileError(error)) throw error
  }
}

function fontSignature(fonts: SatoriOptions['fonts']): string {
  const hash = createHash('sha256')
  for (const font of fonts) {
    hash.update(font.name)
    hash.update(String(font.weight ?? ''))
    hash.update(font.style ?? '')
    hash.update(new Uint8Array(font.data))
  }
  return hash.digest('base64url')
}

function ogImageSignature(
  ctx: BuildCtx,
  fileData: QuartzPluginData,
  title: string,
  description: string,
  fullOptions: SocialImageOptions,
  fontsSignature: string,
): string {
  const cfg = ctx.cfg.configuration
  return createHash('sha256')
    .update(
      JSON.stringify({
        baseUrl: cfg.baseUrl,
        colorScheme: fullOptions.colorScheme,
        dates: fileData.dates,
        description,
        excludeRoot: fullOptions.excludeRoot,
        font: fontsSignature,
        frontmatter: fileData.frontmatter,
        height: fullOptions.height,
        imageStructure: fullOptions.imageStructure.toString(),
        locale: cfg.locale,
        pageTitleSuffix: cfg.pageTitleSuffix,
        slug: fileData.slug,
        text: fileData.text,
        theme: cfg.theme,
        title,
        width: fullOptions.width,
      }),
    )
    .digest('base64url')
}

/**
 * Generates social image (OG/twitter standard) and saves it as `.webp` inside the public folder
 * @param opts options for generating image
 */
async function generateSocialImage(
  { cfg, description, fonts, title, fileData }: ImageOptions,
  userOpts: SocialImageOptions,
): Promise<Buffer> {
  const { width, height } = userOpts
  const imageComponent = userOpts.imageStructure({
    cfg,
    userOpts,
    title,
    description,
    fonts,
    fileData,
  })
  const svg = await satori(imageComponent, {
    width,
    height,
    fonts,
    loadAdditionalAsset: async (languageCode: string, segment: string) => {
      if (languageCode === 'emoji') {
        return `data:image/svg+xml;base64,${btoa(await loadEmoji(getIconCode(segment)))}`
      }
      return languageCode
    },
  })

  return sharp(Buffer.from(svg)).webp({ quality: 80 }).toBuffer()
}

async function processOgImage(
  ctx: BuildCtx,
  fileData: QuartzPluginData,
  fonts: SatoriOptions['fonts'],
  fullOptions: SocialImageOptions,
  fontsSignature: string,
) {
  const cfg = ctx.cfg.configuration
  const slug = fileData.slug!
  const titleSuffix = cfg.pageTitleSuffix ?? ''
  const title =
    (fileData.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title) + titleSuffix
  const rawDescription =
    fileData.frontmatter?.socialDescription ??
    fileData.rawDescription ??
    fileData.frontmatter?.description ??
    unescapeHTML(fileData.description?.trim() ?? i18n(cfg.locale).propertyDefaults.description)
  const description = descriptionToPlainText(rawDescription, slug)
  const outputSlug = `${slug}-og-image` as FullSlug
  const dest = joinSegments(ctx.argv.output, `${outputSlug}.webp`) as FilePath
  const signature = ogImageSignature(ctx, fileData, title, description, fullOptions, fontsSignature)
  const cachePath = path.join(cacheDir, `${signature}.webp`) as FilePath

  if (!(await pathExists(cachePath))) {
    const content = await generateSocialImage(
      { title, description, fonts, cfg, fileData },
      fullOptions,
    )
    await ensureCachedOgImage(cachePath, content)
  }

  return copyFile(cachePath, dest)
}

export const CustomOgImagesEmitterName = 'CustomOgImages'
export const CustomOgImages: QuartzEmitterPlugin<Partial<SocialImageOptions>> = userOpts => {
  const fullOptions = { ...defaultOptions, ...userOpts }

  return {
    name: CustomOgImagesEmitterName,
    getQuartzComponents() {
      return []
    },
    async *emit(ctx, content, _resources) {
      if (ctx.argv.watch && !ctx.argv.force) return []

      const cfg = ctx.cfg.configuration
      const headerFont = cfg.theme.typography.header
      const bodyFont = cfg.theme.typography.body
      const fonts = await getSatoriFonts(cfg, headerFont, bodyFont)
      const fontsSignature = fontSignature(fonts)
      for (const [_tree, vfile] of content) {
        if (vfile.data.frontmatter?.socialImage !== undefined) continue
        yield processOgImage(ctx, vfile.data, fonts, fullOptions, fontsSignature)
      }
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      if (ctx.argv.watch && !ctx.argv.force) return []

      const cfg = ctx.cfg.configuration
      const headerFont = cfg.theme.typography.header
      const bodyFont = cfg.theme.typography.body
      const fonts = await getSatoriFonts(cfg, headerFont, bodyFont)
      const fontsSignature = fontSignature(fonts)

      // find all slugs that changed or were added
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (changeEvent.file.data.frontmatter?.socialImage !== undefined) continue
        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          yield processOgImage(ctx, changeEvent.file.data, fonts, fullOptions, fontsSignature)
        }
      }
    },
    externalResources: ctx => {
      if (!ctx.cfg.configuration.baseUrl) {
        return {}
      }

      const baseUrl = ctx.cfg.configuration.baseUrl
      return {
        additionalHead: [
          pageData => {
            const isRealFile = pageData.filePath !== undefined
            const userDefinedOgImagePath = pageData.frontmatter?.socialImage
            const generatedOgImagePath = isRealFile
              ? `https://${baseUrl}/${pageData.slug!}-og-image.webp`
              : undefined
            const defaultOgImagePath = `https://${baseUrl}/static/og-image.webp`
            const ogImagePath = userDefinedOgImagePath ?? generatedOgImagePath ?? defaultOgImagePath

            const ogImageMimeType = `image/${getFileExtension(ogImagePath)!.slice(1) ?? 'png'}`
            return (
              <>
                {!userDefinedOgImagePath && (
                  <>
                    <meta property="og:image:width" content={fullOptions.width.toString()} />
                    <meta property="og:image:height" content={fullOptions.height.toString()} />
                  </>
                )}

                <meta property="og:image" content={ogImagePath} />
                <meta property="og:image:url" content={ogImagePath} />
                <meta name="twitter:image" content={ogImagePath} />
                <meta property="og:image:type" content={ogImageMimeType} />
              </>
            )
          },
        ],
      }
    },
  }
}
