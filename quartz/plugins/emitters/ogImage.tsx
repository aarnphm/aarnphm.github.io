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
import { StaticResources } from '../../util/resources'
import { SECTIONS, type Section, isSlidePage, sectionOf } from '../../util/section-icon'
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

type SectionIcon = { dataUri: string; hash: string }

// The section mascots (quartz/static/icon-<section>.png), shrunk for the card's header. A missing
// file leaves that section's cards without an icon.
async function loadSectionIcons(): Promise<Map<Section, SectionIcon>> {
  const icons = new Map<Section, SectionIcon>()
  for (const section of SECTIONS) {
    const file = joinSegments(QUARTZ, 'static', `icon-${section}.png`)
    if (!(await pathExists(file as FilePath))) continue
    const png = await sharp(file).resize(192, 192).png().toBuffer()
    icons.set(section, {
      dataUri: `data:image/png;base64,${png.toString('base64')}`,
      hash: createHash('sha256').update(png).digest('base64url'),
    })
  }
  return icons
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
  icon: SectionIcon | undefined,
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
        icon: icon?.hash,
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
  iconBase64?: string,
): Promise<Buffer> {
  const { width, height } = userOpts
  const imageComponent = userOpts.imageStructure({
    cfg,
    userOpts,
    title,
    description,
    fonts,
    fileData,
    iconBase64,
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
  icons: Map<Section, SectionIcon>,
) {
  const cfg = ctx.cfg.configuration
  const slug = fileData.slug!
  const section = sectionOf({
    slug,
    base: fileData.bases === true,
    slides: isSlidePage(slug, fileData.frontmatter?.slides),
  })
  const icon = section ? icons.get(section) : undefined
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
  const signature = ogImageSignature(
    ctx,
    fileData,
    title,
    description,
    fullOptions,
    fontsSignature,
    icon,
  )
  const cachePath = path.join(cacheDir, `${signature}.webp`) as FilePath

  if (!(await pathExists(cachePath))) {
    const content = await generateSocialImage(
      { title, description, fonts, cfg, fileData },
      fullOptions,
      icon?.dataUri,
    )
    await ensureCachedOgImage(cachePath, content)
  }

  return copyFile(cachePath, dest, ctx.incremental ? ctx.argv.output : undefined)
}

export async function createOgImageGenerator(
  ctx: BuildCtx,
  userOpts: Partial<SocialImageOptions> = {},
): Promise<(fileData: QuartzPluginData) => Promise<FilePath>> {
  const fullOptions = { ...defaultOptions, ...userOpts }
  const cfg = ctx.cfg.configuration
  const fonts = await getSatoriFonts(cfg, cfg.theme.typography.header, cfg.theme.typography.body)
  const fontsSignature = fontSignature(fonts)
  const icons = await loadSectionIcons()
  return fileData => processOgImage(ctx, fileData, fonts, fullOptions, fontsSignature, icons)
}

export const CustomOgImagesEmitterName = 'CustomOgImages'

function socialImageResources(
  ctx: BuildCtx,
  fullOptions: SocialImageOptions,
  generated: boolean,
): Partial<StaticResources> {
  const baseUrl = ctx.cfg.configuration.baseUrl
  if (generated && !baseUrl) return {}
  const base = baseUrl ? `https://${baseUrl}` : ''
  return {
    additionalHead: [
      pageData => {
        const hasGeneratedOgImage =
          generated &&
          (pageData.filePath !== undefined || pageData.frontmatter?.generatedSocialImage === true)
        const userDefinedOgImagePath = pageData.frontmatter?.socialImage
        const generatedOgImagePath = hasGeneratedOgImage
          ? `${base}/${pageData.slug!}-og-image.webp`
          : undefined
        const defaultOgImagePath = `${base}/static/og-image.webp`
        const ogImagePath = userDefinedOgImagePath ?? generatedOgImagePath ?? defaultOgImagePath
        const ogImageMimeType = `image/${getFileExtension(ogImagePath)?.slice(1) ?? 'png'}`
        return (
          <>
            {generated && !userDefinedOgImagePath && (
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
}

export const CustomOgImagesFallback: QuartzEmitterPlugin = () => ({
  name: 'CustomOgImagesFallback',
  getQuartzComponents: () => [],
  emit: async () => [],
  externalResources: ctx => socialImageResources(ctx, defaultOptions, false),
})

export const CustomOgImages: QuartzEmitterPlugin<Partial<SocialImageOptions>> = userOpts => {
  const fullOptions = { ...defaultOptions, ...userOpts }

  return {
    name: CustomOgImagesEmitterName,
    getQuartzComponents() {
      return []
    },
    async *emit(ctx, content, _resources) {
      const generateOgImage = await createOgImageGenerator(ctx, fullOptions)
      for (const [_tree, vfile] of content) {
        if (vfile.data.frontmatter?.socialImage !== undefined) continue
        yield generateOgImage(vfile.data)
      }
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      const generateOgImage = await createOgImageGenerator(ctx, fullOptions)

      // find all slugs that changed or were added
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (changeEvent.file.data.frontmatter?.socialImage !== undefined) continue
        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          yield generateOgImage(changeEvent.file.data)
        }
      }
    },
    externalResources: ctx => socialImageResources(ctx, fullOptions, true),
  }
}
