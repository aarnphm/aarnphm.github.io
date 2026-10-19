import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import satori, { SatoriOptions } from 'satori'
import sharp from 'sharp'
import { i18n } from '../../i18n'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { copyFile } from '../../util/copy-file'
import { BuildCtx } from '../../util/ctx'
import { descriptionToPlainText } from '../../util/description'
import { getIconCode } from '../../util/emoji'
import { loadEmoji } from '../../util/emoji-node'
import { unescapeHTML } from '../../util/escape'
import { onModuleRelease } from '../../util/module-caches'
import {
  ImageOptions,
  SocialImageOptions,
  defaultImage,
  defaultImageMeta,
  getSatoriFonts,
} from '../../util/og'
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

function isMissingFileError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
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

let sectionIconsPromise: Promise<Map<Section, SectionIcon>> | null = null
const satoriFontCache = new Map<
  string,
  Promise<{ fonts: SatoriOptions['fonts']; fontsSignature: string }>
>()
const inFlightOgImages = new Map<FilePath, Promise<void>>()

onModuleRelease(() => {
  sectionIconsPromise = null
  satoriFontCache.clear()
  inFlightOgImages.clear()
})

// The section mascots (quartz/static/icon-<section>.png), shrunk for the card's header. A missing
// file leaves that section's cards without an icon.
function loadSectionIcons(): Promise<Map<Section, SectionIcon>> {
  if (sectionIconsPromise) return sectionIconsPromise
  sectionIconsPromise = (async () => {
    const entries = await Promise.all(
      SECTIONS.map(async section => {
        const file = joinSegments(QUARTZ, 'static', `icon-${section}.png`)
        if (!(await pathExists(file as FilePath))) return null
        const png = await sharp(file).resize(192, 192).png().toBuffer()
        return [
          section,
          {
            dataUri: `data:image/png;base64,${png.toString('base64')}`,
            hash: createHash('sha256').update(png).digest('base64url'),
          },
        ] as const
      }),
    )
    const icons = new Map<Section, SectionIcon>()
    for (const entry of entries) {
      if (entry) icons.set(entry[0], entry[1])
    }
    return icons
  })()
  return sectionIconsPromise
}

function loadCachedSatoriFonts(
  cfg: BuildCtx['cfg']['configuration'],
): Promise<{ fonts: SatoriOptions['fonts']; fontsSignature: string }> {
  const key = JSON.stringify({
    baseUrl: cfg.baseUrl,
    header: cfg.theme.typography.header,
    body: cfg.theme.typography.body,
  })
  const cached = satoriFontCache.get(key)
  if (cached) return cached
  const pending = (async () => {
    const fonts = await getSatoriFonts(cfg, cfg.theme.typography.header, cfg.theme.typography.body)
    return { fonts, fontsSignature: fontSignature(fonts) }
  })()
  satoriFontCache.set(key, pending)
  return pending
}

let cacheDirReady: Promise<string | undefined> | null = null

async function ensureCachedOgImage(cachePath: FilePath, content: Buffer): Promise<void> {
  cacheDirReady ??= fs.mkdir(path.dirname(cachePath), { recursive: true })
  await cacheDirReady
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

function computeBaseSignature(
  cfg: BuildCtx['cfg']['configuration'],
  fullOptions: SocialImageOptions,
  fontsSignature: string,
): string {
  return createHash('sha256')
    .update(
      JSON.stringify({
        baseUrl: cfg.baseUrl,
        colorScheme: fullOptions.colorScheme,
        excludeRoot: fullOptions.excludeRoot,
        font: fontsSignature,
        height: fullOptions.height,
        imageStructure:
          fullOptions.imageStructure === defaultImage
            ? 'defaultImage'
            : fullOptions.imageStructure.toString(),
        locale: cfg.locale,
        pageTitleSuffix: cfg.pageTitleSuffix,
        theme: cfg.theme,
        width: fullOptions.width,
      }),
    )
    .digest('base64url')
}

function ogImageSignature(
  cfg: BuildCtx['cfg']['configuration'],
  fileData: QuartzPluginData,
  title: string,
  description: string,
  fullOptions: SocialImageOptions,
  baseSignature: string,
  icon: SectionIcon | undefined,
): string {
  if (fullOptions.imageStructure === defaultImage) {
    const { date, readingTimeText, tags } = defaultImageMeta(cfg, fileData)
    return createHash('sha256')
      .update(
        JSON.stringify({
          base: baseSignature,
          date,
          description,
          icon: icon?.hash,
          readingTimeText,
          tags,
          title,
        }),
      )
      .digest('base64url')
  }

  return createHash('sha256')
    .update(
      JSON.stringify({
        base: baseSignature,
        dates: fileData.dates,
        description,
        frontmatter: fileData.frontmatter,
        icon: icon?.hash,
        slug: fileData.slug,
        text: fileData.text,
        title,
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
  baseSignature: string,
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
    cfg,
    fileData,
    title,
    description,
    fullOptions,
    baseSignature,
    icon,
  )
  const cachePath = path.join(cacheDir, `${signature}.webp`) as FilePath
  const trackedRoot = ctx.incremental ? ctx.argv.output : undefined

  try {
    return await copyFile(cachePath, dest, trackedRoot)
  } catch (error) {
    if (!isMissingFileError(error)) throw error
  }

  let pending = inFlightOgImages.get(cachePath)
  if (!pending) {
    pending = (async () => {
      const content = await generateSocialImage(
        { title, description, fonts, cfg, fileData },
        fullOptions,
        icon?.dataUri,
      )
      await ensureCachedOgImage(cachePath, content)
    })().finally(() => {
      inFlightOgImages.delete(cachePath)
    })
    inFlightOgImages.set(cachePath, pending)
  }
  await pending

  return copyFile(cachePath, dest, trackedRoot)
}

export async function createOgImageGenerator(
  ctx: BuildCtx,
  userOpts: Partial<SocialImageOptions> = {},
): Promise<(fileData: QuartzPluginData) => Promise<FilePath>> {
  const fullOptions = { ...defaultOptions, ...userOpts }
  const cfg = ctx.cfg.configuration
  const [{ fonts, fontsSignature }, icons] = await Promise.all([
    loadCachedSatoriFonts(cfg),
    loadSectionIcons(),
  ])
  const baseSignature = computeBaseSignature(cfg, fullOptions, fontsSignature)
  return fileData => processOgImage(ctx, fileData, fonts, fullOptions, baseSignature, icons)
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
      const targets = content.filter(
        ([_tree, vfile]) => vfile.data.frontmatter?.socialImage === undefined,
      )
      if (targets.length === 0) return
      const generateOgImage = await createOgImageGenerator(ctx, fullOptions)
      const emitted = await mapConcurrent(targets, defaultIoConcurrency, ([_tree, vfile]) =>
        generateOgImage(vfile.data),
      )
      yield* emitted
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      const targets = changeEvents.filter(
        changeEvent =>
          changeEvent.file &&
          changeEvent.file.data.frontmatter?.socialImage === undefined &&
          (changeEvent.type === 'add' || changeEvent.type === 'change'),
      )
      if (targets.length === 0) return
      const generateOgImage = await createOgImageGenerator(ctx, fullOptions)
      const emitted = await mapConcurrent(targets, defaultIoConcurrency, changeEvent =>
        generateOgImage(changeEvent.file!.data),
      )
      yield* emitted
    },
    externalResources: ctx => socialImageResources(ctx, fullOptions, true),
  }
}
