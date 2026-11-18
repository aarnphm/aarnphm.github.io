import { Root } from 'hast'
import { imageSize } from 'image-size'
import fs from 'node:fs/promises'
import path from 'path'
import { visit } from 'unist-util-visit'
import { VFile } from 'vfile'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { MasonryPage, Footer as FooterConstructor } from '../../components/'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { pathToRoot, slugifyFilePath, FilePath, FullSlug, isFilePath } from '../../util/path'
import { normalizeReferencePath } from '../../util/references/catalog'
import { referenceCatalogFor, referenceSourcePath } from '../../util/references/context'
import { parseLiteralReference, resolveLiteralReference } from '../../util/references/literal'
import { referenceFields } from '../../util/references/request'
import { StaticResources } from '../../util/resources'
import { QuartzPluginData } from '../vfile'
import { write, removeWritten } from './helpers'

const imageExtensions = new Set(['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.avif'])

const HeaderInputSize = 64 * 1024
const MaxInputSize = 512 * 1024
const masonryImageCacheFile = path.join('quartz', '.quartz-cache', 'masonry-image-sizes.json')

type MasonryImageSizeCacheEntry = { mtimeMs: number; size: number; width: number; height: number }

declare global {
  var __quartzMasonryImageSizeCache: Map<string, MasonryImageSizeCacheEntry> | undefined
  var __quartzMasonryImageSizeCacheLoaded: boolean | undefined
}

let masonryCacheDirty = false

async function ensureMasonryImageCache(): Promise<Map<string, MasonryImageSizeCacheEntry>> {
  const cache = (globalThis.__quartzMasonryImageSizeCache ??= new Map())
  if (globalThis.__quartzMasonryImageSizeCacheLoaded) return cache
  globalThis.__quartzMasonryImageSizeCacheLoaded = true
  try {
    const raw = JSON.parse(await fs.readFile(masonryImageCacheFile, 'utf8')) as Record<
      string,
      MasonryImageSizeCacheEntry
    >
    if (raw && typeof raw === 'object') {
      for (const [key, entry] of Object.entries(raw)) {
        if (
          entry &&
          typeof entry.mtimeMs === 'number' &&
          typeof entry.size === 'number' &&
          typeof entry.width === 'number' &&
          typeof entry.height === 'number'
        ) {
          cache.set(key, entry)
        }
      }
    }
  } catch {}
  return cache
}

async function flushMasonryImageCache(): Promise<void> {
  if (!masonryCacheDirty || !globalThis.__quartzMasonryImageSizeCache) return
  masonryCacheDirty = false
  try {
    await fs.mkdir(path.dirname(masonryImageCacheFile), { recursive: true })
    const serialized = Object.fromEntries(globalThis.__quartzMasonryImageSizeCache.entries())
    const temp = `${masonryImageCacheFile}.${process.pid}.${Date.now()}.tmp`
    await fs.writeFile(temp, JSON.stringify(serialized))
    await fs.rename(temp, masonryImageCacheFile)
  } catch {}
}

async function imageSizeFromFile(filePath: string): Promise<{ width?: number; height?: number }> {
  const resolved = path.resolve(filePath)
  const cache = await ensureMasonryImageCache()
  try {
    const stat = await fs.stat(resolved)
    if (stat.size <= 0) {
      return { width: undefined, height: undefined }
    }
    const cached = cache.get(resolved)
    if (cached && cached.mtimeMs === stat.mtimeMs && cached.size === stat.size) {
      return { width: cached.width, height: cached.height }
    }

    let handle: Awaited<ReturnType<typeof fs.open>> | undefined
    try {
      handle = await fs.open(resolved, 'r')
      const firstSize = Math.min(stat.size, HeaderInputSize)
      const firstInput = new Uint8Array(firstSize)
      await handle.read(firstInput, 0, firstSize, 0)
      let dimensions: { width?: number; height?: number } | undefined
      try {
        dimensions = imageSize(firstInput)
      } catch {}
      if ((!dimensions?.width || !dimensions?.height) && stat.size > firstSize) {
        const fullSize = Math.min(stat.size, MaxInputSize)
        const fullInput = new Uint8Array(fullSize)
        fullInput.set(firstInput, 0)
        await handle.read(fullInput, firstSize, fullSize - firstSize, firstSize)
        dimensions = imageSize(fullInput)
      }
      const width = dimensions?.width || 800
      const height = dimensions?.height || 600
      cache.set(resolved, { mtimeMs: stat.mtimeMs, size: stat.size, width, height })
      masonryCacheDirty = true
      return { width, height }
    } finally {
      await handle?.close()
    }
  } catch {
    return { width: 800, height: 600 }
  }
}

export interface MasonryImage {
  src: string
  alt: string
  width: number
  height: number
}

async function deleteMasonryImages(ctx: BuildCtx, slug: FullSlug): Promise<void> {
  await removeWritten(ctx, `${slug}.images`, '.json')
}

export const Masonry: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const filteredHeader = sharedPageComponents.header.filter(component => {
    const name = component.displayName || component.name || ''
    return name !== 'Breadcrumbs' && name !== 'StackedNotes'
  })

  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    pageBody: MasonryPage(),
    header: filteredHeader,
    beforeBody: [],
    sidebar: [],
    afterBody: [],
    footer: FooterConstructor({ layout: 'masonry' }),
  }

  const {
    head: Head,
    header: Header,
    beforeBody: BeforeBody,
    pageBody,
    afterBody,
    sidebar,
    footer: Footer,
  } = opts

  return {
    name: 'Masonry',
    getQuartzComponents() {
      return [Head, ...Header, ...BeforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = contentDataFor(content)

      // find all slugs that changed or were added
      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (changeEvent.type !== 'add' && changeEvent.type !== 'change') continue
        const slug = changeEvent.file.data.slug! as FullSlug
        if (changeEvent.file.data.frontmatter?.layout === 'masonry') {
          changedSlugs.add(slug)
        } else if (changeEvent.previousFile?.data.frontmatter?.layout === 'masonry') {
          await deleteMasonryImages(ctx, slug)
        }
      }

      // Asset events have no parsed file. Directory galleries still need a new catalogue.
      const imagesChanged = changeEvents.some(event =>
        imageExtensions.has(path.extname(event.path).toLowerCase()),
      )
      const targets = content.filter(
        ([, file]) =>
          (imagesChanged || changedSlugs.has(file.data.slug!)) &&
          file.data.frontmatter?.layout === 'masonry',
      )
      yield* await mapConcurrent(targets, defaultIoConcurrency, ([tree, file]) =>
        processMasonry(ctx, tree, file, allFiles, opts, resources),
      )
      await flushMasonryImageCache()
    },
    async *emit(ctx, content, resources) {
      const allFiles = contentDataFor(content)
      const targets = content.filter(([, file]) => file.data.frontmatter?.layout === 'masonry')
      yield* await mapConcurrent(targets, defaultIoConcurrency, ([tree, file]) =>
        processMasonry(ctx, tree, file, allFiles, opts, resources),
      )
      await flushMasonryImageCache()
    },
  }
}

async function extractImagesFromTree(tree: Root, contentRoot: string): Promise<MasonryImage[]> {
  const imageNodes: Array<{
    src: string
    alt: string
    width?: number
    height?: number
    needsSize: boolean
    sourcePath?: string
  }> = []

  const dimension = (value: unknown): number | undefined => {
    const parsed =
      typeof value === 'number'
        ? value
        : typeof value === 'string'
          ? parseInt(value, 10)
          : undefined
    return parsed !== undefined && Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
  }
  visit(tree, 'element', node => {
    if (node.tagName !== 'img' || typeof node.properties.src !== 'string') return
    const imgPath = node.properties.src
    const cleanPath = imgPath.startsWith('/') ? imgPath.slice(1) : imgPath
    const width = dimension(node.properties.width)
    const height = dimension(node.properties.height)
    imageNodes.push({
      src: imgPath,
      alt: typeof node.properties.alt === 'string' ? node.properties.alt : '',
      width,
      height,
      needsSize: width === undefined || height === undefined,
      sourcePath: path.join(contentRoot, cleanPath),
    })
  })

  return mapConcurrent(imageNodes, defaultIoConcurrency, async node => {
    if (node.needsSize && node.sourcePath) {
      const dimensions = await imageSizeFromFile(node.sourcePath)
      return {
        src: node.src,
        alt: node.alt,
        width: dimensions.width || 800,
        height: dimensions.height || 600,
      }
    }

    return { src: node.src, alt: node.alt, width: node.width || 800, height: node.height || 600 }
  })
}

function deduplicateImages(images: MasonryImage[]): MasonryImage[] {
  const seen = new Set<string>()
  const deduplicated: MasonryImage[] = []

  for (const img of images) {
    if (!seen.has(img.src)) {
      seen.add(img.src)
      deduplicated.push(img)
    }
  }

  return deduplicated
}

async function extractImagesFromDirectory(
  dirPath: string,
  contentRoot: string,
  files: FilePath[],
  origin: string,
): Promise<MasonryImage[]> {
  const raw = dirPath.replaceAll('\\', '/')
  const directory = normalizeReferencePath(
    raw.startsWith('./') || raw.startsWith('../') ? `${path.posix.dirname(origin)}/${raw}` : raw,
  )
  if (directory === undefined) return []
  const prefix = directory ? `${directory}/` : ''
  const root = contentRoot.replaceAll('\\', '/').replace(/\/$/, '')
  const imageFiles = files
    .map(file => {
      const source = file.replaceAll('\\', '/')
      return source.startsWith(`${root}/`) ? source.slice(root.length + 1) : source
    })
    .filter(
      (file): file is FilePath =>
        isFilePath(file) &&
        file.startsWith(prefix) &&
        imageExtensions.has(path.extname(file).toLowerCase()),
    )

  const images = await mapConcurrent(imageFiles, defaultIoConcurrency, async relativePath => {
    const ext = path.extname(relativePath)
    const slugifiedPath = slugifyFilePath(relativePath, true)
    const filePath = path.join(contentRoot, relativePath)

    const dimensions = await imageSizeFromFile(filePath)

    return {
      src: `/${slugifiedPath}${ext}`,
      alt: path.basename(relativePath, ext),
      width: dimensions.width || 800,
      height: dimensions.height || 600,
    }
  })

  return images
}

async function processMasonry(
  ctx: BuildCtx,
  tree: Root,
  file: VFile,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const { cfg } = ctx
  const slug = file.data.slug!

  // extract images from the current page
  const currentPageImages = await extractImagesFromTree(tree, ctx.argv.directory)

  // extract images from frontmatter masonry references
  const masonryRefs = file.data.frontmatter?.masonry as string[] | undefined
  let allFilesBySlug: Map<string, QuartzPluginData> | undefined

  const referencedImages =
    masonryRefs && Array.isArray(masonryRefs)
      ? (
          await Promise.all(
            masonryRefs.map(async ref => {
              const parsed = parseLiteralReference(ref)
              if (!parsed) return []

              const targetPath = referenceFields(parsed).target

              // try directory extraction first
              const dirImages = await extractImagesFromDirectory(
                targetPath,
                ctx.argv.directory,
                ctx.allFiles,
                referenceSourcePath(file.data),
              )
              if (dirImages.length > 0) {
                return dirImages
              }

              // fall back to file reference
              const resolved = resolveLiteralReference(parsed, slug, {
                catalog: referenceCatalogFor(ctx),
                origin: referenceSourcePath(file.data),
              })
              if (!resolved) return []

              allFilesBySlug ??= new Map(
                allFiles.filter(f => f.slug).map(f => [f.slug!, f] as const),
              )
              const referencedFile = allFilesBySlug.get(resolved.slug)
              if (!referencedFile?.htmlAst) return []

              return extractImagesFromTree(referencedFile.htmlAst, ctx.argv.directory)
            }),
          )
        ).flat()
      : []

  // combine and deduplicate images
  const allImages = deduplicateImages([...currentPageImages, ...referencedImages])

  // write images JSON file
  const imagesJsonSlug = `${slug}.images` as FullSlug
  await write({ ctx, content: JSON.stringify(allImages), slug: imagesJsonSlug, ext: '.json' })

  const fileData: QuartzPluginData = {
    ...file.data,
    masonryImages: allImages,
    masonryJsonPath: `/${slug}.images.json`,
  }

  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg: cfg.configuration,
    children: [],
    tree,
    allFiles,
  }

  const html = renderPage(ctx, slug, componentData, opts, externalResources, false)

  // write HTML page to output
  return write({ ctx, content: html, slug, ext: '.html' })
}

declare module 'vfile' {
  interface DataMap {
    masonryImages: MasonryImage[]
    masonryJsonPath: string
  }
}
