import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { pathToRoot, slugifyFilePath, FilePath, FullSlug } from "../../util/path"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { write } from "./helpers"
import { visit } from "unist-util-visit"
import { Root } from "hast"
import { Node } from "unist"
import { QuartzPluginData } from "../vfile"
import { MasonryPage, Footer as FooterConstructor } from "../../components/"
import { StaticResources } from "../../util/resources"
import { BuildCtx } from "../../util/ctx"
import { VFile } from "vfile"
import { parseWikilink, resolveWikilinkTarget } from "../../util/wikilinks"
import path from "path"
import fs from "fs"
import { imageSize } from "image-size"

const MaxInputSize = 512 * 1024

async function imageSizeFromFile(filePath: string): Promise<{ width?: number; height?: number }> {
  let handle: fs.promises.FileHandle | undefined
  try {
    handle = await fs.promises.open(path.resolve(filePath), "r")
    const { size } = await handle.stat()
    if (size <= 0) {
      return { width: undefined, height: undefined }
    }
    const inputSize = Math.min(size, MaxInputSize)
    const input = new Uint8Array(inputSize)
    await handle.read(input, 0, inputSize, 0)
    return imageSize(input)
  } catch {
    return { width: 800, height: 600 }
  } finally {
    await handle?.close()
  }
}

export interface MasonryImage {
  src: string
  alt: string
  width: number
  height: number
}

export const Masonry: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  const filteredHeader = sharedPageComponents.header.filter((component) => {
    const name = component.displayName || component.name || ""
    return name !== "Breadcrumbs" && name !== "StackedNotes"
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
    footer: FooterConstructor({ layout: "masonry" }),
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
    name: "Masonry",
    getQuartzComponents() {
      return [Head, ...Header, ...BeforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map((c) => c[1].data)

      // find all slugs that changed or were added
      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (changeEvent.type === "add" || changeEvent.type === "change") {
          changedSlugs.add(changeEvent.file.data.slug!)
        }
      }

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (!changedSlugs.has(slug)) continue
        if (file.data.frontmatter?.layout !== "masonry") continue

        yield processMasonry(ctx, tree, file, allFiles, opts, resources)
      }
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map((c) => c[1].data)

      for (const [tree, file] of content) {
        if (file.data.frontmatter?.layout !== "masonry") continue
        yield processMasonry(ctx, tree, file, allFiles, opts, resources)
      }
    },
  }
}

async function extractImagesFromTree(tree: Node, contentRoot: string): Promise<MasonryImage[]> {
  const images: MasonryImage[] = []
  const imageNodes: Array<{
    src: string
    alt: string
    width?: number
    height?: number
    needsSize: boolean
    sourcePath?: string
  }> = []

  visit(tree as Root, "element", (node: any) => {
    if (node.tagName === "img" && node.properties?.src) {
      const imgPath = node.properties.src as string
      const cleanPath = imgPath.startsWith("/") ? imgPath.slice(1) : imgPath
      const sourcePath = path.join(contentRoot, cleanPath)

      if (node.properties.width && node.properties.height) {
        imageNodes.push({
          src: imgPath,
          alt: (node.properties.alt as string) || "",
          width: parseInt(node.properties.width),
          height: parseInt(node.properties.height),
          needsSize: false,
        })
      } else if (fs.existsSync(sourcePath)) {
        imageNodes.push({
          src: imgPath,
          alt: (node.properties.alt as string) || "",
          needsSize: true,
          sourcePath,
        })
      } else {
        imageNodes.push({
          src: imgPath,
          alt: (node.properties.alt as string) || "",
          width: 800,
          height: 600,
          needsSize: false,
        })
      }
    }
  })

  await Promise.all(
    imageNodes.map(async (node) => {
      if (node.needsSize && node.sourcePath) {
        const dimensions = await imageSizeFromFile(node.sourcePath)
        images.push({
          src: node.src,
          alt: node.alt,
          width: dimensions.width || 800,
          height: dimensions.height || 600,
        })
      } else {
        images.push({
          src: node.src,
          alt: node.alt,
          width: node.width || 800,
          height: node.height || 600,
        })
      }
    }),
  )

  return images
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
): Promise<MasonryImage[]> {
  const fullPath = path.join(contentRoot, dirPath)
  const imageExtensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"]

  const files = fs.readdirSync(fullPath)
  const imageFiles = files.filter((file) => {
    const ext = path.extname(file).toLowerCase()
    return imageExtensions.includes(ext)
  })

  const images = await Promise.all(
    imageFiles.map(async (file) => {
      const ext = path.extname(file).toLowerCase()
      const relativePath = path.join(dirPath, file).replace(/\\/g, "/")
      const slugifiedPath = slugifyFilePath(relativePath as FilePath, true)
      const filePath = path.join(fullPath, file)

      const dimensions = await imageSizeFromFile(filePath)

      return {
        src: `/${slugifiedPath}${ext}`,
        alt: path.basename(file, ext),
        width: dimensions.width || 800,
        height: dimensions.height || 600,
      }
    }),
  )

  return images
}

async function processMasonry(
  ctx: BuildCtx,
  tree: Node,
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
  const referencedImages: MasonryImage[] = []
  const masonryRefs = file.data.frontmatter?.masonry as string[] | undefined

  if (masonryRefs && Array.isArray(masonryRefs)) {
    for (const ref of masonryRefs) {
      const parsed = parseWikilink(ref)
      if (!parsed) continue

      const targetPath = parsed.target

      // try directory extraction first
      const dirImages = await extractImagesFromDirectory(targetPath, ctx.argv.directory)
      if (dirImages.length > 0) {
        referencedImages.push(...dirImages)
        continue
      }

      // fall back to file reference
      const resolved = resolveWikilinkTarget(parsed, slug)
      if (!resolved) continue

      const referencedFile = allFiles.find((f) => f.slug === resolved.slug)
      if (!referencedFile?.tree) continue

      const refImages = await extractImagesFromTree(referencedFile.tree, ctx.argv.directory)
      referencedImages.push(...refImages)
    }
  }

  // combine and deduplicate images
  const allImages = deduplicateImages([...currentPageImages, ...referencedImages])

  // write images JSON file
  const imagesJsonSlug = `${slug}.images` as FullSlug
  await write({
    ctx,
    content: JSON.stringify(allImages),
    slug: imagesJsonSlug,
    ext: ".json",
  })

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
    tree: tree as Root,
    allFiles,
  }

  const html = renderPage(ctx, slug, componentData, opts, externalResources, false)

  // write HTML page to output
  return write({ ctx, content: html, slug, ext: ".html" })
}

declare module "vfile" {
  interface DataMap {
    masonryImages: MasonryImage[]
    masonryJsonPath: string
  }
}
