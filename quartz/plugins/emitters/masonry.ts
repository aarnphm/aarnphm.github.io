import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { pathToRoot, slugifyFilePath, FilePath } from "../../util/path"
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

export interface MasonryImage {
  src: string
  alt: string
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

function extractImagesFromTree(tree: Node): MasonryImage[] {
  const images: MasonryImage[] = []
  visit(tree as Root, "element", (node: any) => {
    if (node.tagName === "img" && node.properties?.src) {
      images.push({
        src: node.properties.src as string,
        alt: (node.properties.alt as string) || "",
      })
    }
  })
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

function extractImagesFromDirectory(dirPath: string, contentRoot: string): MasonryImage[] {
  const images: MasonryImage[] = []
  const fullPath = path.join(contentRoot, dirPath)
  const imageExtensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"]

  const files = fs.readdirSync(fullPath)

  for (const file of files) {
    const ext = path.extname(file).toLowerCase()
    if (imageExtensions.includes(ext)) {
      const relativePath = path.join(dirPath, file).replace(/\\/g, "/")
      const slugifiedPath = slugifyFilePath(relativePath as FilePath, true)
      images.push({
        src: `/${slugifiedPath}${ext}`,
        alt: path.basename(file, ext),
      })
    }
  }

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
  const currentPageImages = extractImagesFromTree(tree)

  // extract images from frontmatter masonry references
  const referencedImages: MasonryImage[] = []
  const masonryRefs = file.data.frontmatter?.masonry as string[] | undefined

  if (masonryRefs && Array.isArray(masonryRefs)) {
    for (const ref of masonryRefs) {
      const parsed = parseWikilink(ref)
      if (!parsed) continue

      const targetPath = parsed.target

      // try directory extraction first
      const dirImages = extractImagesFromDirectory(targetPath, ctx.argv.directory)
      if (dirImages.length > 0) {
        referencedImages.push(...dirImages)
        continue
      }

      // fall back to file reference
      const resolved = resolveWikilinkTarget(parsed, slug)
      if (!resolved) continue

      const referencedFile = allFiles.find((f) => f.slug === resolved.slug)
      if (!referencedFile?.tree) continue

      const refImages = extractImagesFromTree(referencedFile.tree)
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
