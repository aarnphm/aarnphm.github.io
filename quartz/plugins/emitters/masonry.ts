import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { pathToRoot } from "../../util/path"
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
  // extract images from the tree
  const images: MasonryImage[] = []
  visit(tree as Root, "element", (node: any) => {
    if (node.tagName === "img" && node.properties?.src) {
      images.push({
        src: node.properties.src as string,
        alt: (node.properties.alt as string) || "",
      })
    }
  })

  const fileData: QuartzPluginData = { ...file.data, masonryImages: images }

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
