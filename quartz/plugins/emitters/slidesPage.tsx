import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { pathToRoot, joinSegments, FullSlug } from "../../util/path"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { write } from "./helpers"
import { BuildCtx } from "../../util/ctx"
import { Node } from "unist"
import { StaticResources } from "../../util/resources"
import { QuartzPluginData } from "../vfile"
import { SlidesContent } from "../../components"

const emitterName = "SlidesPage"

async function processSlides(
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const baseSlug = fileData.slug!
  const slidesSlug = joinSegments(baseSlug, "slides") as FullSlug
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slidesSlug), resources)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }

  const content = renderPage(ctx, slidesSlug, componentData, opts, externalResources, false, false)
  return write({ ctx, content, slug: slidesSlug, ext: ".html" })
}

export const SlidesPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  // slim page layout for slides
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: SlidesContent(),
    ...userOpts,
    sidebar: [],
  }

  const { head: Head, footer: Footer } = sharedPageComponents
  const Header = HeaderConstructor()

  return {
    name: emitterName,
    getQuartzComponents() {
      return [Head, Header, opts.pageBody, Footer]
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map((c) => c[1].data)

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        // skip tag pages and everything that isn’t a primary content page
        if (slug.endsWith("/index") || slug.startsWith("tags/")) continue
        // Only emit slides if explicitly enabled via frontmatter
        if (!file.data.frontmatter?.slides) continue
        yield processSlides(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map((c) => c[1].data)

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
        if (slug.endsWith("/index") || slug.startsWith("tags/")) continue
        if (!file.data.frontmatter?.slides) continue
        yield processSlides(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
  }
}
