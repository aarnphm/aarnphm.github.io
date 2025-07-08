import { BuildCtx } from "../../util/ctx"
import { QuartzEmitterPlugin } from "../types"
import { QuartzPluginData } from "../vfile"
import { ProcessedContent } from "../vfile"
import { Root, Element } from "hast"
import { visit } from "unist-util-visit"
import { pathToRoot, FullSlug } from "../../util/path"
import { write } from "./helpers"
import { renderPage, pageResources as buildPageResources } from "../../components/renderPage"
import * as Component from "../../components"
import { sharedPageComponents, defaultListPageLayout } from "../../../quartz.layout"
import { defaultProcessedContent } from "../vfile"
import { clone } from "../../util/clone"

/** Helper to slugify heading text */
function toSlug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
}

interface CategoryInfo {
  heading: string
  slug: string
  items: Element[]
}

function extractCategories(tree: Root): CategoryInfo[] {
  const categories: CategoryInfo[] = []
  let current: CategoryInfo | null = null

  for (const node of tree.children) {
    if (node.type === "element" && /h2/i.test(node.tagName)) {
      // finalize previous
      if (current) {
        categories.push(current)
      }
      const headingText = (node.children[0] && (node.children[0] as any).value) || ""
      current = { heading: headingText, slug: toSlug(headingText), items: [] }
    } else if (current) {
      // collect until next heading
      current.items.push(node as Element)
    }
  }
  if (current) categories.push(current)
  return categories
}

function createHomeTree(categories: CategoryInfo[], h: typeof import("hastscript").h): Root {
  const children: Element[] = []
  for (const cat of categories) {
    const href = `./${cat.slug}`
    children.push(
      h("div.arena-category", [
        h(
          "a",
          { href, class: "internal arena-link", "data-no-popover": "true" },
          cat.heading,
        ),
      ]),
    )
  }
  return { type: "root", children }
}

function createCategoryTree(cat: CategoryInfo, h: typeof import("hastscript").h): Root {
  return { type: "root", children: cat.items }
}

export const ArenaPage: QuartzEmitterPlugin = () => {
  return {
    name: "ArenaPage",

    getQuartzComponents() {
      // reuse default components; rendering handled in renderPage
      return []
    },

    async *emit(ctx: BuildCtx, content: ProcessedContent[], resources) {
      // Find are.na page
      const arenaContent = content.find(([_, vfile]) => (vfile.data as QuartzPluginData).slug === "are.na")
      if (!arenaContent) return

      const [tree, vfile] = arenaContent
      const categories = extractCategories(tree as Root)
      const h = (await import("hastscript")).h

      // Main home page
      {
        const homeTree = createHomeTree(categories, h)
        const pc = clone(arenaContent) as ProcessedContent
        pc[0] = homeTree
        const qpd = (pc[1].data as QuartzPluginData)
        qpd.frontmatter = qpd.frontmatter ?? {}
        qpd.frontmatter.pageLayout = "default"
        // create static resources from plugins
        const pr = buildPageResources(pathToRoot("are.na" as FullSlug), resources)
        const allFiles = content.map(([, vf]) => vf.data as QuartzPluginData)
        const layoutHome = {
          ...sharedPageComponents,
          ...defaultListPageLayout,
          pageBody: Component.Content(),
        } as any
        const html = renderPage(
          ctx,
          "are.na" as FullSlug,
          {
            ctx,
            externalResources: pr,
            fileData: qpd,
            cfg: ctx.cfg,
            children: [],
            tree: homeTree,
            allFiles,
          } as any,
          layoutHome,
          pr,
          false,
          false,
        )
        yield write({ ctx, slug: "are.na" as FullSlug, ext: ".html", content: html })
      }

      // Category pages
      for (const cat of categories) {
        const catTree = createCategoryTree(cat, h)
        const slug = `are.na/${cat.slug}`
        const pcClone = defaultProcessedContent({ slug } as any)
        pcClone[0] = catTree
        const qpd = pcClone[1].data as QuartzPluginData
        qpd.slug = slug as any
        qpd.frontmatter = { title: cat.heading, pageLayout: "default" }

        const prCat = buildPageResources(pathToRoot(slug as FullSlug), resources)
        const allFiles2 = content.map(([, vf]) => vf.data as QuartzPluginData)
        const layoutCat = {
          ...sharedPageComponents,
          ...defaultListPageLayout,
          pageBody: Component.Content(),
        } as any
        const html = renderPage(
          ctx,
          slug as FullSlug,
          {
            ctx,
            externalResources: prCat,
            fileData: qpd,
            cfg: ctx.cfg,
            children: [],
            tree: catTree,
            allFiles: allFiles2,
          } as any,
          layoutCat,
          prCat,
          false,
          false,
        )
        yield write({ ctx, slug: slug as FullSlug, ext: ".html", content: html })
      }
    },
  }
}

export default ArenaPage