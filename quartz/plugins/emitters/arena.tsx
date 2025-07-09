import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import { pageResources, renderPage } from "../../components/renderPage"
import { ProcessedContent, QuartzPluginData } from "../vfile"
import { FullPageLayout } from "../../cfg"
import { FullSlug, pathToRoot } from "../../util/path"
import { defaultListPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { Content } from "../../components"
import { write } from "./helpers"
import { BuildCtx } from "../../util/ctx"
import type { Root, Element, ElementContent } from "hast"
import { visit } from "unist-util-visit"
import { clone } from "../../util/clone"
import { h } from "hastscript"

interface ArenaPageOptions extends FullPageLayout {
  enableGrid?: boolean
  categoryClass?: string
}

const defaultOptions: ArenaPageOptions = {
  enableGrid: true,
  categoryClass: "arena-category",
}

interface CategoryInfo {
  heading: string
  slug: string
  items: Element[]
}

function toSlug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
}

function extractCategories(tree: Root): CategoryInfo[] {
  const categories: CategoryInfo[] = []
  let current: CategoryInfo | null = null

  for (const node of tree.children) {
    if (node.type === "element" && /h2/i.test(node.tagName)) {
      if (current) {
        categories.push(current)
      }
      const headingText = (node.children[0] && (node.children[0] as any).value) || ""
      current = { heading: headingText, slug: toSlug(headingText), items: [] }
    } else if (current) {
      current.items.push(node as Element)
    }
  }
  if (current) categories.push(current)
  return categories
}

function createHomeTree(categories: CategoryInfo[], opts: ArenaPageOptions,): Root {
  const children: Element[] = [h("div", { class: "arena-grid" }, [])]
  const grid = children[0] as Element
  for (const cat of categories) {
    const href = `./${cat.slug}`
    grid.children.push(
      h(
        `a.${opts.categoryClass}`,
        { href, class: "internal arena-link", "data-no-popover": "true" },
        [h("h3", cat.heading)],
      ),
    )
  }
  return { type: "root", children }
}

function createCategoryTree(cat: CategoryInfo): Root {
  const cards: Element[] = []
  visit({ type: "root", children: cat.items } as Root, { tagName: "li" }, (node: any) => {
    const textContent = (node.children ?? []).map((n: any) => n.value ?? "").join("")
    const linePattern = /^-?\s*(https?:[^\s]+)(?:\s*--\s*(.*))?$/i
    const match = textContent.match(linePattern)
    if (!match) {
      cards.push(
        h("div.arena-card", { style: "background: var(--lightgray);" }, [
          h("div.arena-title", "Invalid entry"),
          h("p.arena-note", "Could not parse: " + textContent),
        ]),
      )
      return
    }
    try {
      const url = match[1]
      const note = match[2] ?? ""
      let subNote = ""
      if (node.children) {
        const sub = node.children.find((c: any) => c.tagName === "ul")
        if (sub && Array.isArray(sub.children)) {
          const subTexts: string[] = []
          sub.children.forEach((li: any) => {
            const t = (li.children ?? []).map((n: any) => n.value ?? "").join("")
            if (t) subTexts.push(t)
          })
          subNote = subTexts.join("\n")
        }
      }
      cards.push(
        h(
          "a.arena-card",
          {
            href: url,
            target: "_blank",
            rel: "noopener noreferrer",
            "data-subnote": subNote || undefined,
          },
          [h("div.arena-title", url), h("p.arena-note", note)],
        ),
      )
    } catch (err) {
      console.error("Error parsing list item: ", textContent, err)
    }
  })

  const gridChildren = cards.length > 0 ? cards : [h("p", "No links available in this section.")]
  const grid = h("div.arena-grid", gridChildren as ElementContent[])
  return { type: "root", children: [grid] }
}

async function processArenaPage(
  ctx: BuildCtx,
  tree: Root,
  file: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: any,
  categories: CategoryInfo[],
  slug: FullSlug,
  isCategory: boolean = false,
  categoryInfo?: CategoryInfo,
) {
  const h = (await import("hastscript")).h

  let processedTree: Root
  if (isCategory && categoryInfo) {
    processedTree = createCategoryTree(categoryInfo, h)
  } else {
    processedTree = createHomeTree(categories, h, opts as ArenaPageOptions)
  }

  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources)

  externalResources.css.push({
    content: `@import "../../components/styles/arena.scss";`,
    inline: true,
  })

  const componentData: QuartzComponentProps = {
    ctx,
    fileData: {
      ...file,
      slug,
      frontmatter: {
        ...file.frontmatter,
        title:
          isCategory && categoryInfo ? categoryInfo.heading : file.frontmatter?.title || "Are.na",
        pageLayout: "default",
      },
    },
    externalResources,
    cfg,
    children: [],
    tree: processedTree,
    allFiles,
  }

  const content = renderPage(ctx, slug, componentData, opts, externalResources, false, false)

  return write({
    ctx,
    content,
    slug,
    ext: ".html",
  })
}

export const ArenaPage: QuartzEmitterPlugin<Partial<ArenaPageOptions>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultListPageLayout,
    pageBody: Content(),
    ...defaultOptions,
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()

  return {
    name: "ArenaPage",
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map((c) => c[1].data)

      const arenaContent = content.find(([_, vfile]) => vfile.data.slug === "are.na")
      if (!arenaContent) return

      const [tree, file] = arenaContent
      const categories = extractCategories(tree as Root)

      yield await processArenaPage(
        ctx,
        tree as Root,
        file.data,
        allFiles,
        opts,
        resources,
        categories,
        "are.na" as FullSlug,
      )

      for (const cat of categories) {
        const catSlug = `are.na/${cat.slug}` as FullSlug
        yield await processArenaPage(
          ctx,
          tree as Root,
          file.data,
          allFiles,
          opts,
          resources,
          categories,
          catSlug,
          true,
          cat,
        )
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

      if (changedSlugs.has("are.na")) {
        const arenaContent = content.find(([_, vfile]) => vfile.data.slug === "are.na")
        if (!arenaContent) return

        const [tree, file] = arenaContent
        const categories = extractCategories(tree as Root)

        yield await processArenaPage(
          ctx,
          tree as Root,
          file.data,
          allFiles,
          opts,
          resources,
          categories,
          "are.na" as FullSlug,
        )

        for (const cat of categories) {
          const catSlug = `are.na/${cat.slug}` as FullSlug
          yield await processArenaPage(
            ctx,
            tree as Root,
            file.data,
            allFiles,
            opts,
            resources,
            categories,
            catSlug,
            true,
            cat,
          )
        }
      }
    },
  }
}

export default ArenaPage
