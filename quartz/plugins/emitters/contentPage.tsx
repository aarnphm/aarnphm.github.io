import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import {
  pageResources,
  renderPage,
  CuriusContent,
  CuriusFriends,
} from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { pathToRoot } from "../../util/path"
import { defaultContentPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { Content } from "../../components"
import { write } from "./helpers"
import { BuildCtx } from "../../util/ctx"
import { Node } from "unist"
import { StaticResources } from "../../util/resources"
import { QuartzPluginData } from "../vfile"

async function processContent(
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = fileData.slug!
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg,
    children: [],
    tree,
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

export const ContentPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: Content(),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()

  return {
    name: "ContentPage",
    getQuartzComponents() {
      return [
        Head,
        Header,
        CuriusFriends,
        CuriusContent,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...sidebar,
        Footer,
      ]
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map((c) => c[1].data)

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (slug.endsWith("/index") || slug.startsWith("tags/") || slug === "are.na") continue
        yield processContent(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map((c) => c[1].data)

      // find all slugs that changed or were added
      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        // If it's a markdown file change, add its own slug
        if (changeEvent.file) {
          if (changeEvent.type === "add" || changeEvent.type === "change") {
            changedSlugs.add(changeEvent.file.data.slug!)
          }
          continue
        }
        // Non-markdown file changed: re-emit any page that depends on it
        if (changeEvent.type === "add" || changeEvent.type === "change") {
          const changedPath = changeEvent.path
          for (const [_, vf] of content) {
            const deps = (vf.data.codeDependencies as string[] | undefined) ?? []
            if (deps.includes(changedPath)) {
              changedSlugs.add(vf.data.slug!)
            }
          }
        }
      }

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (!changedSlugs.has(slug)) continue
        if (slug.endsWith("/index") || slug.startsWith("tags/") || slug === "are.na") continue

        yield processContent(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
  }
}
