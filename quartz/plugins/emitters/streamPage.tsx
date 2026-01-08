import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { pathToRoot } from "../../util/path"
import { sharedPageComponents, defaultContentPageLayout } from "../../../quartz.layout"
import { write } from "./helpers"
import { BuildCtx } from "../../util/ctx"
import { Root } from "hast"
import { StaticResources } from "../../util/resources"
import { QuartzPluginData } from "../vfile"
import StreamPageComponent from "../../components/pages/StreamPage"
import StreamSearchComponent from "../../components/StreamSearch"

// FIXME: for links transformation on date subpages, make sure to not include original slug here.
async function processStreamPage(
  ctx: BuildCtx,
  tree: Root,
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

  const content = renderPage(ctx, slug, componentData, opts, externalResources, false)
  return write({
    ctx,
    content,
    slug,
    ext: ".html",
  })
}

export const StreamPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  const filteredHeader = sharedPageComponents.header.filter((component) => {
    const name = component.displayName || component.name || ""
    return name !== "Breadcrumbs" && name !== "StackedNotes"
  })
  const filteredBefore = defaultContentPageLayout.beforeBody.filter(
    (c) => c.displayName !== "Byline" || c.name !== "Byline",
  )

  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    header: filteredHeader,
    beforeBody: filteredBefore,
    afterBody: [],
    pageBody: StreamPageComponent(),
  }

  const { head: Head, header, beforeBody, pageBody, footer: Footer } = opts
  const Header = HeaderConstructor()
  const StreamSearch = StreamSearchComponent()

  return {
    name: "StreamPage",
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, Footer, StreamSearch]
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map((c) => c[1].data)

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        const slug = data.slug!

        if (slug !== "stream" || !data.streamData) continue

        yield processStreamPage(ctx, tree, data, allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map((c) => c[1].data)
      const changedSlugs = new Set<string>()

      for (const changeEvent of changeEvents) {
        if (changeEvent.file) {
          if (changeEvent.type === "add" || changeEvent.type === "change") {
            changedSlugs.add(changeEvent.file.data.slug!)
          }
          continue
        }

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

      if (!changedSlugs.has("stream")) return

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        const slug = data.slug!
        if (slug !== "stream" || !data.streamData) continue

        yield processStreamPage(ctx, tree, data, allFiles, opts, resources)
      }
    },
  }
}
