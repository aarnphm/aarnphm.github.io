import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import { pageResources, renderPage } from "../../components/renderPage"
import { FullPageLayout } from "../../cfg"
import { pathToRoot, joinSegments, FullSlug } from "../../util/path"
import { sharedPageComponents, defaultContentPageLayout } from "../../../quartz.layout"
import ArenaIndex from "../../components/pages/ArenaIndex"
import ChannelContent from "../../components/pages/ChannelContent"
import { write } from "./helpers"
import { BuildCtx } from "../../util/ctx"
import { Node } from "unist"
import { StaticResources } from "../../util/resources"
import { QuartzPluginData, defaultProcessedContent } from "../vfile"
import { ArenaChannel } from "../transformers/arena"
import { clone } from "../../util/clone"

async function processArenaIndex(
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = "arena" as FullSlug
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const indexFileData = clone(fileData) as QuartzPluginData
  indexFileData.slug = slug
  indexFileData.arenaChannel = undefined
  indexFileData.frontmatter = {
    ...indexFileData.frontmatter,
    title: indexFileData.frontmatter?.title ?? fileData.frontmatter?.title ?? "are.na",
    pageLayout: indexFileData.frontmatter?.pageLayout ?? "default",
  }
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: indexFileData,
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

async function processChannel(
  ctx: BuildCtx,
  channel: ArenaChannel,
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const arenaBase = "arena" as FullSlug
  const channelSlug = joinSegments(arenaBase, channel.slug) as FullSlug
  const cfg = ctx.cfg.configuration

  const [tree] = defaultProcessedContent({
    slug: channelSlug,
    arenaChannel: channel,
    frontmatter: {
      ...baseFileData.frontmatter,
      title: channel.name,
      pageLayout: "default",
    },
  })

  const externalResources = pageResources(pathToRoot(channelSlug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: {
      ...baseFileData,
      slug: channelSlug,
      arenaChannel: channel,
      frontmatter: {
        ...baseFileData.frontmatter,
        title: channel.name,
        pageLayout: "default",
      },
    },
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }

  const content = renderPage(ctx, channelSlug, componentData, opts, externalResources, false)
  return write({
    ctx,
    content,
    slug: channelSlug,
    ext: ".html",
  })
}

export const ArenaPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = (userOpts) => {
  const filteredHeader = sharedPageComponents.header.filter((component) => {
    const name = component.displayName || component.name || ""
    return name !== "Breadcrumbs" && name !== "StackedNotes"
  })

  const indexOpts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    header: filteredHeader,
    afterBody: [],
    sidebar: [],
    pageBody: ArenaIndex(),
  }

  const channelOpts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    header: filteredHeader,
    afterBody: [],
    sidebar: [],
    pageBody: ChannelContent(),
  }

  const { head: Head, footer: Footer } = sharedPageComponents
  const Header = HeaderConstructor()

  return {
    name: "ArenaPage",
    getQuartzComponents() {
      return [Head, Header, indexOpts.pageBody, channelOpts.pageBody, Footer]
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map((c) => c[1].data)

      for (const [tree, file] of content) {
        const slug = file.data.slug!

        if (slug !== "are.na") continue
        if (!file.data.arenaData) continue

        yield processArenaIndex(ctx, tree, file.data, allFiles, indexOpts, resources)

        for (const channel of file.data.arenaData.channels) {
          yield processChannel(ctx, channel, file.data, allFiles, channelOpts, resources)
        }
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
        if (slug !== "are.na") continue
        if (!file.data.arenaData) continue

        yield processArenaIndex(ctx, tree, file.data, allFiles, indexOpts, resources)

        for (const channel of file.data.arenaData.channels) {
          yield processChannel(ctx, channel, file.data, allFiles, channelOpts, resources)
        }
      }
    },
  }
}
