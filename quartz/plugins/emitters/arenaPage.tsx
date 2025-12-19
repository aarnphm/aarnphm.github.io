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
import {
  ArenaChannel,
  ArenaBlock,
  ArenaBlockSearchable,
  ArenaChannelSearchable,
  ArenaSearchIndex,
} from "../transformers/arena"
import { clone } from "../../util/clone"
import { toHtml } from "hast-util-to-html"
import { ElementContent } from "hast"

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

/**
 * Convert ArenaBlock to searchable JSON format.
 * Serializes ElementContent nodes to HTML strings for JSON compatibility.
 */
function serializeBlock(
  block: ArenaBlock,
  channelSlug: string,
  channelName: string,
  hasModalInDom: boolean,
): ArenaBlockSearchable {
  const searchable: ArenaBlockSearchable = {
    id: block.id,
    channelSlug,
    channelName,
    content: block.content,
    highlighted: block.highlighted ?? false,
    pinned: block.pinned ?? false,
    later: block.later ?? false,
    hasModalInDom,
  }

  if (block.title) searchable.title = block.title
  if (block.url) searchable.url = block.url
  if (block.embedDisabled) searchable.embedDisabled = block.embedDisabled
  if (block.embedHtml) searchable.embedHtml = block.embedHtml
  if (block.metadata) searchable.metadata = block.metadata
  if (block.coordinates) searchable.coordinates = block.coordinates
  if (block.internalSlug) searchable.internalSlug = block.internalSlug
  if (block.internalHref) searchable.internalHref = block.internalHref
  if (block.internalHash) searchable.internalHash = block.internalHash
  if (block.internalTitle) searchable.internalTitle = block.internalTitle
  if (block.tags) searchable.tags = block.tags

  // Serialize ElementContent to HTML strings
  if (block.titleHtmlNode) {
    try {
      searchable.titleHtml = toHtml(block.titleHtmlNode as ElementContent)
    } catch {
      // Fallback to title if serialization fails
    }
  }

  if (block.htmlNode) {
    try {
      searchable.blockHtml = toHtml(block.htmlNode as ElementContent)
    } catch {
      // Fallback to content if serialization fails
    }
  }

  // Recursively serialize sub-items
  if (block.subItems && block.subItems.length > 0) {
    searchable.subItems = block.subItems.map((subBlock) =>
      serializeBlock(subBlock, channelSlug, channelName, false),
    )
  }

  return searchable
}

/**
 * Build complete search index from arena data.
 * Tracks which blocks have prerendered modals in DOM (first 5 per channel).
 */
function buildSearchIndex(channels: ArenaChannel[]): ArenaSearchIndex {
  const PREVIEW_BLOCK_LIMIT = 5

  const blocks: ArenaBlockSearchable[] = []
  const channelMetadata: ArenaChannelSearchable[] = []

  for (const channel of channels) {
    channelMetadata.push({
      id: channel.id,
      name: channel.name,
      slug: channel.slug,
      blockCount: channel.blocks.length,
    })

    channel.blocks.forEach((block, index) => {
      // First 5 blocks per channel have prerendered modals in index page
      const hasModalInDom = index < PREVIEW_BLOCK_LIMIT
      blocks.push(serializeBlock(block, channel.slug, channel.name, hasModalInDom))
    })
  }

  return {
    version: "1.0.0",
    blocks,
    channels: channelMetadata,
  }
}

async function emitSearchIndex(ctx: BuildCtx, searchIndex: ArenaSearchIndex) {
  const slug = "static/arena-search" as FullSlug
  const content = JSON.stringify(searchIndex)

  return write({
    ctx,
    content,
    slug,
    ext: ".json",
  })
}

async function processChannelJson(ctx: BuildCtx, channel: ArenaChannel) {
  const slug = joinSegments("arena", channel.slug, "json") as FullSlug
  const baseUrl = ctx.cfg.configuration.baseUrl ?? "aarnphm.xyz"

  const output: Record<string, Record<string, unknown>> = {}

  for (const block of channel.blocks) {
    const url =
      block.url ?? (block.internalSlug ? `https://${baseUrl}/${block.internalSlug}` : null)
    if (!url) continue

    const entry: Record<string, unknown> = {}

    if (block.title) entry.title = block.title
    if (block.tags && block.tags.length > 0) entry.tags = block.tags
    if (block.metadata?.date) entry.date = block.metadata.date
    if (block.metadata?.accessed) entry.accessed = block.metadata.accessed
    if (block.pinned) entry.pinned = true
    if (block.later) entry.later = true
    if (block.highlighted) entry.highlighted = true

    if (block.metadata) {
      const skip = new Set(["date", "accessed", "tags", "pinned", "later"])
      for (const [k, v] of Object.entries(block.metadata)) {
        if (!skip.has(k) && v) entry[k] = v
      }
    }

    output[url] = entry
  }

  return write({
    ctx,
    content: JSON.stringify(output, null, 2),
    slug,
    ext: "",
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

          const jsonEnabled = channel.metadata?.json === "true" || channel.metadata?.json === true
          if (jsonEnabled) {
            yield processChannelJson(ctx, channel)
          }
        }

        // Build and emit search index JSON after all channels are processed
        const searchIndex = buildSearchIndex(file.data.arenaData.channels)
        yield emitSearchIndex(ctx, searchIndex)
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

          const jsonEnabled = channel.metadata?.json === "true" || channel.metadata?.json === true
          if (jsonEnabled) {
            yield processChannelJson(ctx, channel)
          }
        }

        // Build and emit search index JSON after all channels are processed
        const searchIndex = buildSearchIndex(file.data.arenaData.channels)
        yield emitSearchIndex(ctx, searchIndex)
      }
    },
    externalResources: () => ({
      additionalHead: [
        <script src={"https://api.mapbox.com/mapbox-gl-js/v3.15.0/mapbox-gl.js"} defer />,
        <link
          href={"https://api.mapbox.com/mapbox-gl-js/v3.15.0/mapbox-gl.css"}
          rel="stylesheet"
        />,
        <script
          src={"https://cdnjs.cloudflare.com/ajax/libs/pdf.js/5.4.54/pdf.min.mjs"}
          defer
          type="module"
        />,
      ],
    }),
  }
}
