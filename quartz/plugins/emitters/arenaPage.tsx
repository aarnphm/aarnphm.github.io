import { ElementContent } from 'hast'
import { toHtml } from 'hast-util-to-html'
import fs from 'node:fs/promises'
import { Node } from 'unist'
import { sharedPageComponents, defaultContentPageLayout } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import HeaderConstructor from '../../components/Header'
import ArenaIndex from '../../components/pages/ArenaIndex'
import ChannelContent from '../../components/pages/ChannelContent'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { ChangeEvent, QuartzEmitterPlugin } from '../../types/plugin'
import {
  collectArenaEmitState,
  isArenaChannelJsonEnabled,
  planArenaPartialEmit,
  type ArenaEmitState,
} from '../../util/arena-page-partial'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { clone } from '../../util/clone'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { pathToRoot, joinSegments, FullSlug, FilePath } from '../../util/path'
import { StaticResources } from '../../util/resources'
import {
  ArenaChannel,
  ArenaBlock,
  ArenaBlockSearchable,
  ArenaChannelSearchable,
  ArenaSearchIndex,
} from '../transformers/arena'
import { QuartzPluginData, defaultProcessedContent } from '../vfile'
import { write } from './helpers'

async function processArenaIndex(
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = 'arena' as FullSlug
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const indexFileData = clone(fileData) as QuartzPluginData
  indexFileData.slug = slug
  indexFileData.arenaChannel = undefined
  indexFileData.frontmatter = {
    ...indexFileData.frontmatter,
    title: indexFileData.frontmatter?.title ?? fileData.frontmatter?.title ?? 'are.na',
    pageLayout: indexFileData.frontmatter?.pageLayout ?? 'default',
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
  return write({ ctx, content, slug, ext: '.html' })
}

async function processChannel(
  ctx: BuildCtx,
  channel: ArenaChannel,
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const arenaBase = 'arena' as FullSlug
  const channelSlug = joinSegments(arenaBase, channel.slug) as FullSlug
  const cfg = ctx.cfg.configuration

  const [tree] = defaultProcessedContent({
    slug: channelSlug,
    arenaChannel: channel,
    frontmatter: { ...baseFileData.frontmatter, title: channel.name, pageLayout: 'default' },
  })

  const externalResources = pageResources(pathToRoot(channelSlug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: {
      ...baseFileData,
      slug: channelSlug,
      arenaChannel: channel,
      frontmatter: { ...baseFileData.frontmatter, title: channel.name, pageLayout: 'default' },
    },
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }

  const content = renderPage(ctx, channelSlug, componentData, opts, externalResources, false)
  return write({ ctx, content, slug: channelSlug, ext: '.html' })
}

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
  if (block.embedMode) searchable.embedMode = block.embedMode
  if (block.embedHtml) searchable.embedHtml = block.embedHtml
  if (block.metadata) searchable.metadata = block.metadata
  if (block.coordinates) searchable.coordinates = block.coordinates
  if (block.internalSlug) searchable.internalSlug = block.internalSlug
  if (block.internalHref) searchable.internalHref = block.internalHref
  if (block.internalHash) searchable.internalHash = block.internalHash
  if (block.internalTitle) searchable.internalTitle = block.internalTitle
  if (block.tags) searchable.tags = block.tags

  if (block.titleHtmlNode) {
    try {
      searchable.titleHtml = toHtml(block.titleHtmlNode as ElementContent)
    } catch {}
  }

  if (block.htmlNode) {
    try {
      searchable.blockHtml = toHtml(block.htmlNode as ElementContent)
    } catch {}
  }

  if (block.subItems && block.subItems.length > 0) {
    searchable.subItems = block.subItems.map(subBlock =>
      serializeBlock(subBlock, channelSlug, channelName, false),
    )
  }

  return searchable
}

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
      const hasModalInDom = index < PREVIEW_BLOCK_LIMIT
      blocks.push(serializeBlock(block, channel.slug, channel.name, hasModalInDom))
    })
  }

  return { version: '1.0.0', blocks, channels: channelMetadata }
}

async function emitSearchIndex(ctx: BuildCtx, searchIndex: ArenaSearchIndex) {
  const slug = 'static/arena-search' as FullSlug
  const content = JSON.stringify(searchIndex)

  return write({ ctx, content, slug, ext: '.json' })
}

async function processChannelJson(ctx: BuildCtx, channel: ArenaChannel) {
  const slug = joinSegments('arena', channel.slug, 'json') as FullSlug
  const baseUrl = ctx.cfg.configuration.baseUrl ?? 'aarnphm.xyz'

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
      const skip = new Set(['date', 'accessed', 'tags', 'pinned', 'later'])
      for (const [k, v] of Object.entries(block.metadata)) {
        if (!skip.has(k) && v) entry[k] = v
      }
    }

    output[url] = entry
  }

  return write({ ctx, content: JSON.stringify(output, null, 2), slug, ext: '' })
}

async function processChannelOutputs(
  ctx: BuildCtx,
  channel: ArenaChannel,
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
): Promise<FilePath[]> {
  const files = [await processChannel(ctx, channel, baseFileData, allFiles, opts, resources)]
  if (isArenaChannelJsonEnabled(channel)) {
    files.push(await processChannelJson(ctx, channel))
  }
  return files
}

async function processChangedChannelOutputs(
  ctx: BuildCtx,
  channel: ArenaChannel,
  previous: { jsonEnabled: boolean } | undefined,
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
): Promise<FilePath[]> {
  const files = [await processChannel(ctx, channel, baseFileData, allFiles, opts, resources)]
  if (isArenaChannelJsonEnabled(channel)) {
    files.push(await processChannelJson(ctx, channel))
  } else if (previous?.jsonEnabled) {
    await fs.rm(joinSegments(ctx.argv.output, 'arena', channel.slug, 'json'), { force: true })
  }
  return files
}

async function removeChannelOutputs(
  ctx: BuildCtx,
  channelSlug: string,
  jsonEnabled: boolean,
): Promise<void> {
  await fs.rm(joinSegments(ctx.argv.output, 'arena', `${channelSlug}.html`), { force: true })
  if (jsonEnabled) {
    await fs.rm(joinSegments(ctx.argv.output, 'arena', channelSlug, 'json'), { force: true })
  }
}

function hasArenaPageChange(changeEvents: readonly ChangeEvent[]): boolean {
  for (const changeEvent of changeEvents) {
    const slug = changeEvent.file?.data.slug ?? changeEvent.previousFile?.data.slug
    if (slug === 'are.na') return true
  }
  return false
}

export const ArenaPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const filteredHeader = sharedPageComponents.header.filter(component => {
    const name = component.displayName || component.name || ''
    return name !== 'Breadcrumbs' && name !== 'StackedNotes'
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
  let arenaEmitState: ArenaEmitState | undefined

  return {
    name: 'ArenaPage',
    getQuartzComponents() {
      return [
        Head,
        Header,
        ...indexOpts.header,
        ...indexOpts.beforeBody,
        indexOpts.pageBody,
        ...indexOpts.afterBody,
        ...indexOpts.sidebar,
        ...channelOpts.header,
        ...channelOpts.beforeBody,
        channelOpts.pageBody,
        ...channelOpts.afterBody,
        ...channelOpts.sidebar,
        Footer,
      ]
    },
    async *emit(ctx, content, resources) {
      const allFiles = contentDataFor(content)

      for (const [tree, file] of content) {
        const slug = file.data.slug!

        if (slug !== 'are.na') continue
        if (!file.data.arenaData) continue

        const channels = file.data.arenaData.channels
        yield processArenaIndex(ctx, tree, file.data, allFiles, indexOpts, resources)

        const channelFiles = await mapConcurrent(channels, defaultIoConcurrency, channel =>
          processChannelOutputs(ctx, channel, file.data, allFiles, channelOpts, resources),
        )

        for (const files of channelFiles) {
          yield* files
        }

        const searchIndex = buildSearchIndex(channels)
        yield emitSearchIndex(ctx, searchIndex)
        arenaEmitState = collectArenaEmitState(channels)
      }
    },
    partialEmit(ctx, content, resources, changeEvents) {
      if (!hasArenaPageChange(changeEvents)) return null

      return (async function* () {
        const allFiles = contentDataFor(content)

        const changedSlugs = new Set<string>()
        for (const changeEvent of changeEvents) {
          if (!changeEvent.file) continue
          if (changeEvent.type === 'add' || changeEvent.type === 'change') {
            changedSlugs.add(changeEvent.file.data.slug!)
          }
        }

        for (const [tree, file] of content) {
          const slug = file.data.slug!
          if (!changedSlugs.has(slug)) continue
          if (slug !== 'are.na') continue
          if (!file.data.arenaData) continue

          const channels = file.data.arenaData.channels
          const plan = planArenaPartialEmit(arenaEmitState, channels)
          if (!plan.hasChanges) {
            arenaEmitState = plan.nextState
            continue
          }

          yield processArenaIndex(ctx, tree, file.data, allFiles, indexOpts, resources)

          const changedChannelFiles = await mapConcurrent(
            plan.changedChannels,
            defaultIoConcurrency,
            async channel => {
              const previous = arenaEmitState?.channelStates.get(channel.slug)
              return processChangedChannelOutputs(
                ctx,
                channel,
                previous,
                file.data,
                allFiles,
                channelOpts,
                resources,
              )
            },
          )

          for (const files of changedChannelFiles) {
            yield* files
          }

          await mapConcurrent(
            [...plan.deletedChannels],
            defaultIoConcurrency,
            ([channelSlug, previous]) =>
              removeChannelOutputs(ctx, channelSlug, previous.jsonEnabled),
          )

          yield emitSearchIndex(ctx, buildSearchIndex(channels))
          arenaEmitState = plan.nextState
        }
      })()
    },
    externalResources: () => ({ additionalHead: [] }),
  }
}
