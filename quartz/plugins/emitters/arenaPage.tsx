import { Element, ElementContent, Root } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { toMdast } from 'hast-util-to-mdast'
import { h } from 'hastscript'
import { gfmToMarkdown } from 'mdast-util-gfm'
import { toMarkdown } from 'mdast-util-to-markdown'
import fs from 'node:fs/promises'
import { render } from 'preact-render-to-string'
import { Node } from 'unist'
import { sharedPageComponents, defaultContentPageLayout } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import {
  ARENA_CARD_PAGE_SIZE,
  arenaCardPageSource,
  arenaChannelAssets,
  arenaModalSource,
  type ArenaSectionName,
} from '../../components/arena/channel-data'
import {
  arenaChannelSections,
  createArenaChannelRenderer,
} from '../../components/arena/ChannelBlock'
import HeaderConstructor from '../../components/Header'
import ArenaEntry from '../../components/pages/ArenaEntry'
import ArenaFeed from '../../components/pages/ArenaFeed'
import ArenaIndex from '../../components/pages/ArenaIndex'
import ChannelContent from '../../components/pages/ChannelContent'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { ChangeEvent, QuartzEmitterPlugin } from '../../types/plugin'
import { buildArenaFeedManifest, type ArenaFeedManifest } from '../../util/arena-feed'
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
import { removeWritten, write } from './helpers'
import { llmText } from './llm'

function blockSource(block: ArenaBlock): Element {
  const node = h('li', [
    block.htmlNode ??
      h('p', block.url ? h('a', { href: block.url }, block.content) : block.content),
  ])
  const metadata = {
    ...block.metadata,
    tags: block.tags ?? block.metadata?.tags,
    pinned: block.pinned ?? block.metadata?.pinned,
    later: block.later ?? block.metadata?.later,
    highlighted: block.highlighted || undefined,
    importance: block.importance,
    coord: block.coordinates
      ? `${block.coordinates.lat}, ${block.coordinates.lon}`
      : block.metadata?.coord,
  }
  const fields = Object.entries(metadata)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) =>
      h('li', `${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`),
    )
  const children: Element[] = []
  if (fields.length > 0) children.push(h('li', ['[meta]:', h('ul', fields)]))
  children.push(...(block.subItems ?? []).map(blockSource))
  if (children.length > 0) node.children.push(h('ul', children))
  return node
}

function channelSource(channel: ArenaChannel): string {
  const tree: Root = {
    type: 'root',
    children: [h('h1', channel.name), h('ul', channel.blocks.map(blockSource))],
  }
  return toMarkdown(toMdast(tree), { bullet: '-', emphasis: '_', extensions: [gfmToMarkdown()] })
}

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
  const files = [await write({ ctx, content, slug: channelSlug, ext: '.html' })]
  if (baseFileData.frontmatter?.protected !== true) {
    files.push(await llmText(ctx, { ...componentData.fileData, llmsText: channelSource(channel) }))
  } else {
    await removeWritten(ctx, channelSlug, '.md')
  }
  const assetBase = arenaChannelAssets(channel.slug)
  const sections = arenaChannelSections(channel)
  const ordered = [...sections.pinned, ...sections.later, ...sections.blocks]
  const renderBlock = createArenaChannelRenderer(componentData, channel)
  const modalFiles = await mapConcurrent(ordered, defaultIoConcurrency, (block, index) =>
    write({
      ctx,
      slug: arenaModalSource(assetBase, block.id).slice(1),
      ext: '',
      content: render(renderBlock(block, index, 'modal')),
    }),
  )
  files.push(...modalFiles)

  const sectionNames: ArenaSectionName[] = ['pinned', 'later', 'blocks']
  let startIndex = 0
  for (const name of sectionNames) {
    const blocks = sections[name]
    for (let offset = 0; offset < blocks.length; offset += ARENA_CARD_PAGE_SIZE) {
      const loaded = Math.min(offset + ARENA_CARD_PAGE_SIZE, blocks.length)
      const html = blocks
        .slice(offset, loaded)
        .map((block, index) => render(renderBlock(block, startIndex + offset + index)))
        .join('')
      files.push(
        await write({
          ctx,
          slug: arenaCardPageSource(assetBase, name, offset).slice(1),
          ext: '',
          content: JSON.stringify({ html, offset, total: blocks.length }),
        }),
      )
    }
    startIndex += blocks.length
  }

  const searchIndex: ArenaSearchIndex = {
    version: '1.0.0',
    channels: [
      { id: channel.id, slug: channel.slug, name: channel.name, blockCount: ordered.length },
    ],
    blocks: ordered.map(block => ({
      id: block.id,
      entryId: block.entryId,
      channelSlug: channel.slug,
      channelName: channel.name,
      title: block.title,
      content: block.content,
      url: block.url,
      tags: block.tags,
      metadata: block.metadata,
      highlighted: block.highlighted ?? false,
      pinned: block.pinned ?? false,
      later: block.later ?? false,
      hasModalInDom: false,
    })),
  }
  files.push(
    await write({
      ctx,
      slug: `${assetBase.slice(1)}/search`,
      ext: '.json',
      content: JSON.stringify(searchIndex),
    }),
  )
  return files
}

async function processEntries(
  ctx: BuildCtx,
  channel: ArenaChannel,
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
): Promise<FilePath[]> {
  return mapConcurrent(channel.blocks, defaultIoConcurrency, async block => {
    if (!block.entryId) throw new Error(`Arena entry ID missing in ${channel.slug}: ${block.id}`)
    const slug = joinSegments('arena', channel.slug, block.entryId) as FullSlug
    const frontmatter = {
      ...baseFileData.frontmatter,
      title: block.title ?? block.content,
      description: block.content.slice(0, 240),
      pageLayout: 'default' as const,
    }
    const [tree] = defaultProcessedContent({ slug, frontmatter })
    const externalResources = pageResources(pathToRoot(slug), resources, ctx)
    const componentData: QuartzComponentProps = {
      ctx,
      fileData: {
        ...baseFileData,
        slug,
        arenaData: undefined,
        arenaChannel: channel,
        arenaEntry: block,
        frontmatter,
      },
      externalResources,
      cfg: ctx.cfg.configuration,
      children: [],
      tree,
      allFiles,
    }
    const content = renderPage(ctx, slug, componentData, opts, externalResources, false)
    return write({ ctx, content, slug, ext: '.html' })
  })
}

async function processArenaFeed(
  ctx: BuildCtx,
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = 'arena/feed' as FullSlug
  const [tree, file] = defaultProcessedContent({
    ...baseFileData,
    slug,
    arenaData: undefined,
    arenaChannel: undefined,
    frontmatter: {
      ...baseFileData.frontmatter,
      title: 'arena | reader',
      description: 'Read saved links and keep notes.',
      pageLayout: 'default',
    },
  })
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: file.data,
    externalResources,
    cfg: ctx.cfg.configuration,
    children: [],
    tree,
    allFiles,
  }
  const content = renderPage(ctx, slug, componentData, opts, externalResources, false)
  return write({ ctx, content, slug, ext: '.html' })
}

function serializeBlock(
  block: ArenaBlock,
  channelSlug: string,
  channelName: string,
  hasModalInDom: boolean,
): ArenaBlockSearchable {
  const searchable: ArenaBlockSearchable = {
    id: block.id,
    entryId: block.entryId,
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

async function emitFeedManifest(ctx: BuildCtx, manifest: ArenaFeedManifest) {
  return write({ ctx, content: JSON.stringify(manifest), slug: 'static/arena-feed', ext: '.json' })
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
  entryOpts: FullPageLayout,
  resources: StaticResources,
): Promise<FilePath[]> {
  const files = await processChannel(ctx, channel, baseFileData, allFiles, opts, resources)
  files.push(...(await processEntries(ctx, channel, baseFileData, allFiles, entryOpts, resources)))
  if (isArenaChannelJsonEnabled(channel)) {
    files.push(await processChannelJson(ctx, channel))
  }
  return files
}

async function processChangedChannelOutputs(
  ctx: BuildCtx,
  channel: ArenaChannel,
  previous: { jsonEnabled: boolean; entryIds: string[] } | undefined,
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  entryOpts: FullPageLayout,
  resources: StaticResources,
): Promise<FilePath[]> {
  const files = await processChannelOutputs(
    ctx,
    channel,
    baseFileData,
    allFiles,
    opts,
    entryOpts,
    resources,
  )
  const currentIds = new Set(channel.blocks.map(block => block.entryId))
  for (const entryId of previous?.entryIds ?? []) {
    if (!currentIds.has(entryId)) {
      await removeWritten(ctx, joinSegments('arena', channel.slug, entryId), '.html')
    }
  }
  if (!isArenaChannelJsonEnabled(channel) && previous?.jsonEnabled) {
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
  await removeWritten(ctx, joinSegments('arena', channelSlug), '.md')
  await fs.rm(joinSegments(ctx.argv.output, arenaChannelAssets(channelSlug).slice(1)), {
    recursive: true,
    force: true,
  })
  await fs.rm(joinSegments(ctx.argv.output, 'arena', channelSlug), { recursive: true, force: true })
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

  const feedOpts: FullPageLayout = { ...indexOpts, beforeBody: [], pageBody: ArenaFeed() }
  const entryOpts: FullPageLayout = { ...indexOpts, beforeBody: [], pageBody: ArenaEntry() }

  const { head: Head, footer: Footer } = sharedPageComponents
  const Header = HeaderConstructor()
  let arenaEmitState: ArenaEmitState | undefined

  return {
    name: 'ArenaPage',
    getQuartzComponents(ctx) {
      if (ctx.argv.watch && !ctx.argv.force) return []

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
        feedOpts.pageBody,
        entryOpts.pageBody,
        Footer,
      ]
    },
    async *emit(ctx, content, resources) {
      if (ctx.argv.watch && !ctx.argv.force) return

      const allFiles = contentDataFor(content)

      for (const [tree, file] of content) {
        const slug = file.data.slug!

        if (slug !== 'are.na') continue
        if (!file.data.arenaData) continue

        const channels = file.data.arenaData.channels
        const manifest = await buildArenaFeedManifest(
          channels,
          ctx.cfg.configuration.baseUrl ?? 'aarnphm.xyz',
        )
        yield processArenaIndex(ctx, tree, file.data, allFiles, indexOpts, resources)
        yield processArenaFeed(ctx, file.data, allFiles, feedOpts, resources)
        yield emitFeedManifest(ctx, manifest)

        const channelFiles = await mapConcurrent(channels, defaultIoConcurrency, channel =>
          processChannelOutputs(
            ctx,
            channel,
            file.data,
            allFiles,
            channelOpts,
            entryOpts,
            resources,
          ),
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
      if (ctx.argv.watch && !ctx.argv.force) return null
      if (!hasArenaPageChange(changeEvents)) return null

      return (async function* () {
        const allFiles = contentDataFor(content)

        const hasArenaData = content.some(
          ([, file]) => file.data.slug === 'are.na' && file.data.arenaData,
        )
        if (!hasArenaData) {
          await Promise.all([
            removeWritten(ctx, 'arena', '.html'),
            removeWritten(ctx, 'arena/feed', '.html'),
            removeWritten(ctx, 'static/arena-feed', '.json'),
            removeWritten(ctx, 'static/arena-search', '.json'),
            ...[...(arenaEmitState?.channelStates ?? [])].map(([slug, state]) =>
              removeChannelOutputs(ctx, slug, state.jsonEnabled),
            ),
          ])
          arenaEmitState = undefined
          return
        }

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

          const manifest = await buildArenaFeedManifest(
            channels,
            ctx.cfg.configuration.baseUrl ?? 'aarnphm.xyz',
          )

          yield processArenaIndex(ctx, tree, file.data, allFiles, indexOpts, resources)
          yield processArenaFeed(ctx, file.data, allFiles, feedOpts, resources)
          yield emitFeedManifest(ctx, manifest)

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
                entryOpts,
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
