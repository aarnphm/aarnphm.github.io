import type { Root } from 'hast'
import type { StreamEntry } from '../transformers/stream'
import { sharedPageComponents, defaultContentPageLayout } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import HeaderConstructor from '../../components/Header'
import StreamPageComponent from '../../components/pages/StreamPage'
import { pageResources, renderPage } from '../../components/renderPage'
import { isDraftEntry, isRestrictedEntry } from '../../components/stream/Entry'
import StreamSearchComponent from '../../components/StreamSearch'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { pathToRoot } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { buildStreamRouteTree, cloneStreamEntries } from '../../util/stream-route-tree'
import { QuartzPluginData } from '../vfile'
import { write } from './helpers'

const isPublishedStreamEntry = (entry: StreamEntry): boolean =>
  !isDraftEntry(entry) && !isRestrictedEntry(entry)

const filterUnpublishedStreamEntries = (fileData: QuartzPluginData): QuartzPluginData => {
  const entries = fileData.streamData?.entries
  if (!entries) return fileData

  const publishedEntries = entries.filter(isPublishedStreamEntry)
  if (publishedEntries.length === entries.length) return fileData

  return { ...fileData, streamData: { ...fileData.streamData, entries: publishedEntries } }
}

async function processStreamPage(
  ctx: BuildCtx,
  tree: Root,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = fileData.slug!
  const publishedFileData = filterUnpublishedStreamEntries(fileData)
  const streamData = publishedFileData.streamData
  const routeEntries = streamData ? cloneStreamEntries(streamData.entries) : []
  const routeFileData = streamData
    ? { ...publishedFileData, streamData: { ...streamData, entries: routeEntries } }
    : publishedFileData
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources, ctx)

  const componentData: QuartzComponentProps = {
    ctx,
    fileData: routeFileData,
    externalResources,
    cfg,
    children: [],
    tree: streamData ? buildStreamRouteTree(routeEntries, tree) : tree,
    allFiles,
  }

  const content = renderPage(ctx, slug, componentData, opts, externalResources, false)
  return write({ ctx, content, slug, ext: '.html' })
}

export const StreamPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const filteredHeader = sharedPageComponents.header.filter(component => {
    const name = component.displayName || component.name || ''
    return name !== 'Breadcrumbs' && name !== 'StackedNotes'
  })
  const filteredBefore = defaultContentPageLayout.beforeBody.filter(
    c => c.displayName !== 'Byline' || c.name !== 'Byline',
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

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()
  const StreamSearch = StreamSearchComponent()

  return {
    name: 'StreamPage',
    getQuartzComponents() {
      return [
        Head,
        Header,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...sidebar,
        Footer,
        StreamSearch,
      ]
    },
    async *emit(ctx, content, resources) {
      const allFiles = contentDataFor(content)

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        const slug = data.slug!

        if (slug !== 'stream' || !data.streamData) continue

        yield processStreamPage(ctx, tree, data, allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = contentDataFor(content)
      const changedSlugs = new Set<string>()

      for (const changeEvent of changeEvents) {
        if (changeEvent.file) {
          if (changeEvent.type === 'add' || changeEvent.type === 'change') {
            changedSlugs.add(changeEvent.file.data.slug!)
          }
          continue
        }

        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          const changedPath = changeEvent.path
          for (const [_, vf] of content) {
            const deps = (vf.data.codeDependencies as string[] | undefined) ?? []
            if (deps.includes(changedPath)) {
              changedSlugs.add(vf.data.slug!)
            }
          }
        }
      }

      if (!changedSlugs.has('stream')) return

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        const slug = data.slug!
        if (slug !== 'stream' || !data.streamData) continue

        yield processStreamPage(ctx, tree, data, allFiles, opts, resources)
      }
    },
  }
}
