import { rm } from 'node:fs/promises'
import { Node } from 'unist'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { HeadingsConstructor, Content } from '../../components'
import HeaderConstructor from '../../components/Header'
import {
  pageResources,
  renderPage,
  CuriusContent,
  CuriusFriends,
  CuriusNavigation,
} from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import { FilePath, FullSlug, joinSegments, pathToRoot } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { QuartzPluginData } from '../vfile'
import { write } from './helpers'

const isContentPage = (fileData: QuartzPluginData): boolean => {
  const slug = fileData.slug
  if (!slug) return false
  return !(
    slug.endsWith('/index') ||
    slug.startsWith('tags/') ||
    fileData.bases ||
    fileData.jsonCanvas ||
    fileData.streamData ||
    fileData.frontmatter?.layout === 'masonry'
  )
}

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

  const content = renderPage(ctx, slug, componentData, opts, externalResources, false)
  return write({ ctx, content, slug, ext: '.html' })
}

async function deleteContent(ctx: BuildCtx, slug: FullSlug): Promise<void> {
  const dest = joinSegments(ctx.argv.output, `${slug}.html`) as FilePath
  await rm(dest, { force: true })
}

export const ContentPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: Content(),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()
  const Headings = HeadingsConstructor()

  return {
    name: 'ContentPage',
    getQuartzComponents() {
      return [
        Head,
        Header,
        CuriusFriends,
        CuriusContent,
        CuriusNavigation,
        Headings,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...sidebar,
        Footer,
      ]
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map(c => c[1].data)

      for (const [tree, file] of content) {
        if (!isContentPage(file.data)) continue
        yield processContent(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map(c => c[1].data)

      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        const fileData = changeEvent.file.data
        if (changeEvent.type === 'delete') {
          if (isContentPage(fileData)) {
            await deleteContent(ctx, fileData.slug as FullSlug)
          }
          continue
        }
        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          changedSlugs.add(fileData.slug!)
        }
      }

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (!changedSlugs.has(slug)) continue
        if (!isContentPage(file.data)) continue

        yield processContent(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
  }
}
