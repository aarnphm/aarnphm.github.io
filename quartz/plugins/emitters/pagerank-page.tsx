import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import HeaderConstructor from '../../components/Header'
import PageRankContent from '../../components/PageRankContent'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { FullSlug, pathToRoot } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { defaultProcessedContent, ProcessedContent, QuartzPluginData } from '../vfile'
import { write } from './helpers'

const emitterName = 'PageRankPage'

const pageSlug = 'pagerank' as FullSlug

// content/pagerank.md (layout: pagerank) supplies the title, description and OG image; the
// built-in frontmatter keeps the route when that note is absent.
const sourceOf = (content: ProcessedContent[]) =>
  content.find(([, vfile]) => vfile.data.slug === pageSlug) ??
  defaultProcessedContent({
    slug: pageSlug,
    frontmatter: {
      title: 'pagerank',
      pageLayout: 'default',
      tags: [],
      description: 'global note ranking and links across the garden',
    },
  })

async function emitPage(
  ctx: BuildCtx,
  [tree, vfile]: ProcessedContent,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(pageSlug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: vfile.data,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }
  const content = renderPage(ctx, pageSlug, componentData, opts, externalResources, false)
  return write({ ctx, content, slug: pageSlug, ext: '.html' })
}

export const PageRankPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: PageRankContent(),
    ...userOpts,
    header: [],
    beforeBody: [],
    afterBody: [],
    sidebar: [],
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  const Header = HeaderConstructor()

  return {
    name: emitterName,
    getQuartzComponents() {
      return [Head, Header, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *emit(ctx, content, resources) {
      yield emitPage(ctx, sourceOf(content), contentDataFor(content), opts, resources)
    },
    async *partialEmit(ctx, content, resources) {
      yield emitPage(ctx, sourceOf(content), contentDataFor(content), opts, resources)
    },
  }
}
