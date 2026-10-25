import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import HeaderConstructor from '../../components/Header'
import PageRankContent from '../../components/PageRankContent'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { isDevelopmentBuild } from '../../util/build-mode'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { createNoteRanker } from '../../util/note-pagerank'
import { pageRankModel, type PageRankModel } from '../../util/pagerank-view'
import { FilePath, FullSlug, pathToRoot } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { defaultProcessedContent, ProcessedContent, QuartzPluginData } from '../vfile'
import { addGraphFilePages, buildLinkIndex } from './contentIndex/link-index'
import { write } from './helpers'
import { createOgImageGenerator } from './ogImage'

const emitterName = 'PageRankPage'

const pageSlug = 'pagerank' as FullSlug

// content/pagerank.md (layout: pagerank) supplies the title and the on-page description; the
// built-in frontmatter keeps the route when that note is absent. `generatedSocialImage` hands the
// OG image to this emitter, because its description comes from the ranking.
const sourceOf = (content: ProcessedContent[]) =>
  content.find(([, vfile]) => vfile.data.slug === pageSlug) ??
  defaultProcessedContent({
    slug: pageSlug,
    frontmatter: {
      title: 'pagerank',
      pageLayout: 'default',
      tags: [],
      description: 'global note ranking and links across the garden',
      generatedSocialImage: true,
    },
  })

// The page ranks static/contentIndex.json in the browser; this rebuilds that index (ContentIndex
// runs with includeEmptyFiles) so the social description quotes the same numbers.
const indexOptions = {
  enableSiteMap: false,
  enableAtom: false,
  includeEmptyFiles: true,
  enableSecurity: false,
}

const count = (value: number) => value.toLocaleString('en-US')

// The same model the page script builds, so the static page shows the first view it would draw.
function rankingModel(ctx: BuildCtx, content: ProcessedContent[]): PageRankModel {
  const index = buildLinkIndex(ctx, content, indexOptions)
  addGraphFilePages(ctx, index)
  const data = Object.fromEntries(index)
  return pageRankModel(createNoteRanker(data).graph(), slug => data[slug]?.title ?? '')
}

function rankingDescription({ summary }: PageRankModel): string {
  return (
    `pagerank over ${count(summary.notes)} notes and ${count(summary.links)} links. ` +
    `the top 10% of notes hold ${Math.round(summary.top10 * 100)}% of the score, ` +
    `and ${count(summary.noInbound)} notes have no inbound link.`
  )
}

async function emitPage(
  ctx: BuildCtx,
  content: ProcessedContent[],
  opts: FullPageLayout,
  resources: StaticResources,
): Promise<FilePath[]> {
  const [tree, vfile] = sourceOf(content)
  const model = rankingModel(ctx, content)
  const description = rankingDescription(model)
  // The source note has no body; reading time comes from the description, as on triathlon pages.
  const fileData: QuartzPluginData = {
    ...vfile.data,
    text: description,
    readingTime: undefined,
    frontmatter: { ...vfile.data.frontmatter!, socialDescription: description },
  }
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(pageSlug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles: contentDataFor(content),
  }
  const html = renderPage(
    ctx,
    pageSlug,
    componentData,
    { ...opts, pageBody: PageRankContent({ model }) },
    externalResources,
    false,
  )
  const page = write({ ctx, content: html, slug: pageSlug, ext: '.html' })
  if (isDevelopmentBuild(ctx.argv)) return [await page]
  const generateOgImage = await createOgImageGenerator(ctx)
  return Promise.all([page, generateOgImage(fileData)])
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
      yield* await emitPage(ctx, content, opts, resources)
    },
    async *partialEmit(ctx, content, resources) {
      yield* await emitPage(ctx, content, opts, resources)
    },
  }
}
