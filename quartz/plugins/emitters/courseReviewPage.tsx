import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { FlashcardsContent } from '../../components'
import HeaderConstructor from '../../components/Header'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { FullSlug, pathToRoot } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { defaultProcessedContent, QuartzPluginData } from '../vfile'
import { write } from './helpers'

const emitterName = 'CourseReviewPage'
const reviewSlug = 'courses/review' as FullSlug
const prefix = 'courses/'

// One drill across every course deck. The page ships no cards; the client picks due and new
// cards from the manifest and fetches only the deck pages it needs.
async function emitReview(
  ctx: BuildCtx,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const [tree, vfile] = defaultProcessedContent({
    slug: reviewSlug,
    frontmatter: {
      title: 'review',
      tags: [],
      pageLayout: 'default',
      description: 'due flashcards across every course',
    },
  })
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(reviewSlug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: vfile.data,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }
  const content = renderPage(ctx, reviewSlug, componentData, opts, externalResources, false)
  return write({ ctx, content, slug: reviewSlug, ext: '.html' })
}

export const CourseReviewPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: FlashcardsContent({ merge: { prefix, exit: 'courses/index' as FullSlug } }),
    ...userOpts,
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
      yield emitReview(ctx, contentDataFor(content), opts, resources)
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const touched = changeEvents.some(event => event.file?.data.slug?.startsWith(prefix))
      if (!touched) return
      yield emitReview(ctx, contentDataFor(content), opts, resources)
    },
  }
}
