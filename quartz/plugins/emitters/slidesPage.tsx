import { Node } from 'unist'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { SlidesContent } from '../../components'
import HeaderConstructor from '../../components/Header'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { pathToRoot, joinSegments, FullSlug } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { QuartzPluginData } from '../vfile'
import { write, removeWritten } from './helpers'

const emitterName = 'SlidesPage'

const slidesSlugFor = (baseSlug: FullSlug): FullSlug => joinSegments(baseSlug, 'slides') as FullSlug

async function deleteSlides(ctx: BuildCtx, baseSlug: FullSlug): Promise<void> {
  await removeWritten(ctx, slidesSlugFor(baseSlug), '.html')
}

async function processSlides(
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slidesSlug = slidesSlugFor(fileData.slug!)
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slidesSlug), resources, ctx)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }

  const content = renderPage(ctx, slidesSlug, componentData, opts, externalResources, false)
  return write({ ctx, content, slug: slidesSlug, ext: '.html' })
}

export const SlidesPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  // slim page layout for slides
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: SlidesContent(),
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
      const allFiles = contentDataFor(content)

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        // skip tag pages and everything that isn’t a primary content page
        if (slug.endsWith('/index') || slug.startsWith('tags/')) continue
        // Only emit slides if explicitly enabled via frontmatter
        if (!file.data.frontmatter?.slides) continue
        yield processSlides(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = contentDataFor(content)

      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (changeEvent.type !== 'add' && changeEvent.type !== 'change') continue
        const slug = changeEvent.file.data.slug! as FullSlug
        if (changeEvent.file.data.frontmatter?.slides) {
          changedSlugs.add(slug)
        } else if (changeEvent.previousFile?.data.frontmatter?.slides) {
          await deleteSlides(ctx, slug)
        }
      }

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (!changedSlugs.has(slug)) continue
        if (slug.endsWith('/index') || slug.startsWith('tags/')) continue
        if (!file.data.frontmatter?.slides) continue
        yield processSlides(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
  }
}
