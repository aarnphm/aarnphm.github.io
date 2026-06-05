import { Node } from 'unist'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { FlashcardsContent } from '../../components'
import HeaderConstructor from '../../components/Header'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { flashcardsSlug } from '../../util/flashcards-path'
import { FullSlug, pathToRoot } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { QuartzPluginData } from '../vfile'
import { write } from './helpers'

const emitterName = 'FlashcardsPage'

async function processFlashcards(
  ctx: BuildCtx,
  tree: Node,
  fileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = flashcardsSlug(fileData.flashcards!.sourceSlug)
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

export const FlashcardsPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    pageBody: FlashcardsContent(),
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
        if (!file.data.flashcards) continue
        yield processFlashcards(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = contentDataFor(content)

      const changedSlugs = new Set<FullSlug>()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          changedSlugs.add(changeEvent.file.data.slug!)
        }
      }

      for (const [tree, file] of content) {
        if (!file.data.flashcards) continue
        if (!changedSlugs.has(file.data.slug!)) continue
        yield processFlashcards(ctx, tree, file.data, allFiles, opts, resources)
      }
    },
  }
}
