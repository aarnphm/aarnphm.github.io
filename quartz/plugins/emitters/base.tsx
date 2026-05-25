import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { Content, BaseSearchBar, BaseViewSelector } from '../../components'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BaseExpressionDiagnostic, BasesExpressions } from '../../util/base/compiler'
import {
  BaseRenderPlan,
  BaseViewPartialState,
  planBaseViewPartialEmit,
} from '../../util/base/partial-emit'
import { BaseMetadata } from '../../util/base/render'
import { BuildCtx } from '../../util/ctx'
import { pathToRoot, FullSlug } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { QuartzPluginData } from '../vfile'
import { write } from './helpers'

async function* emitBaseViewsForPlan(
  ctx: BuildCtx,
  baseSlug: FullSlug,
  plan: BaseRenderPlan,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
  layout: FullPageLayout,
) {
  const { baseData, rendered } = plan

  for (const renderedView of rendered.views) {
    const slug = renderedView.slug
    const fileData: QuartzPluginData = { ...baseData }
    fileData.slug = slug
    fileData.htmlAst = renderedView.tree
    const frontmatter = fileData.frontmatter
    const title =
      typeof frontmatter?.title === 'string' && frontmatter.title.length > 0
        ? frontmatter.title
        : baseSlug
    fileData.frontmatter = {
      ...frontmatter,
      title: `${title} - ${renderedView.view.name}`,
      pageLayout: frontmatter?.pageLayout ?? 'default',
    }
    fileData.basesMetadata = {
      baseSlug,
      currentView: renderedView.view.name,
      allViews: rendered.allViews,
    }

    const cfg = ctx.cfg.configuration
    const externalResources = pageResources(pathToRoot(slug), resources, ctx)
    const componentData: QuartzComponentProps = {
      ctx,
      fileData,
      externalResources,
      cfg,
      children: [],
      tree: renderedView.tree,
      allFiles,
    }

    const content = renderPage(ctx, slug, componentData, layout, externalResources, false)
    yield write({ ctx, content, slug, ext: '.html' })
  }
}

export const BasePage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    pageBody: Content(),
    beforeBody: [BaseViewSelector(), BaseSearchBar()],
    afterBody: [],
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  let partialState: BaseViewPartialState | undefined

  return {
    name: 'BaseViewPage',
    getQuartzComponents() {
      return [Head, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const plan = planBaseViewPartialEmit(content, changeEvents, partialState)
      partialState = plan.nextState

      for (const slug of plan.slugsToRebuild) {
        const basePlan = plan.basePlans.get(slug)
        if (!basePlan) continue
        yield* emitBaseViewsForPlan(ctx, slug, basePlan, plan.allFiles, resources, opts)
      }
    },
    async *emit(ctx, content, resources) {
      const plan = planBaseViewPartialEmit(content, [], undefined)
      partialState = plan.nextState

      for (const [slug, basePlan] of plan.basePlans) {
        yield* emitBaseViewsForPlan(ctx, slug, basePlan, plan.allFiles, resources, opts)
      }
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    basesMetadata: BaseMetadata
    basesDiagnostics?: BaseExpressionDiagnostic[]
    basesExpressions?: BasesExpressions
  }
}
