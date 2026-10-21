import { access } from 'node:fs/promises'
import type { BaseGeoPayload } from '../../util/base/geo-types'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { Content, BaseSearchBar, BaseViewSelector } from '../../components'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import {
  BASE_QUERY_DATA_SLUG,
  createBaseQueryData,
  createBaseQueryDocument,
  type BaseQueryDocument,
} from '../../util/base/browser-data'
import { BaseExpressionDiagnostic, BasesExpressions } from '../../util/base/compiler'
import {
  baseEmbeddingRecord,
  baseEmbeddingsDir,
  baseEmbeddingsManifestPath,
  baseEmbeddingsTextSlug,
} from '../../util/base/embeddings'
import { buildBaseGeoPayload } from '../../util/base/geo-payload'
import {
  BaseRenderPlan,
  BaseViewPartialState,
  planBaseViewPartialEmit,
} from '../../util/base/partial-emit'
import { buildBasePreview } from '../../util/base/preview'
import { BaseMetadata } from '../../util/base/render'
import { BuildCtx } from '../../util/ctx'
import { documentPreviewSlug } from '../../util/document-preview'
import { onModuleRelease } from '../../util/module-caches'
import { FilePath, pathToRoot, FullSlug, joinSegments, simplifySlug } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { QuartzPluginData } from '../vfile'
import { emitBookcaseCovers } from './bookcase-covers'
import { write } from './helpers'

type BaseViewRender = BaseRenderPlan['rendered']['views'][number]

async function emitBaseView(
  ctx: BuildCtx,
  baseSlug: FullSlug,
  plan: BaseRenderPlan,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
  layout: FullPageLayout,
  renderedView: BaseViewRender,
  semanticManifest: string | undefined,
  queryDocument: ReturnType<typeof createBaseQueryDocument>,
): Promise<FilePath> {
  const { baseData, rendered } = plan
  const slug = renderedView.slug
  const fileData: QuartzPluginData = { ...baseData }
  fileData.basesQueryDocument = queryDocument
  fileData.slug = slug
  fileData.htmlAst = renderedView.tree
  if (semanticManifest) fileData.basesSemantic = { manifest: semanticManifest }
  const basesGeo = buildBaseGeoPayload(plan.baseData.basesConfig, allFiles, plan.memberSlugs)
  if (basesGeo) fileData.basesGeo = basesGeo
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
  return write({ ctx, content, slug, ext: '.html' })
}

// The vectors are built offline from the JSONL (see docs/semantic-search.md); the page only gets
// the toggle when that build has landed in the content tree.
async function baseSemanticManifest(
  ctx: BuildCtx,
  baseSlug: FullSlug,
): Promise<string | undefined> {
  const manifest = joinSegments(ctx.argv.directory, baseEmbeddingsDir, baseSlug, 'manifest.json')
  return access(manifest).then(
    () => baseEmbeddingsManifestPath(baseSlug),
    () => undefined,
  )
}

function emitBaseEmbeddingText(
  ctx: BuildCtx,
  baseSlug: FullSlug,
  memberSlugs: ReadonlySet<string>,
  allFiles: QuartzPluginData[],
): Promise<FilePath> {
  const lines: string[] = []
  for (const file of allFiles) {
    if (!file.slug || !memberSlugs.has(simplifySlug(file.slug))) continue
    const record = baseEmbeddingRecord(file)
    if (record) lines.push(JSON.stringify(record))
  }
  return write({
    ctx,
    slug: baseEmbeddingsTextSlug(baseSlug),
    ext: '.jsonl',
    content: lines.join('\n'),
  })
}

async function emitBaseViewsForPlan(
  ctx: BuildCtx,
  baseSlug: FullSlug,
  plan: BaseRenderPlan,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
  layout: FullPageLayout,
): Promise<FilePath[]> {
  const queryDocument = createBaseQueryDocument(plan.baseData)
  const [semanticManifest, covers] = await Promise.all([
    baseSemanticManifest(ctx, baseSlug),
    emitBookcaseCovers(ctx, plan.rendered.views),
  ])
  // Every view route shares the base summary, so a link to any view previews the same cards.
  const preview = buildBasePreview(baseSlug, plan.baseData, plan.rendered.views)
  const previewSlugs = new Set([baseSlug, ...plan.rendered.views.map(view => view.slug)])
  const [views, previews, embeddingText] = await Promise.all([
    mapConcurrent(plan.rendered.views, defaultIoConcurrency, renderedView =>
      emitBaseView(
        ctx,
        baseSlug,
        plan,
        allFiles,
        resources,
        layout,
        renderedView,
        semanticManifest,
        queryDocument,
      ),
    ),
    Promise.all(
      [...previewSlugs].map(slug =>
        write({
          ctx,
          content: JSON.stringify({ ...preview, slug }),
          slug: documentPreviewSlug(slug),
          ext: '.json',
        }),
      ),
    ),
    emitBaseEmbeddingText(ctx, baseSlug, plan.memberSlugs, allFiles),
  ])
  return [...views, ...previews, embeddingText, ...covers]
}

export const BasePage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    pageBody: Content(),
    beforeBody: [BaseViewSelector(), BaseSearchBar()],
    afterBody: [],
    sidebar: [],
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, sidebar, footer: Footer } = opts
  let partialState: BaseViewPartialState | undefined
  onModuleRelease(() => {
    partialState = undefined
  })

  return {
    name: 'BaseViewPage',
    getQuartzComponents() {
      return [Head, ...header, ...beforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const plan = planBaseViewPartialEmit(content, changeEvents, partialState)
      partialState = plan.nextState
      const embeddingExports = new Map<FullSlug, ReadonlySet<string>>()
      // The view planner can skip body-only edits. Compare the exported row so those edits
      // refresh JSONL without rerendering unchanged HTML or changes beyond the excerpt.
      for (const change of changeEvents) {
        const current = change.file?.data
        const previous = change.previousFile?.data
        const slug = current?.slug ?? previous?.slug
        if (!slug) continue
        const currentRow = current ? baseEmbeddingRecord(current) : null
        const previousRow = previous ? baseEmbeddingRecord(previous) : null
        if (JSON.stringify(currentRow) === JSON.stringify(previousRow)) continue
        for (const [baseSlug, members] of plan.nextState.baseMemberSlugs) {
          if (!plan.slugsToRebuild.has(baseSlug) && members.has(simplifySlug(slug)))
            embeddingExports.set(baseSlug, members)
        }
      }
      if (plan.slugsToRebuild.size === 0 && embeddingExports.size === 0) return

      const allFiles = content.map(([, file]) => file.data)
      if (plan.slugsToRebuild.size > 0) {
        yield await write({
          ctx,
          slug: BASE_QUERY_DATA_SLUG,
          ext: '.json',
          content: JSON.stringify(createBaseQueryData(allFiles)),
        })
      }
      const [files, embeddingFiles] = await Promise.all([
        mapConcurrent(
          [...plan.basePlans].filter(([slug]) => plan.slugsToRebuild.has(slug)),
          defaultIoConcurrency,
          ([slug, basePlan]) =>
            emitBaseViewsForPlan(ctx, slug, basePlan, plan.allFiles, resources, opts),
        ),
        mapConcurrent([...embeddingExports], defaultIoConcurrency, ([slug, members]) =>
          emitBaseEmbeddingText(ctx, slug, members, allFiles),
        ),
      ])
      yield* files.flat()
      yield* embeddingFiles
    },
    async *emit(ctx, content, resources) {
      yield await write({
        ctx,
        slug: BASE_QUERY_DATA_SLUG,
        ext: '.json',
        content: JSON.stringify(createBaseQueryData(content.map(([, file]) => file.data))),
      })
      const plan = planBaseViewPartialEmit(content, [], undefined)
      partialState = plan.nextState

      const files = await mapConcurrent(
        [...plan.basePlans],
        defaultIoConcurrency,
        ([slug, basePlan]) =>
          emitBaseViewsForPlan(ctx, slug, basePlan, plan.allFiles, resources, opts),
      )
      yield* files.flat()
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    basesMetadata: BaseMetadata
    /** Set when an offline vector index exists for this base; the search bar shows the semantic toggle. */
    basesSemantic?: { manifest: string }
    basesGeo?: BaseGeoPayload
    basesDiagnostics?: BaseExpressionDiagnostic[]
    basesExpressions?: BasesExpressions
    basesQueryDocument?: BaseQueryDocument
  }
}
