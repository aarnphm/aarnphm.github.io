import { Root } from 'hast'
import { h } from 'hastscript'
import { QuartzPluginData } from '../../plugins/vfile'
import { ReloadableWeakMap } from '../module-caches'
import { FilePath, FullSlug, joinSegments, simplifySlug, slugifyFilePath } from '../path'
import { isPublicBaseFile } from './browser-data'
import { BaseExpressionDiagnostic, EvalContext, evaluateFilterExpression, Value } from './compiler'
import { buildBookcase } from './render/bookcase'
import { buildCards } from './render/cards'
import { buildCalendar, buildList } from './render/list-calendar'
import { buildMap } from './render/map'
import {
  applySorting,
  fileGroupValues,
  type EvalContextFactory,
  type PropertyExprGetter,
} from './render/property-evaluation'
import { getFileLookup, type RenderElement } from './render/property-presentation'
import { buildTable } from './render/table'
import { buildVaultView } from './render/vault'
import { BaseFile, BaseView } from './types'

function resolveViewSlug(baseSlug: FullSlug, viewName: string, viewIndex: number): FullSlug {
  if (viewIndex === 0) return baseSlug
  const slugifiedName = slugifyFilePath((viewName + '.tmp') as FilePath, true)
  return joinSegments(baseSlug, slugifiedName) as FullSlug
}

function renderDiagnostics(
  diagnostics: BaseExpressionDiagnostic[] | undefined,
  currentSlug: FullSlug,
): RenderElement | undefined {
  if (!diagnostics || diagnostics.length === 0) {
    return undefined
  }
  const items: RenderElement[] = diagnostics.map((diag, index) => {
    const line = diag.span.start.line
    const column = diag.span.start.column
    const location = Number.isFinite(line) && Number.isFinite(column) ? `${line}:${column}` : ''
    const label = location.length > 0 ? `${diag.context} (${location})` : diag.context
    return h('li.base-diagnostics-item', { key: String(index) }, [
      h('div.base-diagnostics-label', label),
      h('div.base-diagnostics-message', diag.message),
      h('code.base-diagnostics-source', diag.source),
    ])
  })
  return h('div.base-diagnostics', [
    h('div.base-diagnostics-title', 'bases diagnostics'),
    h('div.base-diagnostics-meta', [
      h('span', 'page'),
      h('span.base-diagnostics-page', currentSlug),
    ]),
    h('ul.base-diagnostics-list', items),
  ])
}

export type BaseViewMeta = { name: string; type: BaseView['type']; slug: FullSlug }

export type RenderedBaseView = {
  view: BaseView
  slug: FullSlug
  tree: Root
  resultCount: number
  totalCount: number
  matchedSlugs: string[]
  diagnostics: BaseExpressionDiagnostic[]
}

export type BaseMetadata = { baseSlug: FullSlug; currentView: string; allViews: BaseViewMeta[] }

const publicFilesCache = new ReloadableWeakMap<QuartzPluginData[], QuartzPluginData[]>()

function getPublicBaseFiles(allFiles: QuartzPluginData[]): QuartzPluginData[] {
  let filtered = publicFilesCache.get(allFiles)
  if (!filtered) {
    filtered = allFiles.filter(isPublicBaseFile)
    publicFilesCache.set(allFiles, filtered)
  }
  return filtered
}

export function renderBaseViewsForFile(
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  thisFile?: QuartzPluginData,
): { views: RenderedBaseView[]; allViews: BaseViewMeta[] } {
  const config = baseFileData.basesConfig as BaseFile | undefined
  if (!config || !baseFileData.slug) {
    return { views: [], allViews: [] }
  }
  allFiles = getPublicBaseFiles(allFiles)

  const baseSlug = baseFileData.slug as FullSlug
  const expressions = baseFileData.basesExpressions
  const formulaExpressions = expressions?.formulas
  const summaryExpressionsByView = expressions?.viewSummaries
  const viewFilterExpressions = expressions?.viewFilters
  const formulaCaches = new Map<string, Map<string, Value>>()
  const propertyCaches = new Map<string, Map<string, Value>>()
  const runtimeDiagnostics: BaseExpressionDiagnostic[] = []
  const runtimeDiagnosticSet = new Set<string>()
  const propertyExpressions = expressions?.propertyExpressions ?? {}

  const fileIndex = getFileLookup(allFiles).bySimple

  const getPropertyExpr: PropertyExprGetter = property => {
    const key = property.trim()
    if (!key) return null
    return propertyExpressions[key] ?? null
  }

  const baseThisFile = thisFile ?? baseFileData
  const evalContexts = new Map<QuartzPluginData, EvalContext>()

  const getEvalContext: EvalContextFactory = file => {
    let ctx = evalContexts.get(file)
    if (!ctx) {
      const slug = file.slug ? String(file.slug) : file.filePath ? String(file.filePath) : ''
      let cache = formulaCaches.get(slug)
      if (!cache) {
        cache = new Map()
        formulaCaches.set(slug, cache)
      }
      let propertyCache = propertyCaches.get(slug)
      if (!propertyCache) {
        propertyCache = new Map()
        propertyCaches.set(slug, propertyCache)
      }
      ctx = {
        file,
        thisFile: baseThisFile,
        allFiles,
        fileIndex,
        formulas: formulaExpressions,
        formulaSources: config.formulas,
        formulaCache: cache,
        formulaStack: new Set(),
        propertyCache,
        diagnostics: runtimeDiagnostics,
        diagnosticSet: runtimeDiagnosticSet,
      }
      evalContexts.set(file, ctx)
    }
    return ctx
  }

  const allViews = config.views.map((view, idx) => ({
    name: view.name,
    type: view.type,
    slug: resolveViewSlug(baseSlug, view.name, idx),
  }))

  const baseFilterSource = typeof config.filters === 'string' ? config.filters : ''
  const baseMatchedFiles = expressions?.filters
    ? allFiles.filter(file => {
        const ctx = getEvalContext(file)
        ctx.diagnosticContext = 'filters'
        ctx.diagnosticSource = baseFilterSource
        return evaluateFilterExpression(expressions.filters!, ctx)
      })
    : allFiles

  const views: RenderedBaseView[] = []

  for (const [viewIndex, view] of config.views.entries()) {
    const slug = resolveViewSlug(baseSlug, view.name, viewIndex)

    let matchedFiles = baseMatchedFiles
    const viewFilter = viewFilterExpressions ? viewFilterExpressions[String(viewIndex)] : undefined
    if (viewFilter) {
      const viewFilterSource = typeof view.filters === 'string' ? view.filters : ''
      matchedFiles = matchedFiles.filter(file => {
        const ctx = getEvalContext(file)
        ctx.diagnosticContext = `views[${viewIndex}].filters`
        ctx.diagnosticSource = viewFilterSource
        return evaluateFilterExpression(viewFilter, ctx)
      })
    }

    const groupBy = view.groupBy
    const visibleGroups = view.groupOrder !== undefined ? new Set(view.groupOrder) : undefined
    const visibleFiles =
      groupBy && visibleGroups
        ? matchedFiles.filter(file =>
            fileGroupValues(file, groupBy, getEvalContext, getPropertyExpr).some(value =>
              visibleGroups.has(value),
            ),
          )
        : view.type === 'kanban' && !groupBy
          ? []
          : matchedFiles
    const sortedFiles = applySorting(visibleFiles, view.sort, getEvalContext, getPropertyExpr)
    const limitedFiles = view.limit !== undefined ? sortedFiles.slice(0, view.limit) : sortedFiles
    const totalCount = visibleFiles.length
    const resultCount = limitedFiles.length

    const wrapView = (node: RenderElement) =>
      h('div.base-view', { 'data-base-view-type': view.type, 'data-base-view-name': view.name }, [
        node,
      ])

    let viewNode: RenderElement | undefined
    switch (view.type) {
      case 'table':
        viewNode = buildTable(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
          config.summaries,
          summaryExpressionsByView ? summaryExpressionsByView[String(viewIndex)] : undefined,
        )
        break
      case 'list':
        viewNode = buildList(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'card':
      case 'cards':
      case 'gallery':
      case 'board':
      case 'kanban':
        viewNode = buildCards(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'bookcase':
        viewNode = buildBookcase(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'vault':
        viewNode = buildVaultView(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'calendar':
        viewNode = buildCalendar(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'map':
        viewNode = buildMap(
          limitedFiles,
          view,
          slug,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      default:
        viewNode = undefined
    }

    if (!viewNode) continue

    const diagnostics = [...(baseFileData.basesDiagnostics ?? []), ...runtimeDiagnostics]
    const diagnosticsNode = renderDiagnostics(
      diagnostics.length > 0 ? diagnostics : undefined,
      slug,
    )
    const wrapped = wrapView(viewNode)
    const tree: Root = {
      type: 'root',
      children: diagnosticsNode ? [diagnosticsNode, wrapped] : [wrapped],
    }

    const matchedSlugs = matchedFiles.flatMap(file =>
      file.slug ? [simplifySlug(file.slug as FullSlug)] : [],
    )

    views.push({ view, slug, tree, resultCount, totalCount, matchedSlugs, diagnostics })
  }

  return { views, allViews }
}
