import { ElementContent, Root } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { h } from 'hastscript'
import { render } from 'preact-render-to-string'
import { visit } from 'unist-util-visit'
import { QuartzPluginData } from '../../plugins/vfile'
import { QuartzComponent, QuartzComponentProps } from '../../types/component'
import { compileBaseConfig } from '../../util/base/compile'
import { renderBaseViewsForFile } from '../../util/base/render'
import { FullSlug } from '../../util/path'
import BaseViewSelector from '../BaseViewSelector'

export type BaseRenderResult = ReturnType<typeof renderBaseViewsForFile>
export type BaseRenderedView = BaseRenderResult['views'][number]

const BaseViewSelectorComponent = BaseViewSelector()

function renderComponentNodes(
  component: QuartzComponent,
  props: QuartzComponentProps,
): ElementContent[] {
  const Component = component
  const html = render(<Component {...props} />)
  if (!html) return []
  const root = fromHtml(html, { fragment: true }) as Root
  return root.children as ElementContent[]
}

type BaseEmbedNodes = { barNodes: ElementContent[]; viewNodes: ElementContent[] }

export function buildBaseEmbedNodes(
  baseFileData: QuartzPluginData,
  componentData: QuartzComponentProps,
  rendered: BaseRenderResult,
  activeView: BaseRenderedView,
  pinned = false,
): BaseEmbedNodes {
  const baseSlug = baseFileData.slug as FullSlug
  const baseMetadata = { baseSlug, currentView: activeView.view.name, allViews: rendered.allViews }
  const embedData: QuartzComponentProps = {
    ...componentData,
    fileData: { ...componentData.fileData, basesMetadata: baseMetadata },
  }
  // A transclusion that names a view shows that view alone, so it has nothing to switch to.
  const selectorNodes = pinned ? [] : renderComponentNodes(BaseViewSelectorComponent, embedData)
  const resultsLabel =
    activeView.totalCount === activeView.resultCount
      ? `${activeView.totalCount} results`
      : `${activeView.resultCount} of ${activeView.totalCount} results`
  const resultsNode = h(
    'div.base-embed-results',
    { dataBaseEmbedResults: '' },
    h('span', { dataBaseEmbedResultsLabel: '' }, resultsLabel),
  )
  const barNode = h('div.base-embed-bar', [...selectorNodes, resultsNode])
  const views = pinned ? [activeView] : rendered.views
  const viewNodes = views.map(view => {
    const isActive = view.slug === activeView.slug
    return h(
      'div.base-embed-view',
      {
        className: isActive ? ['base-embed-view', 'is-active'] : ['base-embed-view'],
        dataBaseEmbedView: '',
        dataBaseViewName: view.view.name,
        dataBaseViewSlug: view.slug,
        dataBaseViewResultCount: view.resultCount,
        dataBaseViewTotalCount: view.totalCount,
        hidden: isActive ? undefined : true,
      },
      view.tree.children as ElementContent[],
    )
  })
  return { barNodes: [barNode], viewNodes }
}

export function renderBaseEmbeds(root: Root, componentData: QuartzComponentProps): void {
  const { allFiles, fileData } = componentData
  const slug = fileData.slug as FullSlug | undefined
  if (!slug) return

  visit(root, { tagName: 'div' }, node => {
    const baseSource = node.properties?.dataBaseSource as string | undefined
    if (!baseSource) return
    let decoded = baseSource
    try {
      decoded = decodeURIComponent(baseSource)
    } catch {}

    const compiled = compileBaseConfig(decoded, fileData.filePath ?? fileData.slug)
    const baseFileData: QuartzPluginData = {
      slug,
      filePath: fileData.filePath,
      bases: true,
      basesConfig: compiled.config,
      basesDiagnostics: compiled.diagnostics,
      basesExpressions: compiled.expressions,
    }

    const rendered = renderBaseViewsForFile(baseFileData, allFiles, fileData)
    const view = rendered.views[0]
    if (!view) return

    const className = node.properties?.className
    const classList = Array.isArray(className)
      ? className
      : typeof className === 'string'
        ? [className]
        : []
    if (!classList.includes('base-embed')) {
      classList.push('base-embed')
    }
    node.properties = { ...node.properties, className: classList }
    delete node.properties.dataBaseSource
    delete node.properties.dataBaseEmbed
    const { barNodes, viewNodes } = buildBaseEmbedNodes(baseFileData, componentData, rendered, view)
    node.children = [...barNodes, ...viewNodes]
  })
}
