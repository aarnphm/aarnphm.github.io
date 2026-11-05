import { Element, ElementContent } from 'hast'
import { headingRank } from 'hast-util-heading-rank'
import { h } from 'hastscript'
import type { TranscludeOptions } from '../../plugins/transformers/frontmatter'
import { i18n } from '../../i18n'
import { renderBaseViewsForFile } from '../../util/base/render'
import { FilePath, FullSlug, normalizeHastElement, slugifyFilePath } from '../../util/path'
import { referenceSourcePath } from '../../util/references/context'
import { hasHastClass, readTranscludeTarget } from '../../util/transclude-props'
import { buildBaseEmbedNodes, type BaseRenderedView, type BaseRenderResult } from './base-embeds'
import { transcludeCurius } from './transclude-curius'
import { transcludeNoteStack, transcludePageStatus } from './transclude-note-stack'
import { occurrenceContent } from './transclude-occurrence'
import {
  pruneLeadingHeading,
  renderNotebookCellTransclude,
  transcludeAnchor,
  wrapCollapsible,
  type TranscludeScope,
} from './transclude-scope'
import { selectTransclusion } from './transclude-selection'
import {
  calendarEmbedView,
  transcludeCalendar,
  transcludeTriathlonCard,
} from './transclude-triathlon'

export function transcludeBlockquote(scope: TranscludeScope, node: Element): void {
  const { componentData, slug, opts, stats, renderData } = scope
  const { cfg, allFiles, fileData } = componentData
  if (!hasHastClass(node, 'transclude')) {
    return
  }
  if (opts.skipTranscludes) {
    return
  }
  const transclude = readTranscludeTarget(node)
  if (!transclude) return
  const { fragments, selector, inner, targetSlug: transcludeTarget, url, alias } = transclude
  if (transcludeNoteStack(scope, node, transclude.request)) return
  if (node.data?.noteStackMember) {
    node.tagName = 'div'
    node.properties.className = ['note-stack-content']
  }
  let blockRef = transclude.blockRef
  const showSource =
    fileData.frontmatter?.pageLayout !== 'reflection' && !node.data?.noteStackMember

  let baseViewSlug: FullSlug | undefined =
    selector.kind === 'base' && transcludeTarget !== transclude.request.resolvedSlug
      ? transcludeTarget
      : undefined
  let page = renderData.bySlug.get(transcludeTarget)
  if (!page && transclude.request.resolvedPath) {
    const source = transclude.request.resolvedPath
    page = allFiles.find(file => referenceSourcePath(file) === source)
  }
  if (!page) {
    const baseMatch = renderData.baseFiles.find(
      f => f.slug && transcludeTarget.startsWith(`${f.slug}/`),
    )
    if (baseMatch) {
      page = baseMatch
      baseViewSlug = transcludeTarget
    }
  }
  if (!page) {
    return
  }
  const status = transcludePageStatus(page)
  if (status) {
    node.properties['data-transclude-status'] = status
    return
  }

  const calendarView =
    page.frontmatter?.layout === 'triathlon' ? calendarEmbedView(fragments) : undefined
  const transcludeMetadata = transclude.metadata ? { ...transclude.metadata } : undefined

  let transcludePageOpts: TranscludeOptions
  if (page.frontmatter?.transclude) {
    transcludePageOpts = { ...opts, ...page.frontmatter?.transclude }
  } else {
    transcludePageOpts = opts
  }

  if (calendarView) {
    transcludeCalendar(scope, node, page, calendarView, alias, transcludeMetadata)
    return
  }

  if (page?.readingTime && !stats.files.has(page.filePath!)) {
    stats.words += page.readingTime.words
    stats.minutes += page.readingTime.minutes
    stats.files.add(page.filePath!)
  }

  const { headings } = transcludePageOpts
  const title = transcludePageOpts.title && !node.data?.noteStackMember

  if (
    page.htmlAst &&
    renderNotebookCellTransclude(
      scope,
      node,
      page.htmlAst,
      transcludeTarget,
      page.frontmatter?.title,
      selector,
      fragments,
      inner,
      alias,
      transcludeMetadata,
    )
  ) {
    return
  }

  if (transcludeCurius(node, transcludeTarget, fragments, inner, alias)) return
  if (transcludeTriathlonCard(scope, node, page, fragments, inner)) return

  if (blockRef?.startsWith('#^')) {
    const selected = selectTransclusion(page, blockRef)
    const content = selected && occurrenceContent(scope, selected, transcludeTarget)
    if (!content) {
      if (selected) node.properties['data-transclude-status'] = 'budget'
      return
    }
    blockRef = blockRef.slice('#^'.length)
    const children = [...content]
    if (showSource) {
      children.push(
        h('a', { href: inner.properties?.href, class: 'internal transclude-src' }, [
          {
            type: 'text',
            value:
              page.frontmatter?.title ?? i18n(cfg.locale).components.transcludes.linkToOriginal,
          },
        ]),
      )
    }

    if (transcludeMetadata && 'collapsed' in transcludeMetadata) {
      const titleText = alias || `Block: ${blockRef}`
      wrapCollapsible(node, children, titleText, Boolean(transcludeMetadata.collapsed))
    } else {
      node.children = children
    }
  } else if (blockRef?.startsWith('#') && page.htmlAst) {
    const selected = selectTransclusion(page, blockRef)
    const normalizedSection = selected && occurrenceContent(scope, selected, transcludeTarget)
    if (!normalizedSection) {
      if (selected) node.properties['data-transclude-status'] = 'budget'
      return
    }
    blockRef = blockRef.slice(1)
    if (!headings) {
      const heading = normalizedSection.find(
        (child): child is Element => child.type === 'element' && headingRank(child) !== undefined,
      )
      if (typeof heading?.properties.id === 'string') node.properties.id = heading.properties.id
    }
    const sectionContent = headings ? normalizedSection : pruneLeadingHeading(normalizedSection)

    const children = [
      transcludeAnchor(inner.properties?.href as string, url, alias, title),
      ...sectionContent,
    ]

    if (showSource) {
      children.push(
        h('a', { href: inner.properties?.href, class: 'internal transclude-src' }, [
          {
            type: 'text',
            value:
              page.frontmatter?.title ?? i18n(cfg.locale).components.transcludes.linkToOriginal,
          },
        ]),
      )
    }

    const validChildren = children.filter(c => c !== null) as ElementContent[]
    if (transcludeMetadata && 'collapsed' in transcludeMetadata) {
      const titleText = alias || page.frontmatter?.title || `Section: ${blockRef}`
      wrapCollapsible(node, validChildren, titleText, Boolean(transcludeMetadata.collapsed))
    } else {
      node.children = validChildren
    }
  } else if (page.htmlAst || page.bases) {
    let baseTree = page.htmlAst
    let baseRendered: BaseRenderResult | undefined
    let baseView: BaseRenderedView | undefined
    let baseViewPinned = false
    if (page.bases && page.basesConfig && page.basesExpressions && page.slug) {
      baseRendered = renderBaseViewsForFile(page, allFiles, fileData)
      const baseSlug = page.slug as FullSlug
      let targetSlug = baseViewSlug ?? baseSlug
      const aliasText = typeof alias === 'string' && alias !== 'undefined' ? alias.trim() : ''
      if (!baseViewSlug && aliasText.length > 0) {
        const aliasSlug = slugifyFilePath(`${aliasText}.tmp` as FilePath, true)
        const aliasLower = aliasText.toLowerCase()
        const match = baseRendered.allViews.find(view => {
          const viewName = view.name.trim().toLowerCase()
          if (viewName === aliasLower) return true
          const viewSegment = view.slug.split('/').pop() || ''
          const aliasSegment = aliasSlug.split('/').pop() || ''
          return viewSegment === aliasSegment
        })
        if (match) {
          targetSlug = match.slug
        }
      }
      const pinnedView = baseRendered.views.find(entry => entry.slug === targetSlug)
      baseViewPinned = targetSlug !== baseSlug && pinnedView !== undefined
      baseView = pinnedView ?? baseRendered.views[0]
      baseTree = baseView ? baseView.tree : baseTree
    }
    if (!baseTree) {
      return
    }
    if (page.bases && baseRendered && baseView) {
      const { barNodes, viewNodes } = buildBaseEmbedNodes(
        page,
        componentData,
        baseRendered,
        baseView,
        baseViewPinned,
      )
      const normalizedViewNodes = viewNodes.map(child =>
        normalizeHastElement(child as Element, slug, transcludeTarget),
      )
      node.tagName = 'div'
      node.properties = { className: ['base-embed'] }
      const normalizedBarNodes = barNodes.map(child =>
        normalizeHastElement(child as Element, slug, transcludeTarget),
      )
      node.children = [...normalizedBarNodes, ...normalizedViewNodes]
      return
    }
    const selected = selectTransclusion({ ...page, htmlAst: baseTree }, undefined)
    const content = selected && occurrenceContent(scope, selected, transcludeTarget)
    if (!content) {
      if (selected) node.properties['data-transclude-status'] = 'budget'
      return
    }
    const children = [
      transcludeAnchor(inner.properties?.href as string, url, alias, title),
      title
        ? h('h1', [
            {
              type: 'text',
              value:
                page.frontmatter?.title ??
                i18n(cfg.locale).components.transcludes.transcludeOf({ targetSlug: page.slug! }),
            },
          ])
        : null,
      ...content,
    ]

    if (showSource) {
      children.push(
        h('a', { href: inner.properties?.href, class: 'internal transclude-src' }, [
          {
            type: 'text',
            value:
              page.frontmatter?.title ?? i18n(cfg.locale).components.transcludes.linkToOriginal,
          },
        ]),
      )
    }

    const validChildren = children.filter(c => c !== null) as ElementContent[]
    if (transcludeMetadata && 'collapsed' in transcludeMetadata) {
      const titleText = alias || page.frontmatter?.title || page.slug || 'Transclude'
      wrapCollapsible(node, validChildren, titleText, Boolean(transcludeMetadata.collapsed))
    } else {
      node.children = validChildren
    }
  }
}
