import { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import type { TranscludeOptions } from '../../plugins/transformers/frontmatter'
import { i18n } from '../../i18n'
import { renderBaseViewsForFile } from '../../util/base/render'
import { FilePath, FullSlug, normalizeHastElement, slugifyFilePath } from '../../util/path'
import {
  findHeadingSectionBounds,
  hasHastClass,
  readTranscludeTarget,
  transcludeVisitKey,
} from '../../util/transclude-props'
import { buildBaseEmbedNodes, type BaseRenderedView, type BaseRenderResult } from './base-embeds'
import { appendTranscludedNotes, isTrailingNoteSection } from './footnotes-citations'
import { transcludeCurius } from './transclude-curius'
import {
  pruneLeadingHeading,
  renderNotebookCellTransclude,
  transcludeAnchor,
  wrapCollapsible,
  type TranscludeScope,
} from './transclude-scope'
import {
  calendarEmbedView,
  transcludeCalendar,
  transcludeTriathlonCard,
} from './transclude-triathlon'

export function transcludeBlockquote(scope: TranscludeScope, node: Element): void {
  const { componentData, root, slug, opts, visited, stats, features, renderData } = scope
  const { cfg, allFiles, fileData } = componentData
  if (!hasHastClass(node, 'transclude')) {
    return
  }
  if (opts.skipTranscludes) {
    return
  }
  const transclude = readTranscludeTarget(node)
  if (!transclude) return
  const { anchorPath, inner, targetSlug: transcludeTarget, url, alias } = transclude
  let blockRef = transclude.blockRef

  let baseViewSlug: FullSlug | undefined
  let page = renderData.bySlug.get(transcludeTarget)
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

  const calendarView =
    page.frontmatter?.layout === 'triathlon' ? calendarEmbedView(blockRef, anchorPath) : undefined
  const curiusDateView = page.slug === 'curius' && blockRef?.startsWith('#')
  if (!calendarView && !curiusDateView) {
    const visitKey = transcludeVisitKey(transclude)
    if (visited.has(visitKey)) return
    visited.add(visitKey)
  }

  // parse metadata to check for collapsed flag
  let transcludeMetadata: Record<string, unknown> | undefined
  const rawMetadata = transclude.rawMetadata
  if (rawMetadata) {
    try {
      transcludeMetadata = JSON.parse(rawMetadata)
    } catch {
      // ignore parsing errors
    }
  }

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

  const { title, headings } = transcludePageOpts

  if (
    page.htmlAst &&
    renderNotebookCellTransclude(
      scope,
      node,
      page.htmlAst,
      transcludeTarget,
      page.frontmatter?.title,
      blockRef,
      inner,
      alias,
      transcludeMetadata,
    )
  ) {
    return
  }

  if (transcludeCurius(node, transcludeTarget, blockRef, inner, alias)) return
  if (transcludeTriathlonCard(scope, node, page, blockRef, anchorPath, inner)) return

  if (blockRef?.startsWith('#^')) {
    // block transclude
    blockRef = blockRef.slice('#^'.length)
    let blockNode = page.blocks?.[blockRef]
    if (blockNode) {
      if (blockNode.tagName === 'li') blockNode = h('ul', blockNode)

      const children = [normalizeHastElement(blockNode, slug, transcludeTarget)]
      if (fileData.frontmatter?.pageLayout !== 'reflection') {
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
        wrapCollapsible(
          node,
          children.filter(c => c !== null) as ElementContent[],
          titleText,
          Boolean(transcludeMetadata.collapsed),
        )
      } else {
        node.children = children.filter(c => c !== null) as ElementContent[]
      }
    }
  } else if (blockRef?.startsWith('#') && page.htmlAst) {
    // header transclude
    blockRef = blockRef.slice(1)
    const bounds = findHeadingSectionBounds(page.htmlAst.children, blockRef)
    if (!bounds) return

    const normalizedSection = page.htmlAst.children
      .slice(bounds.startIdx, bounds.endIdx)
      .filter((child): child is Element => child.type === 'element')
      .filter(child => !isTrailingNoteSection(child))
      .map(child => normalizeHastElement(child, slug, transcludeTarget) as ElementContent)

    const sectionContent = headings ? normalizedSection : pruneLeadingHeading(normalizedSection)

    const children = [
      transcludeAnchor(inner.properties?.href as string, url, alias, title),
      ...sectionContent,
    ]

    if (fileData.frontmatter?.pageLayout !== 'reflection') {
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

    appendTranscludedNotes(root, features, node, page.htmlAst)
  } else if (page.htmlAst || page.bases) {
    let baseTree = page.htmlAst
    let baseRendered: BaseRenderResult | undefined
    let baseView: BaseRenderedView | undefined
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
      baseView =
        baseRendered.views.find(entry => entry.slug === targetSlug) ?? baseRendered.views[0]
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
      ...(baseTree.children as ElementContent[]).map(child =>
        normalizeHastElement(child as Element, slug, transcludeTarget),
      ),
    ]

    if (fileData.frontmatter?.pageLayout !== 'reflection') {
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
