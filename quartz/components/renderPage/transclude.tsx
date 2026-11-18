import { Element, ElementContent, Root } from 'hast'
import { EXIT, visit } from 'unist-util-visit'
import type { TranscludeOptions } from '../../plugins/transformers/frontmatter'
import type { RenderTreeFeatures, TranscludeScope, TranscludeStats } from './transclude-scope'
import { QuartzPluginData } from '../../plugins/vfile'
import { QuartzComponentProps } from '../../types/component'
import { renderDataFor } from '../../util/ctx'
import { FullSlug, isFullSlug } from '../../util/path'
import { hasHastClass } from '../../util/transclude-props'
import { addRunningHeads, processHeaders, shouldProcessHeaders } from './collapsible-headers'
import { resolveEmbeddedSearch } from './embedded-search'
import { mergeFootnotes, mergeReferences } from './footnotes-citations'
import { transcludeBlockquote } from './transclude-blockquote'
import {
  documentIds,
  expandTransclusionTree,
  placeTransclusionBlocks,
} from './transclude-occurrence'

const DATE_ANCHOR_RE = /#(\d{4}-\d{2}-\d{2})$/

function collectRenderTreeFeatures(root: Root): RenderTreeFeatures {
  const features: RenderTreeFeatures = {
    hasTranscludeBlockquote: false,
    hasReferences: false,
    hasFootnotes: false,
    hasDateAnchor: false,
    hasDataSlugAnchor: false,
  }
  visit(root, 'element', node => {
    if (node.tagName === 'blockquote') {
      if (hasHastClass(node, 'transclude')) {
        features.hasTranscludeBlockquote = true
      }
    } else if (node.tagName === 'section') {
      if (node.properties?.dataReferences != null) {
        features.hasReferences = true
      }
      if (node.properties?.dataFootnotes != null) {
        features.hasFootnotes = true
      }
    } else if (node.tagName === 'a') {
      if (typeof node.properties?.href === 'string' && DATE_ANCHOR_RE.test(node.properties.href)) {
        features.hasDateAnchor = true
      }
      if (typeof (node.properties?.dataSlug ?? node.properties?.['data-slug']) === 'string') {
        features.hasDataSlugAnchor = true
      }
    }

    if (
      features.hasTranscludeBlockquote &&
      features.hasReferences &&
      features.hasFootnotes &&
      features.hasDateAnchor &&
      features.hasDataSlugAnchor
    ) {
      return EXIT
    }
  })
  return features
}

const defaultTranscludeOpts: TranscludeOptions = {
  dynalist: true,
  title: true,
  headings: true,
  skipTranscludes: false,
}

type TransclusionResult = { tree: Root; readingTime: QuartzPluginData['readingTime'] }

export function resolveTransclusions(
  root: Root,
  componentData: QuartzComponentProps,
  { visited }: { visited: Set<FullSlug> },
  userOpts?: Partial<TranscludeOptions>,
): TransclusionResult {
  const { allFiles, fileData } = componentData
  // NOTE: return early these cases, we probably don't want to transclude them anw
  if (fileData.frontmatter?.poem || fileData.frontmatter?.menu) {
    return { tree: root, readingTime: fileData.readingTime }
  }

  // Track total reading stats including transclusions
  const stats: TranscludeStats = {
    words: fileData.readingTime?.words ?? 0,
    minutes: fileData.readingTime?.minutes ?? 0,
    files: new Set([fileData.filePath!]),
  }

  // hierarchy of transclusion: frontmatter > userOpts > defaultOpts
  const slug = fileData.slug as FullSlug
  let opts: TranscludeOptions
  if (userOpts) {
    opts = { ...defaultTranscludeOpts, ...userOpts }
  } else {
    opts = defaultTranscludeOpts
  }

  if (fileData.frontmatter?.transclude) {
    opts = { ...opts, ...fileData.frontmatter?.transclude }
  }

  const { dynalist } = opts
  resolveEmbeddedSearch(root, componentData)
  const features = collectRenderTreeFeatures(root)

  if (features.hasDateAnchor) {
    const renderData = renderDataFor(componentData.ctx, allFiles)
    visit(root, { tagName: 'a' }, node => {
      const props = node.properties
      const href = props?.href
      if (typeof href !== 'string') return
      const date = DATE_ANCHOR_RE.exec(href)?.[1]
      if (!date) return
      const targetSlug = props['dataSlug'] ?? props['data-slug']
      if (typeof targetSlug !== 'string' || !isFullSlug(targetSlug)) return
      const page = renderData.bySlug.get(targetSlug)
      if (page?.frontmatter?.layout !== 'triathlon') return
      props['data-triathlon-date'] = date
    })
  }

  // Canvas and base pages render through their own runtime; the popover reads a summary instead.
  if (features.hasDataSlugAnchor) {
    const previewData = renderDataFor(componentData.ctx, allFiles)
    visit(root, { tagName: 'a' }, node => {
      const props = node.properties
      const targetSlug = props?.['dataSlug'] ?? props?.['data-slug']
      if (typeof targetSlug !== 'string' || !isFullSlug(targetSlug)) return
      const page = previewData.bySlug.get(targetSlug)
      const type = page?.bases ? 'base' : page?.jsonCanvas ? 'canvas' : null
      if (type) props['data-document-preview'] = type
    })
  }

  if (features.hasTranscludeBlockquote) {
    const scope: TranscludeScope = {
      componentData,
      root,
      slug,
      opts,
      visited: new Set<string>(visited),
      stats,
      features,
      renderData: renderDataFor(componentData.ctx, allFiles),
      notebookRuntimeTranscludes: new Map(),
      calendarEmbedCount: 0,
      expansion: { occurrence: 0, remainingNodes: 50000, ids: documentIds(root) },
    }
    expandTransclusionTree(scope, node => transcludeBlockquote(scope, node))
    placeTransclusionBlocks(root)
    resolveEmbeddedSearch(root, componentData)
  }

  // NOTE: handling collapsible nodes
  if (shouldProcessHeaders(fileData, dynalist, slug)) {
    root.children = processHeaders(
      root.children as ElementContent[],
      fileData.frontmatter?.settings?.['heading-toggle'] !== false,
    )
    if (fileData.profile) addRunningHeads(root, fileData.separator)
  }

  // NOTE: We then merge all references and footnotes to final items
  if (features.hasReferences) {
    mergeReferences(root)
  }
  if (features.hasFootnotes) {
    mergeFootnotes(root)
  }

  const readingTime = fileData.readingTime
    ? { ...fileData.readingTime, words: stats.words, minutes: stats.minutes }
    : undefined

  if (slug === 'index') {
    visit(root, { tagName: 'a' }, (node: Element) => {
      node.properties['data-no-popover'] = true
    })
  }

  return { tree: root, readingTime }
}
