import { Element, ElementContent, Root } from 'hast'
import { EXIT, visit } from 'unist-util-visit'
import type { TranscludeOptions } from '../../plugins/transformers/frontmatter'
import type { RenderTreeFeatures, TranscludeScope, TranscludeStats } from './transclude-scope'
import { QuartzPluginData } from '../../plugins/vfile'
import { QuartzComponentProps } from '../../types/component'
import { renderDataFor } from '../../util/ctx'
import { collectHeadingIndex, resolveNestedAnchor } from '../../util/nested-anchor'
import { FullSlug, isFullSlug } from '../../util/path'
import { hasHastClass } from '../../util/transclude-props'
import { processHeaders, shouldProcessHeaders } from './collapsible-headers'
import { mergeFootnotes, mergeReferences } from './footnotes-citations'
import { transcludeBlockquote } from './transclude-blockquote'

const DATE_ANCHOR_RE = /#(\d{4}-\d{2}-\d{2})$/

function collectRenderTreeFeatures(root: Root): RenderTreeFeatures {
  const features: RenderTreeFeatures = {
    hasTranscludeBlockquote: false,
    hasReferences: false,
    hasFootnotes: false,
    hasNestedAnchor: false,
    hasDateAnchor: false,
    hasDataSlugAnchor: false,
  }
  visit(root, 'element', node => {
    if (node.tagName === 'blockquote') {
      if (hasHastClass(node, 'transclude')) {
        features.hasTranscludeBlockquote = true
      }
    } else if (node.tagName === 'section') {
      if (node.properties?.dataReferences === '') {
        features.hasReferences = true
      }
      if (node.properties?.dataFootnotes === '') {
        features.hasFootnotes = true
      }
    } else if (node.tagName === 'a') {
      if (typeof node.properties?.dataAnchorPath === 'string') {
        features.hasNestedAnchor = true
      }
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
      features.hasNestedAnchor &&
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
  const features = collectRenderTreeFeatures(root)

  if (features.hasNestedAnchor) {
    const renderData = renderDataFor(componentData.ctx, allFiles)
    const headingCache = new Map<string, ReturnType<typeof collectHeadingIndex>>()
    visit(root, { tagName: 'a' }, node => {
      const pathAttr = node.properties?.dataAnchorPath
      const targetAttr = node.properties?.dataAnchorTarget
      if (node.properties) {
        delete node.properties.dataAnchorPath
        delete node.properties.dataAnchorTarget
      }
      if (typeof pathAttr !== 'string') return
      let segments: unknown
      try {
        segments = JSON.parse(pathAttr)
      } catch {
        return
      }
      if (!Array.isArray(segments) || segments.length < 2) return

      const cacheKey = typeof targetAttr === 'string' ? targetAttr : ''
      let headings = headingCache.get(cacheKey)
      if (!headings) {
        let tree: Root | undefined
        if (cacheKey) {
          const targetSlug = cacheKey.replace(/^\//, '')
          tree = isFullSlug(targetSlug) ? renderData.bySlug.get(targetSlug)?.htmlAst : undefined
        } else {
          tree = root
        }
        if (!tree) return
        headings = collectHeadingIndex(tree)
        headingCache.set(cacheKey, headings)
      }

      const resolved = resolveNestedAnchor(segments as string[], headings)
      if (!resolved) return
      const href = node.properties?.href
      if (typeof href === 'string') {
        node.properties.href = href.replace(/#.*$/, '') + '#' + resolved
      }
    })
  }

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
      visited,
      stats,
      features,
      renderData: renderDataFor(componentData.ctx, allFiles),
      notebookRuntimeTranscludes: new Map(),
      calendarEmbedCount: 0,
    }
    visit(root, { tagName: 'blockquote' }, node => {
      transcludeBlockquote(scope, node)
    })
  }

  // NOTE: handling collapsible nodes
  if (shouldProcessHeaders(fileData, dynalist, slug)) {
    root.children = processHeaders(root.children as ElementContent[])
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
