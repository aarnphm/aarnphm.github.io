import { Element, ElementContent, Root } from 'hast'
import { headingRank } from 'hast-util-heading-rank'
import { h, s } from 'hastscript'
import type { TranscludeOptions } from '../../plugins/transformers/frontmatter'
import type { RenderData } from '../../util/ctx'
import { i18n } from '../../i18n'
import { QuartzComponentProps } from '../../types/component'
import {
  notebookCellRef,
  notebookCellRuntimeNodes,
  resolveNotebookCell,
} from '../../util/notebook/transclude'
import { FullSlug, normalizeHastElement } from '../../util/path'
import { getHastClassNames } from '../../util/transclude-props'
import { svgOptions } from '../svg'

export type RenderTreeFeatures = {
  hasTranscludeBlockquote: boolean
  hasReferences: boolean
  hasFootnotes: boolean
  hasNestedAnchor: boolean
  hasDateAnchor: boolean
  hasDataSlugAnchor: boolean
}

export interface TranscludeStats {
  words: number
  minutes: number
  files: Set<string>
}

/** State one resolveTransclusions call shares with its blockquote phase. */
export interface TranscludeScope {
  componentData: QuartzComponentProps
  root: Root
  slug: FullSlug
  opts: TranscludeOptions
  visited: Set<FullSlug>
  stats: TranscludeStats
  features: RenderTreeFeatures
  renderData: RenderData
  notebookRuntimeTranscludes: Map<string, number>
  calendarEmbedCount: number
}

export function pruneLeadingHeading(nodes: ElementContent[]): ElementContent[] {
  let removed = false
  return nodes.filter(node => {
    if (
      !removed &&
      node &&
      typeof node === 'object' &&
      (node as Element).type === 'element' &&
      headingRank(node as Element)
    ) {
      removed = true
      return false
    }
    return true
  })
}

export function transcludeAnchor(
  href: string,
  url: string,
  description: string | null,
  title: boolean,
): Element | null {
  if (!title) return null

  const [parent, ...children] = url.split('/')
  const truncated = children.length > 2 ? `${parent}/.../${children[children.length - 1]}` : url
  const metadata: Element[] = [
    h('li', { style: 'font-style: italic; color: var(--gray);' }, [
      { type: 'text', value: `url: ${truncated}` },
    ]),
  ]

  if (description) {
    metadata.push(
      h('li', [
        h('span', { style: 'text-decoration: underline;' }, [
          { type: 'text', value: `description` },
        ]),
        { type: 'text', value: `: ${description}` },
      ]),
    )
  }

  return h('.transclude-ref', { 'data-href': href }, [
    h('ul.metadata', metadata),
    h(
      'button.transclude-title-link',
      { type: 'button', ariaLabel: 'Go to original link' },
      s(
        'svg',
        {
          ...svgOptions,
          fill: 'none',
          stroke: 'currentColor',
          strokewidth: '2',
          class: 'blockquote-link',
        },
        [s('use', { href: '#github-anchor' })],
      ),
    ),
  ])
}

/**
 * wrap transclude content in collapsible structure.
 * creates a title bar with fold button and content area.
 */
export function wrapCollapsible(
  node: Element,
  children: ElementContent[],
  titleText: string,
  collapsed: boolean,
): void {
  const foldButton = h(
    'span.transclude-fold',
    {
      role: 'button',
      type: 'button',
      ariaExpanded: !collapsed,
      ariaLabel: 'Toggle transclude visibility',
    },
    [
      s(
        'svg',
        {
          ...svgOptions,
          fill: 'none',
          stroke: 'currentColor',
          strokeWidth: '2',
          class: 'fold-icon',
        },
        [s('use', { href: collapsed ? '#chevron-right' : '#chevron-down' })],
      ),
    ],
  )

  const titleEl = h('.transclude-title', [
    foldButton,
    h('span.transclude-title-text', [{ type: 'text', value: titleText }]),
  ])

  const contentEl = h('.transclude-content', [h('div', children)])

  const classNames = getHastClassNames(node)
  classNames.push('transclude-collapsible')
  if (collapsed) {
    classNames.push('is-collapsed')
  }
  node.properties = { ...node.properties, className: classNames }
  delete node.properties.class

  node.children = [titleEl, contentEl]
}

function notebookRuntimeTranscludeCount(
  { slug, notebookRuntimeTranscludes }: TranscludeScope,
  transcludeTarget: FullSlug,
  cellId: string,
): number {
  const key = `${slug}\u0000${transcludeTarget}\u0000${cellId}`
  const count = notebookRuntimeTranscludes.get(key) ?? 0
  notebookRuntimeTranscludes.set(key, count + 1)
  return count
}

export function renderNotebookCellTransclude(
  scope: TranscludeScope,
  node: Element,
  pageTree: Root,
  transcludeTarget: FullSlug,
  pageTitle: string | undefined,
  blockRef: string | undefined,
  inner: Element,
  alias: string,
  transcludeMetadata: Record<string, unknown> | undefined,
): boolean {
  const { slug } = scope
  const { cfg, fileData } = scope.componentData
  const ref = notebookCellRef(blockRef)
  if (!ref) return false
  const resolved = resolveNotebookCell(pageTree, ref)
  if (!resolved) return false
  const { id: cellId, frame: cell } = resolved

  const children: (ElementContent | null)[] = [
    ...notebookCellRuntimeNodes(pageTree, {
      slug,
      transcludeTarget,
      cellId,
      count: notebookRuntimeTranscludeCount(scope, transcludeTarget, cellId),
    }),
    normalizeHastElement(cell, slug, transcludeTarget) as ElementContent,
  ]
  if (fileData.frontmatter?.pageLayout !== 'reflection') {
    children.push(
      h('a', { href: inner.properties?.href, class: 'internal transclude-src' }, [
        {
          type: 'text',
          value: pageTitle ?? i18n(cfg.locale).components.transcludes.linkToOriginal,
        },
      ]),
    )
  }

  const validChildren = children.filter(c => c !== null) as ElementContent[]
  if (transcludeMetadata && 'collapsed' in transcludeMetadata) {
    wrapCollapsible(
      node,
      validChildren,
      alias || `Cell: ${cellId}`,
      Boolean(transcludeMetadata.collapsed),
    )
  } else {
    node.children = validChildren
  }
  return true
}
