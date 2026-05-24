import { Cite } from '@citation-js/core'
import fs from 'fs'
import path from 'path'
import { JSX } from 'preact'
import type { FrontmatterLink } from '../plugins/transformers/frontmatter'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import style from './styles/seealsoTree.scss'
import '@citation-js/plugin-bibtex'

const MAX_DEPTH = 5
const MAX_CHILDREN_PER_NODE = 5

interface CitationEntry {
  id?: string
  title?: unknown
}

interface CitationLibrary {
  data: CitationEntry[]
}

interface SeeAlsoTreeNode {
  uniqueId: string
  title: string
  href: string
  labelText: string
  isCitation: boolean
  targetSlug?: FullSlug
  children: SeeAlsoTreeNode[]
}

function isCitationLibrary(value: unknown): value is CitationLibrary {
  return typeof value === 'object' && value !== null && 'data' in value && Array.isArray(value.data)
}

function getDisplayTitle(
  slug: FullSlug,
  file: QuartzComponentProps['fileData'] | undefined,
  alias?: string,
): string {
  if (alias && alias.trim().length > 0) {
    return alias.trim()
  }

  const frontmatterTitle = file?.frontmatter?.title
  if (typeof frontmatterTitle === 'string' && frontmatterTitle.length > 0) {
    return frontmatterTitle
  }

  const fragment = slug.split('/').pop() || slug
  return fragment.replace(/\.[^/.]+$/, '').replace(/-/g, ' ')
}

let loadedBib: CitationLibrary | null = null
function getCitationTitle(bibKey: string): string | undefined {
  if (!loadedBib) {
    const bibPath = path.join(process.cwd(), 'content/References.bib')
    if (fs.existsSync(bibPath)) {
      const bibContent = fs.readFileSync(bibPath, 'utf8')
      const parsedBib = new Cite(bibContent, { generateGraph: false })
      if (isCitationLibrary(parsedBib)) {
        loadedBib = parsedBib
      }
    }
  }

  if (loadedBib) {
    const entry = loadedBib.data.find(item => item.id === bibKey)
    return typeof entry?.title === 'string' ? entry.title : undefined
  }
  return undefined
}

export default (() => {
  const SeeAlso: QuartzComponent = ({ fileData, allFiles, displayClass }: QuartzComponentProps) => {
    const fmLinks = fileData.frontmatterLinks as Record<string, FrontmatterLink[]> | undefined

    const rootLinks = fmLinks?.['seealso']

    if (!rootLinks || rootLinks.length === 0) {
      return null
    }

    const slugToFile = new Map<FullSlug, QuartzComponentProps['fileData']>()
    for (const data of allFiles) {
      const slug = data.slug as FullSlug | undefined
      if (slug) {
        slugToFile.set(slug, data)
      }
    }

    const seealsoBySlug = new Map<FullSlug, FrontmatterLink[]>()
    for (const data of allFiles) {
      const slug = data.slug as FullSlug | undefined
      if (!slug) continue
      const links = (data.frontmatterLinks as Record<string, FrontmatterLink[]> | undefined)?.[
        'seealso'
      ]
      if (links && links.length > 0) {
        seealsoBySlug.set(slug, links)
      }
    }

    const currentSlug = fileData.slug as FullSlug | undefined
    if (!currentSlug) {
      return null
    }
    const sourceSlug = currentSlug

    const visited = new Set<string>([sourceSlug])
    const lines: JSX.Element[] = []
    const nbsp = '\u00a0'
    const segmentPad = nbsp.repeat(3)
    const segmentWithBar = `│${segmentPad}`
    const segmentEmpty = `${nbsp}${segmentPad}`

    const formatReadingLabel = (minutes?: number): string => {
      let value = 0
      if (typeof minutes === 'number' && Number.isFinite(minutes) && minutes > 0) {
        value = Math.ceil(minutes)
      }
      if (value < 10) {
        return `[${nbsp}${value}m]`
      }
      return `[${value}m]`
    }

    function buildNode(link: FrontmatterLink, depth: number): SeeAlsoTreeNode | undefined {
      const targetSlug = link.slug
      const isCitation = targetSlug.startsWith('@')
      const uniqueId = targetSlug

      if (visited.has(uniqueId)) {
        return undefined
      }
      visited.add(uniqueId)

      let title = link.alias || targetSlug
      let href = '#'
      let minutes: number | undefined
      let childLinks: FrontmatterLink[] = []

      if (isCitation) {
        const bibKey = targetSlug.substring(1)
        const citeTitle = getCitationTitle(bibKey)
        if (citeTitle) {
          title = citeTitle
        }
        href = `#bib-${bibKey.toLowerCase()}`
      } else {
        const targetFile = slugToFile.get(targetSlug)
        title = getDisplayTitle(targetSlug, targetFile, link.alias)
        href = resolveRelative(sourceSlug, targetSlug)
        minutes = targetFile?.readingTime?.minutes
        const rawChildren = depth < MAX_DEPTH ? (seealsoBySlug.get(targetSlug) ?? []) : []
        childLinks = rawChildren.slice(0, MAX_CHILDREN_PER_NODE)
      }

      return {
        uniqueId,
        title,
        href,
        labelText: isCitation ? '[cite]' : formatReadingLabel(minutes),
        isCitation,
        targetSlug: isCitation ? undefined : targetSlug,
        children: buildNodes(childLinks, depth + 1),
      }
    }

    function buildNodes(links: FrontmatterLink[], depth: number): SeeAlsoTreeNode[] {
      const nodes: SeeAlsoTreeNode[] = []
      for (const link of links.slice(0, MAX_CHILDREN_PER_NODE)) {
        const node = buildNode(link, depth)
        if (node) {
          nodes.push(node)
        }
      }
      return nodes
    }

    const renderNode = (
      node: SeeAlsoTreeNode,
      isLast: boolean,
      ancestorHasSibling: boolean[],
    ): void => {
      const segments: string[] = []
      for (const hasSibling of ancestorHasSibling) {
        segments.push(hasSibling ? segmentWithBar : segmentEmpty)
      }
      const branchGlyph = isLast ? '└── ' : '├── '
      const prefix = segments.join('') + branchGlyph

      const nextAncestors = [...ancestorHasSibling, !isLast]

      lines.push(
        <div class="seealso-tree-line" key={node.uniqueId}>
          <span class="seealso-prefix" aria-hidden="true">
            {prefix}
          </span>
          <span class="seealso-label">{node.labelText}</span>
          <a
            href={node.href}
            class={node.isCitation ? 'seealso-title' : 'seealso-title internal'}
            data-no-popover={node.isCitation}
            data-slug={node.targetSlug}
          >
            {node.title}
          </a>
        </div>,
      )

      if (node.children.length > 0) {
        node.children.forEach((child, idx) =>
          renderNode(child, idx === node.children.length - 1, nextAncestors),
        )
      }
    }

    const topLevel = buildNodes(rootLinks, 0)
    if (topLevel.length === 0) {
      return null
    }
    topLevel.forEach((node, idx) => renderNode(node, idx === topLevel.length - 1, []))

    return (
      <section class={classNames(displayClass, 'seealso-tree', 'main-col')}>
        <div class="seealso-tree-lines">{lines}</div>
      </section>
    )
  }

  SeeAlso.css = style

  return SeeAlso
}) satisfies QuartzComponentConstructor
