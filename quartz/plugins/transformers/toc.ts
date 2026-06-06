import type { Element as HastElement, Root as HastRoot } from 'hast'
import type { Heading, Root } from 'mdast'
import Slugger from 'github-slugger'
import { headingRank } from 'hast-util-heading-rank'
import { toString } from 'mdast-util-to-string'
import { visit } from 'unist-util-visit'
import type { Wikilink } from '../../extensions/micromark-extension-ofm-wikilinks'
import { isWikilink } from '../../extensions/micromark-extension-ofm-wikilinks'
import { QuartzTransformerPlugin } from '../../types/plugin'

export interface Options {
  maxDepth: 1 | 2 | 3 | 4 | 5 | 6
  minEntries: number
  showByDefault: boolean
}

const defaultOptions: Options = { maxDepth: 3, minEntries: 1, showByDefault: true }
const maxDetailedTocEntries = 50

export interface TocEntry {
  depth: number
  text: string
  slug: string // this is just the anchor (#some-slug), not the canonical slug
}

export interface TocCollection {
  entries: TocEntry[]
  sourceEntries: number
}

export type TocRenderOptions = Pick<Options, 'maxDepth' | 'minEntries'> & {
  display: boolean
  sourceEntries: number
}
type TocHeadingChild = Heading['children'][number] | Wikilink

const slugAnchor = new Slugger()

function normalizeHeadingText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

function normalizeWikilinkText(node: Wikilink): string {
  const { alias, anchor, target } = node.data?.wikilink ?? {}

  const trimmedAlias = alias?.trim()
  if (trimmedAlias) {
    return trimmedAlias
  }

  const trimmedAnchor = anchor?.trim().replace(/^#\^?/, '')
  if (trimmedAnchor) {
    return trimmedAnchor
  }

  const trimmedTarget = target?.trim()
  if (trimmedTarget) {
    return trimmedTarget
  }

  return ''
}

function extractHeadingText(node: Heading): string {
  const children: TocHeadingChild[] = node.children
  const content = children
    .map(child => (isWikilink(child) ? normalizeWikilinkText(child) : toString(child)))
    .join('')
  return normalizeHeadingText(content)
}

function finalizeToc(toc: TocEntry[], highestDepth: number, minEntries: number): TocCollection {
  if (toc.length === 0 || toc.length <= minEntries)
    return { entries: [], sourceEntries: toc.length }

  const visibleToc =
    toc.length > maxDetailedTocEntries ? toc.filter(entry => entry.depth === highestDepth) : toc

  return {
    entries: visibleToc.map(entry => ({ ...entry, depth: entry.depth - highestDepth })),
    sourceEntries: toc.length,
  }
}

export function collectMarkdownTocData(
  tree: Root,
  opts: Pick<Options, 'maxDepth' | 'minEntries'>,
): TocCollection {
  slugAnchor.reset()
  const toc: TocEntry[] = []
  let highestDepth: number = opts.maxDepth

  visit(tree, 'heading', node => {
    if (node.depth > opts.maxDepth) return

    const normalizedText = extractHeadingText(node)
    const text = normalizedText.length > 0 ? normalizedText : toString(node)
    highestDepth = Math.min(highestDepth, node.depth)
    toc.push({ depth: node.depth, text, slug: slugAnchor.slug(text) })
  })

  return finalizeToc(toc, highestDepth, opts.minEntries)
}

export function collectMarkdownToc(
  tree: Root,
  opts: Pick<Options, 'maxDepth' | 'minEntries'>,
): TocEntry[] {
  return collectMarkdownTocData(tree, opts).entries
}

function headingId(node: HastElement): string | undefined {
  const id = node.properties?.id
  return typeof id === 'string' && id.length > 0 ? id : undefined
}

function isHidden(node: HastElement): boolean {
  const ariaHidden = node.properties?.ariaHidden ?? node.properties?.['aria-hidden']
  return ariaHidden === true || ariaHidden === 'true'
}

function hasClass(node: HastElement, name: string): boolean {
  const className = node.properties?.className
  if (typeof className === 'string') {
    return className.split(/\s+/).includes(name)
  }
  return Array.isArray(className) && className.includes(name)
}

function extractHtmlHeadingText(node: HastElement | HastRoot): string {
  const textParts: string[] = []

  for (const child of node.children) {
    if (child.type === 'text') {
      textParts.push(child.value)
    } else if (child.type === 'element' && child.tagName !== 'annotation' && !isHidden(child)) {
      textParts.push(extractHtmlHeadingText(child))
    }
  }

  return normalizeHeadingText(textParts.join(''))
}

export function collectHtmlTocData(
  tree: HastRoot,
  opts: Pick<Options, 'maxDepth' | 'minEntries'>,
): TocCollection {
  slugAnchor.reset()
  const toc: TocEntry[] = []
  let highestDepth: number = opts.maxDepth

  const collect = (node: HastRoot | HastElement, insideBaseEmbed: boolean): void => {
    for (const child of node.children) {
      if (child.type !== 'element') continue

      const childInsideBaseEmbed = insideBaseEmbed || hasClass(child, 'base-embed')
      const depth = headingRank(child)
      if (!childInsideBaseEmbed && depth !== undefined && depth <= opts.maxDepth) {
        const text = extractHtmlHeadingText(child)
        if (text.length > 0) {
          const fallbackSlug = slugAnchor.slug(text)
          const existingId = headingId(child)
          const slug = existingId ?? fallbackSlug
          child.properties = { ...child.properties, id: slug, dataHeadingAlias: text }

          highestDepth = Math.min(highestDepth, depth)
          toc.push({ depth, text, slug })
        }
      }

      if (child.children.length > 0) {
        collect(child, childInsideBaseEmbed)
      }
    }
  }

  collect(tree, false)

  return finalizeToc(toc, highestDepth, opts.minEntries)
}

export function collectHtmlToc(
  tree: HastRoot,
  opts: Pick<Options, 'maxDepth' | 'minEntries'>,
): TocEntry[] {
  return collectHtmlTocData(tree, opts).entries
}

export const TableOfContents: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: 'TableOfContents',
    markdownPlugins() {
      return [
        () => {
          return async (tree: Root, file) => {
            const display = file.data.frontmatter?.enableToc ?? opts.showByDefault
            file.data.tocOptions = {
              maxDepth: opts.maxDepth,
              minEntries: opts.minEntries,
              display: !!display,
              sourceEntries: 0,
            }
            if (display) {
              const toc = collectMarkdownTocData(tree, opts)
              file.data.tocOptions.sourceEntries = toc.sourceEntries
              if (toc.entries.length > 0) file.data.toc = toc.entries
            }
          }
        },
      ]
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    toc: TocEntry[]
    tocOptions?: TocRenderOptions
  }
}
