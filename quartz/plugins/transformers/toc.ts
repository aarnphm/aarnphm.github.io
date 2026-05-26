import type { Element as HastElement, Root as HastRoot } from 'hast'
import type { Heading, Root } from 'mdast'
import Slugger from 'github-slugger'
import { headingRank } from 'hast-util-heading-rank'
import { toText } from 'hast-util-to-text'
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

export interface TocEntry {
  depth: number
  text: string
  slug: string // this is just the anchor (#some-slug), not the canonical slug
}

export type TocRenderOptions = Pick<Options, 'maxDepth' | 'minEntries'> & { display: boolean }
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

function finalizeToc(toc: TocEntry[], highestDepth: number, minEntries: number): TocEntry[] {
  if (toc.length === 0 || toc.length <= minEntries) return []
  return toc.map(entry => ({ ...entry, depth: entry.depth - highestDepth }))
}

export function collectMarkdownToc(
  tree: Root,
  opts: Pick<Options, 'maxDepth' | 'minEntries'>,
): TocEntry[] {
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

function headingId(node: HastElement): string | undefined {
  const id = node.properties?.id
  return typeof id === 'string' && id.length > 0 ? id : undefined
}

export function collectHtmlToc(
  tree: HastRoot,
  opts: Pick<Options, 'maxDepth' | 'minEntries'>,
): TocEntry[] {
  slugAnchor.reset()
  const toc: TocEntry[] = []
  let highestDepth: number = opts.maxDepth

  visit(tree, 'element', node => {
    const depth = headingRank(node)
    if (depth === undefined || depth > opts.maxDepth) return

    const text = normalizeHeadingText(toText(node))
    if (text.length === 0) return

    const fallbackSlug = slugAnchor.slug(text)
    const slug = headingId(node) ?? fallbackSlug
    if (!headingId(node)) {
      node.properties = { ...node.properties, id: slug }
    }

    highestDepth = Math.min(highestDepth, depth)
    toc.push({ depth, text, slug })
  })

  return finalizeToc(toc, highestDepth, opts.minEntries)
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
            }
            if (display) {
              const toc = collectMarkdownToc(tree, opts)
              if (toc.length > 0) file.data.toc = toc
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
