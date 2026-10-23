import type { Node } from 'unist'
import matter from 'gray-matter'
import { fromHtml } from 'hast-util-from-html'
import yaml from 'js-yaml'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { visit } from 'unist-util-visit'
import {
  isWikilink,
  wikilink,
  wikilinkFromMarkdown,
} from '../../extensions/micromark-extension-ofm-wikilinks'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { isNotebookParseError } from '../../util/notebook/types'
import { isRecord } from '../../util/type-guards'

export const IMAGE_EXTENSION = /\.(?:jpe?g|png|webp|gif|svg|avif|heic|heif|bmp|tiff?|ico)$/i
export type ImageReferenceSource = 'markdown' | 'html' | 'canvas' | 'base' | 'poster' | 'notebook'
export interface ImageReference {
  target: string
  caption?: string
  source: ImageReferenceSource
}

function markdownReferences(text: string, source: ImageReferenceSource): ImageReference[] {
  const tree = fromMarkdown(text.replace(/%%[\s\S]*?%%/g, ''), {
    extensions: [wikilink()],
    mdastExtensions: [wikilinkFromMarkdown({ obsidian: false })],
  })
  const references: ImageReference[] = []
  const definitions = new Map<string, string>()
  visit(tree, 'definition', node => {
    definitions.set(node.identifier.toLowerCase(), node.url)
  })
  visit(tree, node => {
    if (node.type === 'image')
      references.push({ target: node.url, caption: node.alt ?? undefined, source })
    if (node.type === 'imageReference') {
      const target = definitions.get(node.identifier.toLowerCase())
      if (target) references.push({ target, caption: node.alt ?? undefined, source })
    }
    if (node.type === 'html') {
      visit(fromHtml(node.value, { fragment: true }), 'element', element => {
        const target =
          element.tagName === 'img' ? element.properties.src : element.properties.poster
        if (typeof target === 'string')
          references.push({
            target,
            caption:
              typeof element.properties.alt === 'string' ? element.properties.alt : undefined,
            source: source === 'markdown' ? 'html' : source,
          })
      })
    }
  })
  visit(tree, (node: Node) => {
    if (isWikilink(node) && node.data?.wikilink.embed) {
      const link = node.data.wikilink
      references.push({ target: link.target, caption: link.alias, source })
    }
  })
  return references
}

function valueReferences(value: unknown, source: ImageReferenceSource): ImageReference[] {
  if (Array.isArray(value)) return value.flatMap(item => valueReferences(item, source))
  if (isRecord(value)) return Object.values(value).flatMap(item => valueReferences(item, source))
  if (typeof value !== 'string') return []
  const references = markdownReferences(value, source)
  // Base and poster values commonly contain a bare wikilink rather than an embed.
  const tree = fromMarkdown(value, {
    extensions: [wikilink()],
    mdastExtensions: [wikilinkFromMarkdown({ obsidian: false })],
  })
  visit(tree, (node: Node) => {
    if (!isWikilink(node)) return
    const link = node.data?.wikilink
    if (link && !link.embed) references.push({ target: link.target, caption: link.alias, source })
  })
  if (!references.length && IMAGE_EXTENSION.test(value.split(/[?#]/)[0]))
    references.push({ target: value, source })
  return references
}

export function parseImageReferences(
  sourcePath: string,
  text: string,
  acceptContext?: (frontmatter: Record<string, unknown>) => boolean,
): { frontmatter: Record<string, unknown>; references: ImageReference[] } {
  if (sourcePath.endsWith('.ipynb')) {
    const notebook = parseNotebookDoc(text, sourcePath)
    if (isNotebookParseError(notebook)) throw new Error(notebook.reason)
    if (acceptContext && !acceptContext(notebook.metadata))
      return { frontmatter: notebook.metadata, references: [] }
    return {
      frontmatter: notebook.metadata,
      references: notebook.cells.flatMap(cell =>
        cell.cellType === 'markdown' ? markdownReferences(cell.source, 'notebook') : [],
      ),
    }
  }
  if (sourcePath.endsWith('.canvas')) {
    const data: unknown = JSON.parse(text)
    if (!isRecord(data) || !Array.isArray(data.nodes)) throw new Error('Invalid Canvas nodes')
    const frontmatter = isRecord(data.frontmatter) ? data.frontmatter : {}
    if (acceptContext && !acceptContext(frontmatter)) return { frontmatter, references: [] }
    const references: ImageReference[] = []
    for (const node of data.nodes) {
      if (!isRecord(node)) continue
      if (node.type === 'file' && typeof node.file === 'string')
        references.push({ target: node.file, source: 'canvas' })
      if (node.type === 'text' && typeof node.text === 'string')
        references.push(...markdownReferences(node.text, 'canvas'))
    }
    return { frontmatter, references }
  }
  if (sourcePath.endsWith('.base')) {
    const parsed: unknown = yaml.load(text, { schema: yaml.JSON_SCHEMA })
    if (!isRecord(parsed)) throw new Error('Invalid Base document')
    if (acceptContext && !acceptContext(parsed)) return { frontmatter: parsed, references: [] }
    return { frontmatter: parsed, references: valueReferences(parsed, 'base') }
  }
  const parsed = matter(text, {
    engines: {
      yaml: value => {
        const data: unknown = yaml.load(value, { schema: yaml.JSON_SCHEMA })
        return isRecord(data) ? data : {}
      },
    },
  })
  const frontmatter: Record<string, unknown> = parsed.data
  if (acceptContext && !acceptContext(frontmatter)) return { frontmatter, references: [] }
  const references = markdownReferences(parsed.content, 'markdown')
  for (const key of ['posters', 'poster', 'image', 'cover', 'socialImage'])
    references.push(...valueReferences(frontmatter[key], 'poster'))
  return { frontmatter, references }
}
