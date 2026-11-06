import type { Root } from 'mdast'
import type { VFile } from 'vfile'
import GithubSlugger from 'github-slugger'
import { toString } from 'hast-util-to-string'
import { toHast } from 'mdast-util-to-hast'
import { visit } from 'unist-util-visit'
import type { ReferenceHeading } from './catalog-types'
import { referenceSourcePath } from './context'
import { referenceHastHandler } from './render'

export function indexReferenceDocument(tree: Root, file: VFile): void {
  const slugger = new GithubSlugger()
  const headings: ReferenceHeading[] = []
  const parents: { depth: number; text: string }[] = []
  const blocks = new Set<string>()
  const queries: string[] = []
  visit(tree, node => {
    if (node.type === 'code' && node.lang === 'query') queries.push(node.value)
    if (node.type === 'heading') {
      const rendered = toHast(node, { handlers: { wikilink: referenceHastHandler } })
      const text = rendered ? toString(rendered) : ''
      const existing = node.data?.hProperties?.id
      const generated = slugger.slug(text)
      const anchor = typeof existing === 'string' ? existing : generated
      while (parents.length && parents[parents.length - 1].depth >= node.depth) parents.pop()
      headings.push({ text, anchor, depth: node.depth, path: [...parents.map(p => p.text), text] })
      parents.push({ depth: node.depth, text })
    } else if (node.type === 'text') {
      const match = /\^([-_A-Za-z0-9]+)$/.exec(node.value)
      if (match) blocks.add(match[1])
    }
  })
  file.data.embeddedSearchQueries = queries
  file.data.referenceCatalogInput = {
    path: referenceSourcePath(file.data),
    slug: file.data.slug,
    aliases: file.data.frontmatter?.aliases,
    headings,
    blocks: [...blocks],
    baseViews: file.data.basesConfig?.views.map(view => ({ name: view.name })),
    ...(file.data.frontmatter?.pageLayout === 'latent' ? { latent: true } : {}),
  }
}
