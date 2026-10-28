import { Element } from 'hast'
import { headingRank } from 'hast-util-heading-rank'
import { h } from 'hastscript'
import rehypeSlug from 'rehype-slug'
import remarkGfm from 'remark-gfm'
import smartypants from 'remark-smartypants'
import { visit } from 'unist-util-visit'
import { QuartzTransformerPlugin } from '../../types/plugin'

export const checkFootnoteRef = ({ type, tagName, properties }: Element) =>
  type === 'element' && tagName === 'a' && Boolean(properties) && properties.dataFootnoteRef != null

export const checkFootnoteSection = ({ type, tagName, properties }: Element) =>
  type === 'element' && tagName === 'section' && properties.dataFootnotes == ''

export const GitHubFlavoredMarkdown: QuartzTransformerPlugin = () => ({
  name: 'GitHubFlavoredMarkdown',
  markdownPlugins: () => [remarkGfm, smartypants],
  htmlPlugins: () => [
    rehypeSlug,
    () => tree => {
      visit(tree, node => {
        if (headingRank(node) !== undefined) {
          if (node.properties.id === 'footnote-label') {
            node.children = [{ type: 'text', value: 'remarque' }]
          }
          node.children = [h('span.highlight-span', node.children)]
        }
      })
    },
    () => tree => {
      visit(tree, node => {
        if (checkFootnoteSection(node as Element)) {
          const className = Array.isArray(node.properties.className)
            ? node.properties.className
            : (node.properties.className = [])
          className.push('main-col')
        }
      })
    },
  ],
})
