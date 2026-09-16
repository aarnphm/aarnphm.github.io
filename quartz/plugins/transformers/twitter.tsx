import { Element } from 'hast'
import { Link, Paragraph, PhrasingContent } from 'mdast'
import { visit } from 'unist-util-visit'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { parseTwitterPostUrl } from '../../util/twitter'
import { fetchTwitterPost } from '../../util/twitter-content'
import { wikiTextTransform } from './ofm'

export function filterEmbedTwitter(node: Element): boolean {
  const href = node.properties.href
  if (href === undefined || typeof href !== 'string') return false
  return node.children.length !== 0 && parseTwitterPostUrl(href) !== null
}

const isWhitespaceNode = (node: PhrasingContent) => {
  if (node.type !== 'text') return false
  return node.value.trim() === ''
}

const isNakedLink = (parent: Paragraph, child: Link) => {
  const meaningfulChildren = parent.children.filter(node => !isWhitespaceNode(node))
  if (meaningfulChildren.length !== 1 || meaningfulChildren[0] !== child) {
    return false
  }

  const linkText = child.children
    .map(node => (node.type === 'text' ? node.value : ''))
    .join('')
    .trim()

  return linkText.length === 0 || linkText === child.url
}

export const Twitter: QuartzTransformerPlugin = () => ({
  name: 'Twitter',
  textTransform(_, src) {
    src = wikiTextTransform(src)

    return src
  },
  markdownPlugins() {
    return [
      () => async (tree, file) => {
        const fileData = file.data
        if (fileData.slug === 'are.na') return

        const promises: Promise<void>[] = []

        visit(tree, 'paragraph', (node: Paragraph, index, parent) => {
          if (!parent || index === undefined) return
          const link = node.children.find(child => child.type === 'link')
          if (!link || !parseTwitterPostUrl(link.url) || !isNakedLink(node, link)) return
          promises.push(
            fetchTwitterPost(link.url).then(value => {
              parent.children.splice(index, 1, { type: 'html', value })
            }),
          )
        })

        if (promises.length > 0) await Promise.all(promises)
      },
    ]
  },
})
