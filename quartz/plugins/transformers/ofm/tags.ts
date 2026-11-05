import type { Link, Nodes, Root } from 'mdast'
import type { Data } from 'vfile'
import { pathToRoot } from '../../../util/path'
import { registerTag } from '../../../util/tags'

export function tagLink(value: string, data: Data): Link {
  const tag = registerTag(data, value)
  const base = data.slug ? pathToRoot(data.slug) : '.'
  return {
    type: 'link',
    url: `${base}/tags/${tag}`,
    data: { tagIdentity: tag, hProperties: { className: ['tag-link'] } },
    children: [{ type: 'text', value }],
  }
}

export function lowerTags(
  tree: Root,
  data: Data,
  isProtected: (node: Nodes) => boolean = () => false,
): void {
  function visit(node: Nodes, inLink = false) {
    if (!('children' in node)) return
    const protectedLabel = inLink || node.type === 'link' || node.type === 'linkReference'
    for (let index = 0; index < node.children.length; index++) {
      const child = node.children[index]
      if (child.type === 'ofmTag') {
        node.children[index] =
          protectedLabel || isProtected(child)
            ? { type: 'text', value: `#${child.value}`, position: child.position }
            : { ...tagLink(child.value, data), position: child.position }
      } else visit(child, protectedLabel)
    }
  }
  visit(tree)
}

declare module 'mdast' {
  interface LinkData {
    tagIdentity?: string
  }
}
