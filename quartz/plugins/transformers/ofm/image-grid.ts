import type { Root as HtmlRoot } from 'hast'
import type { Root } from 'mdast'
import { visit } from 'unist-util-visit'
import { isWikilink } from '../../../extensions/micromark-extension-ofm-wikilinks/node'

export function markImageGridCandidates(tree: Root): void {
  visit(tree, 'paragraph', node => {
    const media = node.children.filter(child => child.type === 'image' || isWikilink(child))
    if (
      media.length < 2 ||
      !node.children.every(
        child =>
          child.type === 'image' ||
          isWikilink(child) ||
          (child.type === 'text' && child.value.trim() === ''),
      )
    )
      return
    node.data = {
      ...node.data,
      hProperties: { ...node.data?.hProperties, 'data-reference-grid': true },
    }
  })
}

/** Group resolved image placeholders before the media owners turn them into figures or viewers. */
export function groupImageGridHast(tree: HtmlRoot): void {
  visit(tree, 'element', node => {
    if (
      node.properties.dataReferenceGrid === undefined &&
      node.properties['data-reference-grid'] === undefined
    )
      return
    delete node.properties.dataReferenceGrid
    delete node.properties['data-reference-grid']
    const images = node.children.filter(
      child => child.type === 'element' && child.tagName === 'img',
    )
    if (
      node.tagName !== 'p' ||
      images.length < 2 ||
      !node.children.every(
        child =>
          (child.type === 'element' && child.tagName === 'img') ||
          (child.type === 'text' && child.value.trim() === ''),
      )
    )
      return
    node.tagName = 'div'
    node.properties.className = ['image-grid']
  })
}
