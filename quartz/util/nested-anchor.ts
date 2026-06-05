import type { Element, Root } from 'hast'
import { headingRank } from 'hast-util-heading-rank'
import { visit } from 'unist-util-visit'

export interface HeadingRef {
  depth: number
  id: string
  base: string
}

const dedupSuffix = /^(.+)-\d+$/

function baseSlug(id: string): string {
  const match = dedupSuffix.exec(id)
  return match ? match[1] : id
}

export function collectHeadingIndex(tree: Root | Element): HeadingRef[] {
  const headings: HeadingRef[] = []
  visit(tree, 'element', (node: Element) => {
    const depth = headingRank(node)
    if (depth === undefined) return
    const id = node.properties?.id
    if (typeof id !== 'string' || id.length === 0) return
    headings.push({ depth, id, base: baseSlug(id) })
  })
  return headings
}

export function resolveNestedAnchor(
  segments: string[],
  headings: HeadingRef[],
): string | undefined {
  if (segments.length === 0) return undefined
  let start = 0
  let end = headings.length
  let parentDepth = Number.NEGATIVE_INFINITY
  let resolved: string | undefined
  for (const segment of segments) {
    let index = -1
    for (let i = start; i < end; i++) {
      if (headings[i].depth <= parentDepth) break
      if (headings[i].base === segment) {
        index = i
        break
      }
    }
    if (index === -1) return undefined
    resolved = headings[index].id
    parentDepth = headings[index].depth
    start = index + 1
    let subtreeEnd = headings.length
    for (let j = index + 1; j < headings.length; j++) {
      if (headings[j].depth <= parentDepth) {
        subtreeEnd = j
        break
      }
    }
    end = subtreeEnd
  }
  return resolved
}
