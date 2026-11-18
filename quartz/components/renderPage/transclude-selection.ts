import type { Element, ElementContent, Root, RootContent } from 'hast'
import { h } from 'hastscript'
import type { QuartzPluginData } from '../../plugins/vfile'
import { clone } from '../../util/clone'
import { findHeadingSectionBounds } from '../../util/transclude-props'
import { isTrailingNoteSection } from './footnotes-citations'

function elementPath(root: Root, target: Element): Element[] | undefined {
  function find(node: RootContent, ancestors: Element[]): Element[] | undefined {
    if (node.type !== 'element') return undefined
    const path = [...ancestors, node]
    if (node === target || (target.properties.id && node.properties.id === target.properties.id))
      return path
    for (const child of node.children) {
      const found = find(child, path)
      if (found) return found
    }
  }
  for (const child of root.children) {
    const found = find(child, [])
    if (found) return found
  }
}

function completeBlock(page: QuartzPluginData, id: string): Element | undefined {
  const block =
    page.blocks?.[id] ??
    Object.entries(page.blocks ?? {}).find(([key]) => key.toLowerCase() === id.toLowerCase())?.[1]
  if (!block || !page.htmlAst) return block
  const path = elementPath(page.htmlAst, block)
  if (!path) return block
  const itemIndex = path.findLastIndex(node => node.tagName === 'li')
  if (itemIndex === -1) return block
  const item = path[itemIndex]
  const list = path[itemIndex - 1]
  if (!list || (list.tagName !== 'ul' && list.tagName !== 'ol')) return item
  const properties = { ...list.properties }
  delete properties.id
  if (list.tagName === 'ol') {
    const start = Number(list.properties.start ?? 1)
    const ordinal = list.children
      .filter(child => child.type === 'element' && child.tagName === 'li')
      .indexOf(item)
    properties.start = (Number.isFinite(start) ? start : 1) + Math.max(0, ordinal)
  }
  return h(list.tagName, properties, [item])
}

/** Definitions are selected before rebasing so references retain their source-document identity. */
function requiredNotes(nodes: ElementContent[], tree: Root): Element[] {
  const sections = tree.children.filter(
    (node): node is Element => node.type === 'element' && isTrailingNoteSection(node),
  )
  const entries = new Map<string, Element>()
  for (const section of sections) {
    const list = section.children.find(
      (node): node is Element =>
        node.type === 'element' && (node.tagName === 'ol' || node.tagName === 'ul'),
    )
    for (const entry of list?.children ?? []) {
      if (entry.type === 'element' && typeof entry.properties.id === 'string')
        entries.set(entry.properties.id, entry)
    }
  }
  const selected = new Set<string>()
  function follow(node: ElementContent): void {
    if (node.type !== 'element') return
    const href = node.properties.href
    if (typeof href === 'string' && href.startsWith('#')) {
      const id = href.slice(1)
      const entry = entries.get(id)
      if (entry && !selected.has(id)) {
        selected.add(id)
        follow(entry)
      }
    }
    for (const child of node.children) follow(child)
  }
  for (const node of nodes) follow(node)
  return sections.flatMap(section => {
    const copy = clone(section)
    const list = copy.children.find(
      (node): node is Element =>
        node.type === 'element' && (node.tagName === 'ol' || node.tagName === 'ul'),
    )
    if (!list) return []
    list.children = list.children.filter(
      node =>
        node.type === 'element' &&
        typeof node.properties.id === 'string' &&
        selected.has(node.properties.id),
    )
    return list.children.length ? [copy] : []
  })
}

export function selectTransclusion(
  page: QuartzPluginData,
  blockRef: string | undefined,
): ElementContent[] | undefined {
  const tree = page.htmlAst
  if (!tree) return undefined
  let nodes: ElementContent[]
  if (blockRef?.startsWith('#^')) {
    const block = completeBlock(page, blockRef.slice(2))
    if (!block) return undefined
    nodes = [block]
  } else if (blockRef?.startsWith('#')) {
    const bounds = findHeadingSectionBounds(tree.children, blockRef.slice(1))
    if (!bounds) return undefined
    nodes = tree.children
      .slice(bounds.startIdx, bounds.endIdx)
      .filter(
        (node): node is ElementContent =>
          node.type !== 'doctype' && (node.type !== 'element' || !isTrailingNoteSection(node)),
      )
  } else {
    return clone(tree.children.filter((node): node is ElementContent => node.type !== 'doctype'))
  }
  return [...clone(nodes), ...requiredNotes(nodes, tree)]
}
